from torch.ao import sparsity
import torch
import pytorch_lightning as pl
import torch.nn as nn
from typing import List
from torch.ao.sparsity.experimental.data_sparsifier.lightning.callbacks.data_sparsity import (
    PostTrainingDataSparsity,
    TrainingAwareDataSparsity
)
from torch.ao.sparsity.experimental.data_sparsifier.lightning.callbacks._data_sparstity_utils import _get_valid_name
from torch.ao.sparsity.experimental.data_sparsifier.base_data_sparsifier import SUPPORTED_TYPES
import warnings
import math
from torch.nn.utils.parametrize import is_parametrized


class DummyModel(nn.Module):
    def __init__(self, iC: int, oC: List[int]):
        super().__init__()
        self.linears = nn.Sequential()
        self.emb = nn.Embedding(1024, 32)
        self.emb_bag = nn.EmbeddingBag(1024, 32)
        i = iC
        for idx, c in enumerate(oC):
            self.linears.append(nn.Linear(i, c, bias=False))
            if idx < len(oC) - 1:
                self.linears.append(nn.ReLU())
            i = c


class DummyLightningModule(pl.LightningModule):
    def __init__(self, iC: int, oC: List[int]):
        super().__init__()
        self.model = DummyModel(iC, oC)

    def forward(self):
        pass


class StepSLScheduler(sparsity.BaseDataScheduler):
    """The sparsity param of each data group is multiplied by gamma every step_size epochs.
    """
    def __init__(self, data_sparsifier, schedule_param='sparsity_level',
                 step_size=1, gamma=2, last_epoch=-1, verbose=False):

        self.gamma = gamma
        self.step_size = step_size
        super().__init__(data_sparsifier, schedule_param, last_epoch, verbose)

    def get_schedule_param(self):
        if not self._get_sp_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)
        data_groups = self.data_sparsifier.data_groups
        if (self.last_epoch == 0) or (self.last_epoch % self.step_size != 0):
            return {name: config[self.schedule_param] for name, config in data_groups.items()}

        return {name: config[self.schedule_param] * self.gamma for name, config in data_groups.items()}


class TestPostTrainingDataNormSparsifierCallback:
    def _get_callback(self, sparsifier_args):
        callback = PostTrainingDataSparsity(data_sparsifier_type=sparsity.DataNormSparsifier, data_sparsifier_args=sparsifier_args)
        return callback

    def _get_pl_module(self):
        pl_module = DummyLightningModule(100, [128, 256, 16])
        return pl_module

    def check_on_validation_start(self, pl_module, callback, sparsifier_args):
        callback.on_validation_start(42, pl_module)  # 42 is a dummy value as trainer not used

        # check sparsifier config
        for key, value in sparsifier_args.items():
            assert callback.data_sparsifier.defaults[key] == value

        # assert that the model is correctly attached to the sparsifier
        for name, param in pl_module.model.named_parameters():
            valid_name = _get_valid_name(name)
            if type(param) not in SUPPORTED_TYPES:
                assert valid_name not in callback.data_sparsifier.state
                assert valid_name not in callback.data_sparsifier.data_groups
                continue
            assert valid_name in callback.data_sparsifier.data_groups
            assert valid_name in callback.data_sparsifier.state

            mask = callback.data_sparsifier.get_mask(name=valid_name)

            # assert that some level of sparsity is achieved
            assert (1.0 - mask.float().mean()) > 0.0

            # make sure that non-zero values in data after squash mask are equal to original values
            sparsified_data = callback.data_sparsifier.get_data(name=valid_name, return_original=False)
            assert torch.all(sparsified_data[sparsified_data != 0] == param[sparsified_data != 0])

    def run_all_checks(self, sparsifier_args):
        pl_module = self._get_pl_module()
        callback = self._get_callback(sparsifier_args)

        self.check_on_validation_start(pl_module, callback, sparsifier_args)


class TestTrainingAwareDataNormSparsifierCallback:
    """Class to test in-training version of lightning callback
    Simulates model training and makes sure that each hook is doing what is expected
    """
    def _get_callback(self, sparsifier_args, scheduler_args):
        callback = TrainingAwareDataSparsity(
            data_sparsifier_type=sparsity.DataNormSparsifier,
            data_sparsifier_args=sparsifier_args,
            data_scheduler_type=StepSLScheduler,
            data_scheduler_args=scheduler_args
        )
        return callback

    def _get_pl_module(self):
        pl_module = DummyLightningModule(100, [128, 256, 16])
        return pl_module

    def check_on_train_start(self, pl_module, callback, sparsifier_args, scheduler_args):

        callback.on_train_start(42, pl_module)  # 42 is a dummy value

        # sparsifier and scheduler instantiated
        assert callback.data_scheduler is not None and callback.data_sparsifier is not None

        for key, value in sparsifier_args.items():
            callback.data_sparsifier.defaults[key] == value

        for key, value in scheduler_args.items():
            assert getattr(callback.data_scheduler, key) == value

    def _simulate_update_param_model(self, pl_module):
        """This function might not be needed as the model is being copied
        during train_epoch_end() but good to have if things change in the future
        """
        for _, param in pl_module.model.named_parameters():
            param.data = param + 1

    def check_on_train_epoch_start(self, pl_module, callback):
        callback.on_train_epoch_start(42, pl_module)
        if callback.data_sparsifier_state_dict is None:
            return

        data_sparsifier_state_dict = callback.data_sparsifier.state_dict()

        # compare container objects
        container_obj1 = data_sparsifier_state_dict['_container']
        container_obj2 = callback.data_sparsifier_state_dict['_container']
        assert len(container_obj1) == len(container_obj2)
        for key, value in container_obj2.items():
            assert key in container_obj1
            assert torch.all(value == container_obj1[key])

        # compare state objects
        state_obj1 = data_sparsifier_state_dict['state']
        state_obj2 = callback.data_sparsifier_state_dict['state']
        assert len(state_obj1) == len(state_obj2)
        for key, value in state_obj2.items():
            assert key in state_obj1
            assert 'mask' in value and 'mask' in state_obj1[key]
            assert torch.all(value['mask'] == state_obj1[key]['mask'])

        # compare data_groups dict
        data_grp1 = data_sparsifier_state_dict['data_groups']
        data_grp2 = callback.data_sparsifier_state_dict['data_groups']
        assert len(data_grp1) == len(data_grp2)
        for key, value in data_grp2.items():
            assert key in data_grp1
            assert value == data_grp1[key]

    def check_on_train_epoch_end(self, pl_module, callback):
        callback.on_train_epoch_end(42, pl_module)
        data_scheduler = callback.data_scheduler
        base_sl = data_scheduler.base_param

        for name, _ in pl_module.model.named_parameters():
            valid_name = _get_valid_name(name)
            mask = callback.data_sparsifier.get_mask(name=valid_name)
            assert (1.0 - mask.float().mean()) > 0

            last_sl = data_scheduler.get_last_param()
            last_epoch = data_scheduler.last_epoch

            log_last_sl = math.log(last_sl[valid_name])
            log_actual_sl = math.log(base_sl[valid_name] * (data_scheduler.gamma ** last_epoch))
            assert log_last_sl == log_actual_sl

    def check_on_train_end(self, pl_module, callback):
        callback.on_train_end(42, pl_module)

        # check that the masks have been squashed
        for name, _ in pl_module.model.named_parameters():
            valid_name = _get_valid_name(name)
            assert not is_parametrized(callback.data_sparsifier._continer, valid_name)

    def run_all_checks(self, sparsifier_args, scheduler_args):
        pl_module = self._get_pl_module()
        callback = self._get_callback(sparsifier_args, scheduler_args)

        self.check_on_train_start(pl_module, callback, sparsifier_args, scheduler_args)

        num_epochs = 5
        for _ in range(0, num_epochs):
            self.check_on_train_epoch_start(pl_module, callback)
            self._simulate_update_param_model(pl_module)
            self.check_on_train_epoch_end(pl_module, callback)


if __name__ == "__main__":
    callback_tester = TestPostTrainingDataNormSparsifierCallback()
    sparsifier_args = {
        'sparsity_level': 0.5,
        'sparse_block_shape': (1, 4),
        'zeros_per_block': 4
    }
    callback_tester.run_all_checks(sparsifier_args)

    callback_tester_ta = TestTrainingAwareDataNormSparsifierCallback()
    scheduler_args = {
        'gamma': 2,
        'step_size': 1
    }

    callback_tester_ta.run_all_checks(sparsifier_args, scheduler_args)
