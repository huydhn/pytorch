from copy import deepcopy
import torch
from dataclasses import dataclass, field
from typing import Any, Optional, Dict
import pytorch_lightning as pl

from ._data_sparstity_utils import _create_data_sparsifier, _attach_model_to_data_sparsifier, _log_sparsified_level


@dataclass
class _DataSparsity(pl.callbacks.Callback):
    data_sparsifier_type: Any
    data_sparsifier_args: Dict  # this the arguments for sparsifier_type (except data_list)
    data_sparsifier: Any = field(init=False, default=None)
    sparsified: Optional[torch.nn.Module] = field(init=False, default=None)

    def __post_init__(self):
        pass


@dataclass
class PostTrainingDataSparsity(_DataSparsity):
    """Lightning callback that enables post-training sparsity.

    This callback aims to sparsify the model inside lightning module after training.
    **Note that the model is copied and then sparsified, so the existing model is not modified**

    The sparsified model can be used for comparison and can be accessed using
        <callback_obj>.sparsified

    Args:
        data_sparsifier_type (some implemented class of BaseDataSparsifier)
            The data sparsifier object of this type is created when the
            training starts.
            Note: Objects should not be passed in here as they are created
            once the training completes.

        data_sparsifier_args (Dict)
            Dictionary of args to be passed to the data sparsifier.
            Note: data_list arg should be ignored

    Hooks implemented:
        on_validation_start()
            1. copies the model and attaches it to the sparsifier
            2. sparsier step() is called
            3. squashes the mask()
    """

    def on_validation_start(self, trainer, pl_module) -> None:
        copied_model = deepcopy(pl_module.model).eval()  # do we really need to put in cpu?
        self.sparsified = copied_model
        self.data_sparsifier = _create_data_sparsifier(self.data_sparsifier_args, self.data_sparsifier_type)

        _attach_model_to_data_sparsifier(self.sparsified, self.data_sparsifier)

        self.data_sparsifier.step()

        self.data_sparsifier.squash_mask()  # currently squashes params for all mask

        _log_sparsified_level(self.sparsified, self.data_sparsifier)

# Nothing to do on_validation_end()
