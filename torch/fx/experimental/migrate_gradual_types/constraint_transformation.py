# mypy: ignore-errors

import itertools
from torch.fx.experimental.migrate_gradual_types.constraint_generator import BinConstraintT
from torch.fx.experimental.migrate_gradual_types.constraint import T, BinConstraintD, Conj
from torch.fx.experimental.migrate_gradual_types.constraint import Disj, TGreatestUpperBound
from torch.fx.experimental.migrate_gradual_types.constraint import DGreatestUpperBound
from torch.fx.experimental.migrate_gradual_types.constraint import CalcConv, CalcMaxPool
from torch.fx.experimental.migrate_gradual_types.constraint import CalcProduct, CanReshape
from torch.fx.experimental.migrate_gradual_types.constraint import ApplyBroadcasting, Prod, F
from torch.fx.experimental.migrate_gradual_types.operation import op_eq, op_precision, op_leq, op_matching
from torch.fx.experimental.migrate_gradual_types.operation import op_consistency, op_neq
from torch.fx.experimental.migrate_gradual_types.operation import op_mul, op_add, op_sub, op_div, op_mod
from torch.fx.experimental.migrate_gradual_types.util import gen_tensor_dims, gen_nat_constraints, gen_dvar
from torch.fx.tensor_type import TensorType, Dyn


# run this as the first transformation step till you get a fixed point
# handles: consistency, matching, precision, GLB, leq, broadcasting, reshape, convolution..
def transform_constraint(constraint, counter):
    # tensor constraints
    if isinstance(constraint, BinConstraintT):
        # precision constraints
        if constraint.op == op_precision:
            if constraint.lhs == Dyn:
                return T(), counter
            elif isinstance(constraint.lhs, TensorType):
                is_fully_static = all([d != Dyn for d in constraint.lhs.__args__])
                if is_fully_static:
                    return BinConstraintT(constraint.lhs, constraint.rhs, op_eq), counter
                else:
                    new_dims = []

                    for _ in range(len(constraint.lhs.__args__)):
                        dim, counter = gen_dvar(counter)
                        new_dims.append(dim)

                    new_dim_constraints = [BinConstraintD(old_dim, new_dim, op_precision) for
                                           new_dim, old_dim in zip(new_dims, constraint.lhs.__args__)] + \
                                          [BinConstraintT(constraint.rhs, TensorType(new_dims), op_eq)] + \
                                          [BinConstraintD(1, new_dim, op_leq) for
                                           new_dim in new_dims]
                    return Conj(new_dim_constraints), counter

        # matching
        elif constraint.op == op_matching:
            assert isinstance(constraint.rhs, TensorType)
            d1 = constraint.rhs.__args__[0]
            d2 = constraint.rhs.__args__[1]
            d3 = constraint.rhs.__args__[2]
            d4 = constraint.rhs.__args__[3]

            conj = [BinConstraintT(constraint.lhs, Dyn, op_eq),
                    BinConstraintD(d1, Dyn, op_eq),
                    BinConstraintD(d2, Dyn, op_eq),
                    BinConstraintD(d3, Dyn, op_eq),
                    BinConstraintD(d4, Dyn, op_eq)]
            return Disj([Conj(conj),
                         BinConstraintT(constraint.lhs, TensorType([d1, d2, d3, d4]), op_eq)]), counter

        elif constraint.op == op_consistency:
            c_dyn = Disj([BinConstraintT(constraint.lhs, Dyn, op_eq), BinConstraintT(constraint.rhs, Dyn, op_eq)])
            [c_tensor_1, c_tensor_2, c_tensor_3, c_tensor_4], counter = gen_consistency_constraints(constraint, counter)

            return Disj([c_dyn, c_tensor_1, c_tensor_2, c_tensor_3, c_tensor_4]), counter

        elif constraint.op == op_leq:
            assert isinstance(constraint.rhs, int)
            disj = []
            for i in range(1, constraint.rhs + 1):
                dims = []
                for j in range(1, i + 1):
                    dim_var, counter = gen_dvar(counter)
                    dims.append(dim_var)
                disj.append(BinConstraintT(constraint.lhs, TensorType(dims), op_eq))
            return Disj(disj), counter
        else:
            return constraint, counter

    # dimension constraints
    elif isinstance(constraint, BinConstraintD):

        if constraint.op == op_precision:
            if isinstance(constraint.lhs, int):
                return BinConstraintD(constraint.lhs, constraint.rhs, op_eq), counter
            elif constraint.lhs == Dyn:
                return T(), counter

        elif constraint.op == op_consistency:
            return Disj([BinConstraintD(constraint.lhs, constraint.rhs, op_eq),
                         BinConstraintD(constraint.rhs, Dyn, op_eq), BinConstraintD(constraint.lhs, Dyn, op_eq)]), counter

        else:
            return constraint, counter

    elif isinstance(constraint, Conj):
        new = []
        for c in constraint.conjucts:
            new_c, counter = transform_constraint(c, counter)
            new.append(new_c)
        return Conj(new), counter

    elif isinstance(constraint, Disj):
        new = []
        for c in constraint.disjuncts:
            new_c, counter = transform_constraint(c, counter)
            new.append(new_c)
        return Disj(new), counter

    # this is up to size 1 and 2 only for now
    elif isinstance(constraint, TGreatestUpperBound):
        c1 = Conj([Disj([BinConstraintT(constraint.rhs1, Dyn, op_eq),
                         BinConstraintT(constraint.rhs2, Dyn, op_eq)]), BinConstraintT(constraint.res, Dyn, op_eq)])

        [c2, c3, c4, c5], counter = gen_greatest_upper_bound(constraint, counter)

        return Disj([c1, c2, c3, c4, c5]), counter

    elif isinstance(constraint, DGreatestUpperBound):
        c1 = Conj([BinConstraintD(constraint.rhs1, Dyn, op_eq), BinConstraintD(constraint.res, constraint.rhs2, op_eq)])
        c2 = Conj([BinConstraintD(constraint.rhs2, Dyn, op_eq), BinConstraintD(constraint.res, constraint.rhs1, op_eq)])
        c3 = Conj([BinConstraintD(constraint.rhs2, constraint.rhs1, op_eq), BinConstraintD(constraint.res, constraint.rhs1, op_eq)])
        return Disj([c1, c2, c3]), counter

    elif isinstance(constraint, CalcConv):
        transformed, counter = transform_conv(constraint, counter)
        return transformed, counter

    elif isinstance(constraint, CalcMaxPool):
        transformed, counter = transform_maxpool(constraint, counter)
        return transformed, counter

    elif isinstance(constraint, CalcProduct):
        transformed, counter = transform_flatten(constraint, counter)
        return transformed, counter

    elif isinstance(constraint, CanReshape):
        transformed, counter = transform_can_reshape(constraint, counter)
        return transformed, counter

    elif isinstance(constraint, ApplyBroadcasting):
        transformed, counter = transform_apply_broadcasting(constraint, counter)
        return transformed, counter
    else:
        return constraint, counter


def transform_conv(constraint, counter):
    """
    :param constraint: Calc-conv
    :return: new counter and the transformed constraint
    """

    assert isinstance(constraint, CalcConv)

    d, counter = gen_tensor_dims(4, counter)
    conv_result = TensorType([d[0], d[1], d[2], d[3]])

    # the convolution result is a tensor of size 4
    c1 = BinConstraintT(constraint.conv_result, conv_result, op_eq)

    # the second dimension of the output is equal to the output channels
    c2 = Conj([BinConstraintD(d[1], constraint.c_out, op_eq), BinConstraintD(d[1], Dyn, op_neq)])

    # the input corresponds to the output in the first dimension of the convolution
    c3 = BinConstraintD(constraint.matching_constraint[0], d[0], op_eq)

    c4, c5 = calc_last_two_dims(constraint, d)

    leq_constraints = Conj([BinConstraintD(0, d[0], op_leq),
                            BinConstraintD(0, d[1], op_leq),
                            BinConstraintD(0, d[2], op_leq),
                            BinConstraintD(0, d[3], op_leq)])

    return Conj([c1, c2, c3, c4, c5, leq_constraints]), counter


def calc_last_two_dims(constraint, d):
    """
    Generates constraints for the last two dimensions of a convolution or a maxpool output
    :param d: the variables for the output dimensions
    :return:
    """

    assert isinstance(constraint, CalcConv) or isinstance(constraint, CalcMaxPool)

    b3 = constraint.matching_constraint[2]
    b4 = constraint.matching_constraint[3]

    b3_dyn = Conj([BinConstraintD(d[2], Dyn, op_eq), BinConstraintD(b3, Dyn, op_eq)])
    b4_dyn = Conj([BinConstraintD(d[3], Dyn, op_eq), BinConstraintD(b4, Dyn, op_eq)])

    d3_not_dyn = Conj([BinConstraintD(d[2], Dyn, op_neq), BinConstraintD(b3, Dyn, op_neq)])
    d4_not_dyn = Conj([BinConstraintD(d[3], Dyn, op_neq), BinConstraintD(b4, Dyn, op_neq)])

    # transform parameters into tuples incase they are not already
    padding = (constraint.padding, constraint.padding) \
        if isinstance(constraint.padding, int) else constraint.padding
    kernel = (constraint.kernel, constraint.kernel) \
        if isinstance(constraint.kernel, int) else constraint.kernel
    stride = (constraint.stride, constraint.stride) \
        if isinstance(constraint.stride, int) else constraint.stride
    dilation = (constraint.dilation, constraint.dilation) \
        if isinstance(constraint.dilation, int) else constraint.dilation

    f1 = BinConstraintD(b3, BinConstraintD(2, padding[0], op_mul), op_add)
    f2 = BinConstraintD(dilation[0], BinConstraintD(kernel[0], 1, op_sub), op_mul)
    f3 = BinConstraintD(BinConstraintD(BinConstraintD(f1, f2, op_sub), 1, op_sub), stride[0], op_div)
    f4 = BinConstraintD(f3, 1, op_add)

    c4 = Disj([b3_dyn, Conj([d3_not_dyn, BinConstraintD(d[2], f4, op_eq)])])

    f11 = BinConstraintD(b4, BinConstraintD(2, padding[1], op_mul), op_add)
    f22 = BinConstraintD(dilation[1], BinConstraintD(kernel[1], 1, op_sub), op_mul)
    f33 = BinConstraintD(BinConstraintD(BinConstraintD(f11, f22, op_sub), 1, op_sub), stride[1], op_div)
    f44 = BinConstraintD(f33, 1, op_add)

    c5 = Disj([b4_dyn, Conj([d4_not_dyn, BinConstraintD(d[3], f44, op_eq)])])

    return c4, c5


def transform_maxpool(constraint, counter):
    """
    :param constraint: MaxPool
    :return: new counter and the transformed constraint
    """

    assert isinstance(constraint, CalcMaxPool)

    d, counter = gen_tensor_dims(4, counter)
    maxpool_result = TensorType([d[0], d[1], d[2], d[3]])

    # the maxpool result is a tensor of size 4
    c1 = BinConstraintT(constraint.maxpool_result, maxpool_result, op_eq)

    # the input corresponds to the output in the first and second dimension of maxpool
    c2 = BinConstraintD(constraint.matching_constraint[1], d[1], op_eq)
    c3 = BinConstraintD(constraint.matching_constraint[0], d[0], op_eq)
    c4, c5 = calc_last_two_dims(constraint, d)

    leq_constraints = Conj([BinConstraintD(0, d[0], op_leq),
                            BinConstraintD(0, d[1], op_leq),
                            BinConstraintD(0, d[2], op_leq),
                            BinConstraintD(0, d[3], op_leq)])

    return Conj([c1, c2, c3, c4, c5, leq_constraints]), counter


def transform_apply_broadcasting(constraint, counter):
    """
    Handles broadcasting constraints
    :param constraint: a broadcasting constraint
    :return: target language constraint. Needs one iteration.
    """
    assert isinstance(constraint, ApplyBroadcasting)
    e11, e12 = constraint.res1, constraint.res2
    e1, e2 = constraint.input1, constraint.input2

    e1_dyn = BinConstraintT(e1, Dyn, op_eq)
    e2_dyn = BinConstraintT(e2, Dyn, op_eq)

    e1_equal_e11 = BinConstraintT(e1, e11, op_eq)
    e2_equal_e12 = BinConstraintT(e2, e12, op_eq)
    e1_dyn_constraint = Conj([e1_dyn, e1_equal_e11, e2_equal_e12])
    e2_dyn_constraint = Conj([e2_dyn, e1_equal_e11, e2_equal_e12])

    # generate dimensions to create tensors of size 1
    final_tensor_1_constraint, _, _, nat_dims_1, counter = \
        gen_broadcasting_constraints(e1, e2, e11, e12, 1, counter)

    # generate dimensions to create tensors of size 2
    final_tensor_2_constraint_no_padding, final_tensor_2_constraint_padding_arg1, \
        final_tensor_2_constraint_padding_arg2, nat_dims_2, counter = \
        gen_broadcasting_constraints(e1, e2, e11, e12, 2, counter)

    # generate dimensions to create tensors of size 3
    final_tensor_3_constraint_no_padding, final_tensor_3_constraint_padding_arg1, \
        final_tensor_3_constraint_padding_arg2, nat_dims_3, counter = \
        gen_broadcasting_constraints(e1, e2, e11, e12, 3, counter)

    # generate dimensions to create tensors of size 4
    final_tensor_4_constraint_no_padding, final_tensor_4_constraint_padding_arg1, \
        final_tensor_4_constraint_padding_arg2, nat_dims_4, counter = \
        gen_broadcasting_constraints(e1, e2, e11, e12, 4, counter)

    final_result = Disj([
        e1_dyn_constraint,
        e2_dyn_constraint,
        final_tensor_1_constraint,
        final_tensor_2_constraint_no_padding,
        final_tensor_2_constraint_padding_arg1,
        final_tensor_2_constraint_padding_arg2,
        final_tensor_3_constraint_no_padding,
        final_tensor_3_constraint_padding_arg1,
        final_tensor_3_constraint_padding_arg2,
        final_tensor_4_constraint_no_padding,
        final_tensor_4_constraint_padding_arg1,
        final_tensor_4_constraint_padding_arg2
    ])

    return Conj([final_result, *nat_dims_1, *nat_dims_2, *nat_dims_3, *nat_dims_4]), counter
    # return final_result, counter


def generate_all_possibilities(my_list):
    # generate all possibilities of being equal or not equal to dyn for my_list
    eq_possibilities = [BinConstraintD(my_list[i], Dyn, op_eq) for i in range(len(my_list))]
    neq_possibilities = [BinConstraintD(my_list[i], Dyn, op_neq) for i in range(len(my_list))]
    d_possibilities = []

    for i in zip(eq_possibilities, neq_possibilities):
        d_possibilities.append(list(i))
    all_possibilities = list(itertools.product(*d_possibilities))
    return all_possibilities

def transform_flatten(constraint, counter):
    """
    - Check that start and end dimensions are valid
    - Generate flattened constraints
    :param constraint: a flatten constraint
    """
    assert isinstance(constraint, CalcProduct)

    start = constraint.start
    end = constraint.end
    dims = constraint.dims_to_flatten
    flattened = constraint.flattened
    n = len(constraint.dims_to_flatten)

    # this will be evaluated right here
    boundary_check = (0 <= start and start < end and end <= n)

    c_boundary = T() if boundary_check else F()

    lhs = dims[0:start]
    rhs = dims[end:]
    mid = dims[start:end]

    all_possibilities = generate_all_possibilities(mid)

    all_constraints = []

    for p in all_possibilities:
        p = list(p)
        # this tells us there is a dynamic variable
        contains_dyn = not(all([constraint.op == op_neq for constraint in p]))
        if contains_dyn:
            mid_var = [Dyn]
            total_constraints = lhs + mid_var + rhs
            if len(total_constraints) > 4:
                all_constraints.append(F())
            else:
                all_constraints.append(Conj([BinConstraintT(flattened, TensorType(lhs + mid_var + rhs), op_eq)] + p))
        else:
            new_var, counter = gen_dvar(counter)
            mid_eq_prod = Conj([BinConstraintD(new_var, Prod(mid), op_eq), BinConstraintD(new_var, Dyn, op_neq)])
            mid_var = [new_var]
            total_constraints = lhs + mid_var + rhs
            if len(total_constraints) > 4:
                all_constraints.append(F())
            else:
                all_constraints.append(Conj([BinConstraintT(flattened, TensorType(lhs + mid_var + rhs), op_eq), mid_eq_prod] + p))

    return Conj([Disj(all_constraints), c_boundary]), counter


def transform_can_reshape(constraint, counter):
    """
    Handle reshape constraints
    """

    assert isinstance(constraint, CanReshape)

    d, counter = gen_tensor_dims(4, counter)

    d1 = d[0]
    d2 = d[1]
    d3 = d[2]
    d4 = d[3]

    target = constraint.target.__args__

    is_fully_static = all([d != Dyn for d in target])

    # dynamic tensor
    c1_dyn = BinConstraintT(constraint.src, Dyn, op_eq)
    c2_tensor1 = BinConstraintT(constraint.src, TensorType([d1]), op_eq)
    c2_tensor2 = BinConstraintT(constraint.src, TensorType([d1, d2]), op_eq)
    c2_tensor3 = BinConstraintT(constraint.src, TensorType([d1, d2, d3]), op_eq)
    c2_tensor4 = BinConstraintT(constraint.src, TensorType([d1, d2, d3, d4]), op_eq)

    d1_eq_dyn = BinConstraintD(d1, Dyn, op_eq)
    d1_neq_dyn = BinConstraintD(d1, Dyn, op_neq)

    d2_eq_dyn = BinConstraintD(d2, Dyn, op_eq)
    d2_neq_dyn = BinConstraintD(d2, Dyn, op_neq)

    d3_eq_dyn = BinConstraintD(d3, Dyn, op_eq)
    d3_neq_dyn = BinConstraintD(d3, Dyn, op_neq)

    d4_eq_dyn = BinConstraintD(d3, Dyn, op_eq)
    d4_neq_dyn = BinConstraintD(d3, Dyn, op_neq)

    nat_d1 = BinConstraintD(0, d1, op_leq)
    nat_d2 = BinConstraintD(0, d2, op_leq)
    nat_d3 = BinConstraintD(0, d3, op_leq)
    nat_d4 = BinConstraintD(0, d4, op_leq)

    if is_fully_static:
        # size 1 tensor
        c3_tensor1 = Disj([d1_eq_dyn,
                           (Conj([d1_neq_dyn,
                                  BinConstraintD(d1, Prod(target), op_eq)]))])
        all_tensor_1 = Conj([c2_tensor1, c3_tensor1])

        # size 2 tensor
        all_tensor_2 = Conj([c2_tensor2, gen_all_reshape_possibilities([d1, d2], target)])

        # size 3 tensor
        all_tensor_3 = Conj([c2_tensor3, gen_all_reshape_possibilities([d1, d2, d3], target)])

        # size 4 tensor
        all_tensor_4 = Conj([c2_tensor4, gen_all_reshape_possibilities([d1, d2, d3, d4], target)])

        return Conj([Disj([c1_dyn, all_tensor_1, all_tensor_2, all_tensor_3, all_tensor_4]),
                     nat_d1, nat_d2, nat_d3, nat_d4]), counter

    # then there must be exactly one occurrence of dyn
    else:
        new_target = []

        for n in target:
            if n != Dyn:
                new_target.append(n)

        # tensor 1
        c3_tensor1 = Disj([d1_eq_dyn,
                           (Conj([d1_neq_dyn,
                                  is_dim_div_by_target(new_target, d1)]))])
        all_tensor_1 = Conj([c2_tensor1, c3_tensor1])

        # tensor 2
        c21 = Disj([d1_eq_dyn, d2_eq_dyn])
        c22 = Conj([d1_neq_dyn, d2_neq_dyn, is_dim_div_by_target(new_target, Prod([d1, d2]))])
        all_tensor_2 = Conj([c2_tensor2, Disj([c21, c22])])

        # tensor 3
        c31 = Disj([d1_eq_dyn, d2_eq_dyn, d3_eq_dyn])
        c32 = Conj([d1_neq_dyn, d2_neq_dyn, d3_neq_dyn, is_dim_div_by_target(new_target, Prod([d1, d2, d3]))])
        all_tensor_3 = Conj([c2_tensor3, Disj([c31, c32])])

        # tensor 4
        c41 = Disj([d1_eq_dyn, d2_eq_dyn, d3_eq_dyn, d4_eq_dyn])
        c42 = Conj([d1_neq_dyn, d2_neq_dyn, d3_neq_dyn, d4_neq_dyn, is_dim_div_by_target(new_target, Prod([d1, d2, d3, d4]))])
        all_tensor_4 = Conj([c2_tensor4, Disj([c41, c42])])

        return Conj([Disj([c1_dyn, all_tensor_1, all_tensor_2, all_tensor_3, all_tensor_4]),
                     nat_d1, nat_d2, nat_d3, nat_d4]), counter


def is_target_div_by_dim(target, dim):
    """
    Generate constraints to check if target dimensions are divisible by the input dimension(s)
    """
    return BinConstraintD(BinConstraintD(Prod(target), dim, op_mod), 0, op_eq)


def is_dim_div_by_target(target, dim):
    """
        Generate constraints to check if the input dimension(s) is divisible by the target dimensions
    :param target:
    :param dim:
    :return:
    """
    return BinConstraintD(BinConstraintD(dim, Prod(target), op_mod), 0, op_eq)


def gen_all_reshape_possibilities(list_of_dims, target):
    """
    Consider all possibilities what the input dimensions could be (number or dynamic)
    Then generate the appropriate constraints using multiplication or mod depending on the possibility
    The possibiities we consider here are the cross product of being equal to dyn or not equal to dyn
    for the input. Target is fixed because at most one dimension could be dyn.
    We have different cases for this.
    """

    all_possibilities = generate_all_possibilities(list_of_dims)

    all_constraints = []

    for p in all_possibilities:
        to_multiply = []

        p = list(p)

        for constraint in p:
            assert isinstance(constraint, BinConstraintD)
            if constraint.op == op_neq:
                to_multiply.append(constraint.lhs)

        if not to_multiply:
            all_constraints.append(Conj(p))

        elif len(to_multiply) < len(list_of_dims):
            all_constraints.append(Conj(p + [is_target_div_by_dim(target, Prod(to_multiply))]))
        else:
            all_constraints.append(Conj(p + [BinConstraintD(Prod(list_of_dims),
                                                            Prod(target), op_eq)]))

    return Disj(all_constraints)


def broadcast_dim(tensor_input1, tensor_input2, res1, res2, index, padding=False):
    """
    Note that this is is an asymetric function
    :param tensor_input1: the input which will have broadcasting or None if padding is false
    :param tensor_input2: the input which may not have broadcasting
    :param res1: the simulated result of broadcasting
    :param res2: the simulated result of broadcasting
    :param index: Index of the dimension we will broadcast
    :return:
    """


    if tensor_input1[index] is None:
        assert padding

    if not padding:
        return Conj([BinConstraintD(tensor_input1[index], 1, op_eq),
                     BinConstraintD(res1[index], res2[index], op_eq),
                     BinConstraintD(res2[index], tensor_input2[index], op_eq)])

    else:
        # we don't set the input dimension to 1, since it doesn't exist.
        return Conj([BinConstraintD(res1[index], res2[index], op_eq),
                     BinConstraintD(res2[index], tensor_input2[index], op_eq)])


# TODO: generate NAT constraints for the fresh variables you generate
def apply_padding(e1_var, e11, e2, e12, d2, d11, d12, counter):

    res = []

    for i in range(1, len(d2)):

        d1, counter = gen_tensor_dims(i, counter)

        nat_constraints = gen_nat_constraints(d1)

        e1 = BinConstraintT(e1_var, TensorType(d1), op_eq)

        simulate_padding = [None] * (len(d2) - i)

        assert len(simulate_padding + d1) == len(d2)

        broadcast_padding = []

        for j in range((len(d2) - i)):
            broadcast_padding.append(broadcast_dim(simulate_padding, d2, d11, d12, j, True))

        all_broadcasting_possibilities = generate_all_broadcasting_possibilities_no_padding(d1,
                                                                                            d2[(len(d2) - i):],
                                                                                            d11[(len(d2) - i):],
                                                                                            d12[(len(d2) - i):])

        c = Conj([e1, e11, e2, e12,
                  *broadcast_padding,
                  all_broadcasting_possibilities,
                  *nat_constraints
                  ])
        res.append(c)


    return Disj(res), counter


def no_broadcast_dim_with_index(d1, d2, d3, d4, i):
    """
    Note that is is a symetric function
    No broadcasting for a given dimension
    :param d1: input1
    :param d2: input2
    :param d3: mock value 1 (for d1)
    :param d4: mock value 2 (for d2)
    :return:
    """
    return Conj([
        Disj([
            Conj([BinConstraintD(d1[i], 1, op_eq),
                  BinConstraintD(d2[i], 1, op_eq)]),

            Conj([BinConstraintD(d1[i], 1, op_neq),
                  BinConstraintD(d2[i], 1, op_neq)])]),

        BinConstraintD(d1[i], d3[i], op_eq),
        BinConstraintD(d2[i], d4[i], op_eq)])



def gen_lists_of_dims(num_tensors, dim_size, counter):
    """
    Generate lists of dimensions for a fixed size.
    """
    res = []

    for _ in range(num_tensors):
        dims, counter = gen_tensor_dims(dim_size, counter)
        res.append(dims)

    return res, counter


def create_equality_constraints_for_broadcasting(e1, e2, e11, e12, d1, d2, d11, d12):
    """
    Creates equality constraints of for when broadcasting does not apply.
    """
    e1_tensor = BinConstraintT(e1, TensorType(d1), op_eq)
    e11_tensor = BinConstraintT(e11, TensorType(d11), op_eq)
    e2_tensor = BinConstraintT(e2, TensorType(d2), op_eq)
    e12_tensor = BinConstraintT(e12, TensorType(d12), op_eq)
    return [e1_tensor, e11_tensor, e2_tensor, e12_tensor]


def gen_consistency_constraints(constraint, counter):
    """
    Generate dconsistency constraints for all dimensions 1...4
    """
    all_constraints = []

    for i in range(1, 5):
        new_dims_rhs_1, counter = gen_tensor_dims(i, counter)
        new_dims_rhs_2, counter = gen_tensor_dims(i, counter)

        nat_constraints = gen_nat_constraints(new_dims_rhs_1 + new_dims_rhs_2)

        c_tensor_i = Conj([BinConstraintT(constraint.lhs, TensorType(new_dims_rhs_1), op_eq),
                           BinConstraintT(constraint.rhs, TensorType(new_dims_rhs_2), op_eq)] +
                          [BinConstraintD(d1, d2, op_consistency) for
                           d1, d2 in zip(new_dims_rhs_1, new_dims_rhs_2)] + nat_constraints)

        all_constraints.append(c_tensor_i)

    return all_constraints, counter


def gen_greatest_upper_bound(constraint, counter):
    """
    Generate greatest upper bound constraints for dimensions 1..4
    """

    all_constraints = []

    for i in range(1, 5):
        c = []
        dims1, counter = gen_tensor_dims(i, counter)
        c1tensor = TensorType(dims1)

        dims2, counter = gen_tensor_dims(i, counter)
        c2tensor = TensorType(dims2)

        dims3, counter = gen_tensor_dims(i, counter)
        c3tensor = TensorType(dims3)

        c += [BinConstraintT(constraint.rhs1, c1tensor, op_eq),
              BinConstraintT(constraint.rhs2, c2tensor, op_eq),
              BinConstraintT(constraint.res, c3tensor, op_eq)] + \
            gen_nat_constraints(dims1 + dims2 + dims3)

        assert len(c3tensor.__args__) == len(c1tensor.__args__) == len(c2tensor.__args__)
        for i in range(len(c3tensor.__args__)):
            c.append(DGreatestUpperBound(c3tensor.__args__[i],
                                         c1tensor.__args__[i],
                                         c2tensor.__args__[i]))

        all_constraints.append(Conj(c))
    return all_constraints, counter


def generate_all_broadcasting_possibilities_no_padding(d1, d2, d11, d12):
    """
    Assuming no padding, generate all possibilities of broadcasting on every possible dimension.
    Cross product of n dimensions
    """

    size = len(d1)

    res2 = []
    for i in range(size):
        t1 = broadcast_dim(d1, d2, d11, d12, i)
        t2 = broadcast_dim(d2, d1, d12, d11, i)
        t3 = no_broadcast_dim_with_index(d1, d2, d11, d12, i)

        res2.append(Disj([t1, t2, t3]))

    return Conj(res2)


def gen_broadcasting_constraints(e1, e2, e11, e12, i, counter):

    # generate dimensions to create 4 tensors of size i each
    dims, counter = gen_lists_of_dims(4, i, counter)
    [d1, d2, d3, d4] = dims
    nat_dims_i = gen_nat_constraints(list(itertools.chain(*dims)))

    initialize_tensors_constraints = create_equality_constraints_for_broadcasting(e1, e2, e11, e12,
                                                                                  d1, d2, d3, d4)

    [e1_tensor, e11_tensor, e2_tensor, e12_tensor] = initialize_tensors_constraints

    # without padding, broadcast all possibilities for tensors of size i
    final_tensor_constraint_no_padding = Conj([*initialize_tensors_constraints,
                                               generate_all_broadcasting_possibilities_no_padding(d1, d2, d3, d4)])

    # # # with padding, broadcast all possibilities for tensors of size i
    final_tensor_constraint_padding_arg1, counter = \
        apply_padding(e1, e11_tensor, e2_tensor, e12_tensor, d2, d3, d4, counter)

    final_tensor_constraint_padding_arg2, counter = \
        apply_padding(e2, e12_tensor, e1_tensor, e11_tensor, d1, d4, d3, counter)

    return final_tensor_constraint_no_padding, \
        final_tensor_constraint_padding_arg1, \
        final_tensor_constraint_padding_arg2, nat_dims_i, counter
