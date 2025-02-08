import torch
from torch_geometric.nn import MessagePassing

from temp.deprecate.global_vars import DEVICE


def scale_node_degrees(data_obj):
    idx, data = data_obj

    if "is_transformed" in data:
        return data_obj

    if data.num_con_nodes > 0:
        norm_con_degree = node_degree_scaling(
            data.edge_index_var, (data.num_var_nodes, data.num_con_nodes)
        )
        data.con_node_features[:, -1] = norm_con_degree.view(-1)

        norm_var_degree = node_degree_scaling(
            data.edge_index_con, (data.num_con_nodes, data.num_var_nodes)
        )
        data.var_node_features[:, -1] = norm_var_degree.view(-1)
    else:
        data.var_node_features[:, -1] = 0

    data.is_transformed = True

    return idx, data


def node_degree_normalization(data):
    if data.num_con_nodes > 0:
        norm_con_degree = node_degree_scaling(
            data.edge_index_var, (data.num_var_nodes, data.num_con_nodes)
        )
        data.con_node_features[:, -1] = norm_con_degree

        norm_var_degree = node_degree_scaling(
            data.edge_index_con, (data.num_con_nodes, data.num_var_nodes)
        )
        data.var_node_features[:, -1] = norm_var_degree
    else:
        data.var_node_features[:, -1] = 0

    return data


def Abc_normalization(data):
    # Normalization of constraint matrix
    norm_rhs, max_coeff = normalize_rhs(
        data.edge_index_var,
        data.edge_features_var,
        data.rhs,
        (data.num_var_nodes, data.num_con_nodes),
    )
    data.rhs = norm_rhs
    data.con_node_features[:, 1] = norm_rhs.view(-1)
    data.edge_features_var /= max_coeff[data.edge_index_var[1]]
    data.edge_features_con /= max_coeff[data.edge_index_con[0]]

    # Normalization of objective coefficients
    data.obj /= data.obj.abs().max()
    data.var_node_features[:, -2] = data.obj.view(-1)

    return data


def AbcNorm(data_obj):
    if isinstance(data_obj, tuple):
        idx, data = data_obj
    else:
        data = data_obj

    if "is_transformed" in data:
        return data

    data = data.clone()

    # Normalizing A, b, and c coefficients
    data = Abc_normalization(data)

    # Node degree normalization
    data = node_degree_normalization(data)

    if "dual_val" in data:
        data.dual_val /= data.dual_val.abs().max()

    if "relaxed_sol_val" in data:
        data.relaxed_sol_val /= data.relaxed_sol_val.abs().max()

    data.is_transformed = True

    if isinstance(data_obj, tuple):
        return idx, data

    return data


class NormalizeRHS(MessagePassing):
    def __init__(self):
        super(NormalizeRHS, self).__init__(aggr="max", flow="source_to_target")

    def forward(self, edge_index, coeff, rhs, size):
        abs_coeff = self.propagate(edge_index, edge_attr=coeff, size=size)
        abs_rhs = torch.abs(rhs)
        max_coeff = (
            torch.cat((abs_coeff, abs_rhs), dim=-1).max(dim=-1).values.view(-1, 1)
        )
        norm_rhs = rhs / max_coeff
        return norm_rhs, max_coeff

    def message(self, edge_attr):
        return torch.abs(edge_attr)


class NodeDegreeScaling(MessagePassing):
    def __init__(self):
        super(NodeDegreeScaling, self).__init__(aggr="add", flow="source_to_target")

    def forward(self, edge_index, size):
        connected = torch.ones((size[0], 1), dtype=torch.float)
        node_degree = self.propagate(edge_index, connected=connected, size=size)
        norm_node_degree = node_degree / node_degree.max()

        return norm_node_degree.view(-1)

    def message(self, connected_j):
        return connected_j


class NodeDegreeCalculation(MessagePassing):
    def __init__(self):
        super(NodeDegreeCalculation, self).__init__(aggr="add", flow="source_to_target")

    def forward(self, edge_index, size):
        connected = torch.ones((size[0], 1), dtype=torch.float, device=DEVICE)
        total_degree = self.propagate(edge_index, connected=connected, size=size)

        return total_degree

    def message(self, connected_j):
        return connected_j


class ConstraintValuation(MessagePassing):
    def __init__(self):
        super(ConstraintValuation, self).__init__(aggr="add", flow="source_to_target")

    def forward(self, assignment, edge_index, coeff, rhs, lb, ub, con_kind, size):
        # con_kind = 1 for less than constraints (<=) and con_kind = 0 for equality constraints (=)
        if lb is None or ub is None:
            x = assignment
        else:  # assignment is decision values normalized between lb and ub
            x = (assignment * (ub - lb) + lb).view(-1, 1)
        Ax = self.propagate(edge_index, x=x, edge_attr=coeff, size=size)
        difference = Ax - rhs
        violation = torch.relu(difference) * con_kind + torch.abs(difference) * (
            1 - con_kind
        )

        return Ax, violation

    def message(self, x_j, edge_attr):
        return x_j * edge_attr

    def update(self, aggr_out):
        return aggr_out


class SumViolation(MessagePassing):
    def __init__(self):
        super(SumViolation, self).__init__(aggr="add", flow="source_to_target")

    def forward(self, violation, edge_index, size):
        output = self.propagate(edge_index, x=violation, size=size)

        return output

    def message(self, x_j):
        return x_j

    def update(self, aggr_out):
        return aggr_out


normalize_rhs = NormalizeRHS()
node_degree_scaling = NodeDegreeScaling()
get_node_degrees = NodeDegreeCalculation()
constraint_valuation = ConstraintValuation()
sum_violation = SumViolation()
