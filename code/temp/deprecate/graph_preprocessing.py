import os
import random

import gurobipy as gp
import networkx as nx
import numpy as np
import torch
from global_vars import *
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.nn import MessagePassing


def get_bipartite_graph(info):
    vtype_mapping = {"C": "continuous", "B": "binary", "I": "integer"}

    n_vars = info.var_info.n
    var_names = [f"v_{i}" for i in range(info.var_info.n)]
    var_idxs = list(range(n_vars))
    var_domains = [vtype_mapping[t] for t in info.var_info.types]
    var_lb = [b for b in info.var_info.lbs]
    var_ub = [b for b in info.var_info.ubs]
    obj_multiplier = info.obj_info.sense
    obj_coeff = [info.obj_info.ks[i] for i in range(info.var_info.n)]
    obj_coeff = obj_multiplier * np.array(obj_coeff)

    n_cons = info.con_info.n
    con_names = [f"c_{i}" for i in range(info.con_info.n)]
    con_idxs = list(range(n_cons))
    rhs = info.con_info.rhs
    senses = [info.con_info.OP_TO_ENUM[t] for t in info.con_info.types]
    con_multiplier = np.where(np.array(senses) == ">=", -1, 1)
    changed_senses = np.where(np.array(senses) == "==", "E", "L")

    edge_names = []
    edge_weights = []
    for i in range(info.con_info.n):
        for j, c in zip(info.con_info.lhs_p[i], info.con_info.lhs_c[i]):
            con_name = con_names[i]
            var_name = var_names[j]
            edge_names.append((con_name, var_name))
            edge_weights.append(con_multiplier[i] * c)

    rhs = np.array(rhs) * con_multiplier
    g = nx.Graph()
    g.add_nodes_from(var_names + con_names)
    g.add_edges_from(edge_names)
    assert len(edge_weights) == len(edge_names)
    nx.set_edge_attributes(g, dict(zip(edge_names, edge_weights)), name="coeff")
    nx.set_node_attributes(
        g,
        dict(zip(var_names + con_names, [0] * n_vars + [1] * n_cons)),
        name="bipartite",
    )
    nx.set_node_attributes(g, dict(zip(var_names, var_lb)), name="lb")
    nx.set_node_attributes(g, dict(zip(var_names, var_ub)), name="ub")
    nx.set_node_attributes(g, dict(zip(var_names, var_domains)), name="domain")
    nx.set_node_attributes(g, dict(zip(var_names, obj_coeff)), name="obj_coeff")
    nx.set_node_attributes(g, dict(zip(var_names, var_idxs)), name="index")

    nx.set_node_attributes(g, dict(zip(con_names, rhs)), name="rhs")
    nx.set_node_attributes(g, dict(zip(con_names, changed_senses)), name="kind")
    nx.set_node_attributes(g, dict(zip(con_names, con_idxs)), name="index")

    return g, con_names


def add_label(g, info, solution):
    var_names = [f"v_{i}" for i in range(info.var_info.n)]
    obj_multiplier = info.obj_info.sense
    obj_vector = obj_multiplier * solution[:, 0]
    incumbent_ind = np.argmin(obj_vector)
    norm_obj_vector = StandardScaler().fit_transform(obj_vector.reshape(-1, 1)).ravel()
    solutions_matrix = solution[:, 1:]

    incumbent_sol_vector = solutions_matrix[incumbent_ind]
    mean_bias_vector = np.mean(solutions_matrix, axis=0).ravel()
    sol_weights = np.array(
        torch.softmax(torch.from_numpy(-norm_obj_vector), 0)
    ).reshape(-1, 1)
    weighted_bias_vector = np.sum(solutions_matrix * sol_weights, 0).ravel()

    assert len(var_names) == weighted_bias_vector.shape[0]

    nx.set_node_attributes(
        g, dict(zip(var_names, incumbent_sol_vector)), name="incumbent"
    )
    nx.set_node_attributes(g, dict(zip(var_names, mean_bias_vector)), name="mean_bias")
    nx.set_node_attributes(
        g, dict(zip(var_names, weighted_bias_vector)), name="weighted_bias"
    )
    return g


# Preprocess indices of bipartite graphs to make batching work.
class BipartiteData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key in ["edge_index_var"]:
            return torch.tensor([self.num_var_nodes, self.num_con_nodes]).view(2, 1)
        elif key in ["edge_index_con"]:
            return torch.tensor([self.num_con_nodes, self.num_var_nodes]).view(2, 1)
        elif key in ["index_con"]:
            return self.num_con_nodes
        elif key in ["index_var"]:
            return self.num_var_nodes
        else:
            return 0


def create_data_object(graph, is_labeled=True):
    """Create a BipartiteData object from a graph.

    Args:
        instance_name: Name of the instance
        graph: NetworkX graph containing all problem information
        is_labeled: Whether to include solution labels
        save_dir: Directory to save the data object
        preprocess_start_time: Start time for preprocessing
    """
    bipartite_vals = np.array([data["bipartite"] for _, data in graph.nodes(data=True)])

    num_con_nodes = int(bipartite_vals.sum())
    num_var_nodes = len(bipartite_vals) - num_con_nodes

    obj = np.zeros((num_var_nodes, 1))
    is_binary = np.zeros(num_var_nodes, dtype=bool)
    lb = np.zeros(num_var_nodes)
    ub = np.zeros(num_var_nodes)
    feat_var = np.zeros((num_var_nodes, 5))  # [isb, isc, isi, obj_coeff, degree]

    if is_labeled:
        y_real = np.zeros(num_var_nodes)
        y_norm_real = np.zeros(num_var_nodes)
        y_incumbent = np.zeros(num_var_nodes)

    # Initialize arrays for constraint features
    feat_con = np.zeros((num_con_nodes, 3))  # [kind, rhs, degree]
    rhs = np.zeros((num_con_nodes, 1))
    con_kind = np.zeros((num_con_nodes, 1))

    # Edge lists and features
    edge_list_var = []
    edge_list_con = []
    edge_features_var = []
    edge_features_con = []

    index_con = []
    index_var = []

    for node, node_data in graph.nodes(data=True):
        if node_data["bipartite"] == 0:
            idx = node_data["index"]
            index_var.append(0)

            isb = int(node_data["domain"] == "binary")
            isc = int(node_data["domain"] == "continuous")
            isi = int(node_data["domain"] == "integer")

            is_binary[idx] = isb
            lb[idx] = node_data["lb"]
            ub[idx] = node_data["ub"]

            assert ub[idx] >= lb[idx]

            if is_labeled:
                w_bias = node_data["weighted_bias"]
                incumbent = node_data["incumbent"]

                if abs(ub[idx] - lb[idx]) > 1e-6:
                    norm_bias = (w_bias - lb[idx]) / (ub[idx] - lb[idx])
                    norm_incumbent = (incumbent - lb[idx]) / (ub[idx] - lb[idx])

                    if not (0 <= norm_bias <= 1):
                        w_bias = np.clip(w_bias, lb[idx], ub[idx])
                        norm_bias = (w_bias - lb[idx]) / (ub[idx] - lb[idx])

                    if not (0 <= norm_incumbent <= 1):
                        incumbent = np.clip(incumbent, lb[idx], ub[idx])
                        norm_incumbent = (incumbent - lb[idx]) / (ub[idx] - lb[idx])

                    y_real[idx] = w_bias
                    y_norm_real[idx] = norm_bias
                    y_incumbent[idx] = norm_incumbent
                else:
                    y_norm_real[idx] = 1 if ub[idx] != 0 or lb[idx] != 0 else 0
                    y_incumbent[idx] = 1 if ub[idx] != 0 or lb[idx] != 0 else 0

                assert 0 <= y_incumbent[idx] <= 1 and 0 <= y_norm_real[idx] <= 1

            feat_var[idx] = [isb, isc, isi, node_data["obj_coeff"], graph.degree[node]]
            obj[idx] = node_data["obj_coeff"]

        elif node_data["bipartite"] == 1:  # Constraint node
            idx = node_data["index"]
            index_con.append(0)

            rhs[idx] = node_data["rhs"]
            kind = float(node_data["kind"] == "L")
            con_kind[idx] = kind
            feat_con[idx] = [kind, node_data["rhs"], graph.degree[node]]

    for s, t, edge_data in graph.edges(data=True):
        s_data = graph.nodes[s]
        t_data = graph.nodes[t]

        if s_data["bipartite"] == 1:
            con_idx = s_data["index"]
            var_idx = t_data["index"]
        else:
            con_idx = t_data["index"]
            var_idx = s_data["index"]

        edge_list_con.append([con_idx, var_idx])
        edge_features_con.append([edge_data["coeff"]])
        edge_list_var.append([var_idx, con_idx])
        edge_features_var.append([edge_data["coeff"]])

    data = BipartiteData()
    data.obj = torch.tensor(obj, dtype=torch.float)
    data.is_binary = torch.tensor(is_binary, dtype=torch.bool)
    data.lb = torch.tensor(lb, dtype=torch.float)
    data.ub = torch.tensor(ub, dtype=torch.float)

    if is_labeled:
        data.y_real = torch.tensor(y_real, dtype=torch.float)
        data.y_norm_real = torch.tensor(y_norm_real, dtype=torch.float)
        data.y_incumbent = torch.tensor(y_incumbent, dtype=torch.float)

    data.var_node_features = torch.tensor(feat_var, dtype=torch.float)
    data.con_node_features = torch.tensor(feat_con, dtype=torch.float)
    data.rhs = torch.tensor(rhs, dtype=torch.float)
    data.con_kind = torch.tensor(con_kind, dtype=torch.float)
    data.edge_features_con = torch.tensor(edge_features_con, dtype=torch.float)
    data.edge_features_var = torch.tensor(edge_features_var, dtype=torch.float)
    data.num_var_nodes = torch.tensor(num_var_nodes)
    data.num_con_nodes = torch.tensor(num_con_nodes)
    data.edge_index_var = torch.tensor(edge_list_var, dtype=torch.long).t().contiguous()
    data.edge_index_con = torch.tensor(edge_list_con, dtype=torch.long).t().contiguous()

    data.index_con = torch.tensor(index_con, dtype=torch.long)
    data.index_var = torch.tensor(index_var, dtype=torch.long)

    if is_labeled:
        Ax, violation = constraint_valuation(
            data.y_incumbent,
            data.edge_index_var,
            data.edge_features_var,
            data.rhs,
            data.lb,
            data.ub,
            data.con_kind,
            (data.num_var_nodes, data.num_con_nodes),
        )
        data.Ax = Ax
        if violation.max() > 1e-5:
            print(">>>", str(violation.max()))

    return data


# Preprocessing to create Torch dataset
class GraphDataset(InMemoryDataset):
    def __init__(
        self,
        prob_name,
        dt_type,
        dt_name,
        instance_dir,
        graph_dir,
        instance_names,
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):
        self.prob_name = prob_name
        self.dt_name = dt_name
        self.dt_type = dt_type
        self.instance_dir = instance_dir
        self.graph_dir = graph_dir
        self.instance_names = instance_names

        super(GraphDataset, self).__init__(
            str(graph_dir), transform, pre_transform, pre_filter
        )

        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return self.instance_names

    @property
    def processed_file_names(self):
        return [self.dt_name]

    def download(self):
        pass

    def process(self):
        data_list = []
        for i, instance_name in enumerate(self.instance_names):
            data_file = f"{instance_name}_data.pt"
            graph_path = self.graph_dir.joinpath(instance_name + "_labeled_graph.pkl")
            instance = self.instance_dir.joinpath(
                instance_name + INSTANCE_FILE_TYPES[self.prob_name]
            )

            if data_file in os.listdir(self.processed_dir):
                print(data_file, "available")
                data = torch.load(self.processed_dir + "/" + data_file)

            else:
                model = gp.read(str(instance))
                graph = nx.read_gpickle(graph_path)
                is_labeled = self.dt_type in ["train", "val"]
                data = create_data_object(
                    instance_name, model, graph, is_labeled, self.processed_dir
                )

            data_list.append(data)

        random.shuffle(data_list)
        all_data, slices = self.collate(data_list)
        torch.save((all_data, slices), self.processed_paths[0])

    def get(self, idx):
        data = super(GraphDataset, self).get(idx)
        return idx, data


class ModelDataset(InMemoryDataset):
    def __init__(self, data_dir: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_dir = data_dir

    def process(self):
        data_paths = (p for p in os.listdir(self.data_dir) if p.endswith(".pt"))
        data_paths = [os.path.join(self.data_dir, p) for p in data_paths]
        objs = [torch.load(f) for f in data_paths]
        random.shuffle(objs)
        self.data, self.slices = self.collate(objs)

    def get(self, idx):
        data = super().get(idx)
        return idx, data


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
