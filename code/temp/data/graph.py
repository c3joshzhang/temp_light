import networkx as nx
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler


def get_bipartite_graph(info):
    vtype_mapping = {"C": "continuous", "B": "binary", "I": "integer"}

    n_vars = info.var_info.n
    var_names = [f"v_{i}" for i in range(info.var_info.n)]
    var_idxs = list(range(n_vars))
    var_domains = [vtype_mapping[t] for t in info.var_info.types]
    var_lb = [b for b in info.var_info.lbs]
    var_ub = [b for b in info.var_info.ubs]
    obj_multiplier = info.obj_info.sense
    obj_coeff = [info.obj_info.ks.get(i, 0) for i in range(info.var_info.n)]
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
