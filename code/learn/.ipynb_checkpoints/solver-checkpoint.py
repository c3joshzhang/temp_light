import collections
from typing import Dict, List, Optional, Tuple

import gurobipy as gp
import numpy as np

from learn.info import ConInfo, ModelInfo


def build_partial_model(
    info: ModelInfo,
    model=None,
    const_vars: Optional[Dict[int, float]] = None,
):

    model = model or gp.Model()

    const_vars = const_vars or {}
    mapping = {}
    vs = []
    for i in range(info.var_info.n):

        if i in const_vars:
            v = model.addVar(
                vtype=info.var_info.types[i],
                ub=const_vars[i],
                lb=const_vars[i],
            )
            vs.append(v)
            continue

        mapping[len(mapping)] = i

        bounds = {}
        if info.var_info.lbs[i] != float("inf") and info.var_info.lbs[i] != -float(
            "inf"
        ):
            bounds["lb"] = info.var_info.lbs[i]
        if info.var_info.ubs[i] != float("inf") and info.var_info.ubs[i] != -float(
            "inf"
        ):
            bounds["ub"] = info.var_info.ubs[i]

        v = model.addVar(vtype=info.var_info.types[i], **bounds)
        vs.append(v)

    for i in range(info.con_info.n):

        satisfied = all(j in const_vars for j in info.con_info.lhs_p[i])
        if satisfied:
            continue

        vs_in_con = [vs[j] for j in info.con_info.lhs_p[i]]
        ks_in_con = info.con_info.lhs_c[i]
        lhs = sum(k * v for k, v in zip(ks_in_con, vs_in_con))
        rhs = info.con_info.rhs[i]

        con_type = info.con_info.types[i]
        if ConInfo.OP_TO_ENUM[con_type] == "<=":
            model.addConstr(lhs <= rhs)
            continue

        if ConInfo.OP_TO_ENUM[con_type] == "==":
            model.addConstr(lhs == rhs)
            continue

        if ConInfo.OP_TO_ENUM[con_type] == ">=":
            model.addConstr(lhs >= rhs)
            continue

    obj_val = sum(
        vs[i] * info.obj_info.ks[i] for i in range(len(vs)) if i in info.obj_info.ks
    )
    model.setObjective(obj_val, info.obj_info.sense)
    model.update()
    return model, mapping


# TODO: double check if partitions cover all the original nodes
def fennel_partition(
    edges: Tuple[List[int], List[int]], n_groups: int, alpha: float, gamma: float
) -> List[Tuple[List[int], List[int]]]:

    u_list, v_list = edges
    if not u_list or not v_list or len(u_list) != len(v_list):
        raise ValueError("Edges must be non-empty type of equal length lists")

    max_node = max(max(u_list), max(v_list))
    n_nodes = max_node + 1

    adjacency = [[] for _ in range(n_nodes)]
    for u, v in zip(u_list, v_list):
        adjacency[u].append(v)
        adjacency[v].append(u)

    visited = [False] * n_nodes
    order = []
    for start_node in range(n_nodes):
        if not visited[start_node]:
            queue = collections.deque([start_node])
            visited[start_node] = True

            while queue:
                curr = queue.popleft()
                order.append(curr)
                for neigh in adjacency[curr]:
                    if not visited[neigh]:
                        visited[neigh] = True
                        queue.append(neigh)

    group_of = [-1] * n_nodes
    group_size = [0] * n_groups

    for node in order:
        neighbor_count = [0] * n_groups
        for neigh in adjacency[node]:
            g = group_of[neigh]
            if g != -1:
                neighbor_count[g] += 1

        best_group = None
        best_cost = float("inf")
        for g in range(n_groups):
            cost_g = -neighbor_count[g] + alpha * ((group_size[g]) ** gamma)
            if cost_g < best_cost:
                best_cost = cost_g
                best_group = g

        group_of[node] = best_group
        group_size[best_group] += 1

    edges_in_group = [([], []) for _ in range(n_groups)]
    for u, v in zip(u_list, v_list):
        g_u = group_of[u]
        g_v = group_of[v]
        if g_u == g_v:
            edges_in_group[g_u][0].append(u)
            edges_in_group[g_u][1].append(v)

    return edges_in_group


import torch
from scipy.sparse import csr_matrix
from tqdm import tqdm


def get_constraint_side_matrices(x: List[float], con_info: ConInfo):

    n_con = len(con_info.lhs_p)
    n_var = len(x)

    idxs = [[], []]
    vals = []

    for con_idx in tqdm(range(n_con)):
        var_idxs = con_info.lhs_p[con_idx]
        var_cefs = con_info.lhs_c[con_idx]
        for var_idx, var_cef in zip(var_idxs, var_cefs):
            idxs[0].append(con_idx)
            idxs[1].append(var_idx)
            vals.append(var_cef)

    lhs = torch.sparse_coo_tensor(idxs, vals, (n_con, n_var))
    var_vals = np.array(x)[:, np.newaxis]
    rhs = np.array(con_info.rhs)[:, np.newaxis]
    return lhs, var_vals, rhs


def get_constraint_violations(lhs, vs, rhs, ops):
    lt_ops = ops == ConInfo.ENUM_TO_OP["<="]
    eq_ops = ops == ConInfo.ENUM_TO_OP["=="]
    gt_ops = ops == ConInfo.ENUM_TO_OP[">="]

    lhs_vs = lhs @ torch.as_tensor(vs).float()
    lhs_vs = lhs_vs.numpy()

    diff = lhs_vs - rhs
    violations = np.zeros_like(diff, dtype=bool)
    violations[lt_ops] = diff[lt_ops] <= 0
    violations[gt_ops] = diff[gt_ops] >= 0
    violations[eq_ops] = diff[eq_ops] == 0
    diff[violations] = 0
    return np.abs(diff)


def get_high_confidence_prediction_indices(ps, ratio):
    # TODO: replace with top_k
    n_remain = int(len(ps) * ratio)
    return sorted(range(len(ps)), key=lambda i: ps[i], reverse=True)[:n_remain]


def unfix_violation_variables(
    lhs, x, rhs, ops, var_info, con_info, violations, fixed: List[int]
):
    fixed = set(fixed)

    violations_idxs = np.where(violations != 0)[0]
    violations_idxs = sorted(violations_idxs, key=lambda idx: violations[idx])
    ops = ops.reshape(-1)

    for i in tqdm(violations_idxs):
        diff = violations[i]

        lhs_p = con_info.lhs_p[i]
        if diff == float("inf") or diff == float("-inf"):
            for p in lhs_p:
                if p not in fixed:
                    continue
                fixed.remove(p)

        lhs_c = con_info.lhs_c[i]
        op = ops[i]

        p_c = sorted(zip(lhs_p, lhs_c), key=lambda pc: pc[0] in fixed)
        for p, c in p_c:

            if diff <= 0:
                break

            # # TODO: add doc
            # # TODO: more concise
            if ConInfo.OP_TO_ENUM[op] == ">=":
                if c >= 0:
                    diff -= (var_info.ubs[p] - x[p]) * c
                else:
                    diff -= (var_info.lbs[p] - x[p]) * c

            if ConInfo.OP_TO_ENUM[op] == "<=":
                if c >= 0:
                    diff += (var_info.lbs[p] - x[p]) * c
                else:
                    diff += (var_info.ubs[p] - x[p]) * c

            if p in fixed:
                fixed.remove(p)
    return fixed
