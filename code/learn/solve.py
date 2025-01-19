from typing import Dict, List, Optional

import gurobipy as gp

from learn.info import ConInfo, ModelInfo


def build_partial_model(
    info: ModelInfo,
    model=None,
    const_vars: Optional[Dict[int, float]] = None,
    const_cons: Optional[List[int]] = None,
):

    model = model or gp.Model()

    const_vars = const_vars or {}
    const_cons = const_cons or {}

    vs = []
    for i in range(info.var_info.n):
        if i in const_vars:
            vs.append(const_vars[i])
            continue

        v = model.addVar(
            lb=info.var_info.lbs[i],
            ub=info.var_info.ubs[i],
            vtype=info.var_info.types[i],
        )
        vs.append(v)

    for i in range(info.con_info.n):

        satisfied = (i in const_cons) or all(
            j in const_vars for j in info.con_info.lhs_p[i]
        )
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

    obj_val = sum(vs[i] * info.obj_info.ks[i] for i in range(len(vs)))
    model.setObjective(obj_val, info.obj_info.sense)
    model.update()
    return model


import collections
from typing import List, Tuple

import numpy as np


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
