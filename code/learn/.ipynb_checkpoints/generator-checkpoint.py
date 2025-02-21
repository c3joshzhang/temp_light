import math
import random

import gurobipy as gp
import numpy as np

np.random.seed(0)
random.seed(0)


def maximum_independent_set_problem(
    num_nodes=64,
    edge_prob=0.3,
) -> gp.Model:
    edges = []
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if np.random.rand() < edge_prob:
                edges.append((i, j))

    m = gp.Model("maximum_independent_set")
    x = m.addVars(num_nodes, vtype=gp.GRB.BINARY, name="x")

    for i, j in edges:
        m.addConstr(x[i] + x[j] <= 1, name=f"edge_{i}_{j}")

    v = m.addVar(name="obj_val", vtype=gp.GRB.CONTINUOUS)
    v = gp.quicksum(x[i] for i in range(num_nodes))
    m.setObjective(v, gp.GRB.MAXIMIZE)
    m.update()
    return m
