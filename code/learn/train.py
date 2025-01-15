from typing import List, Union

import numpy as np
import torch

from .feature import ConFeature, EdgFeature, VarFeature


class Inst:
    def __init__(
        self,
        v_features: List[VarFeature],
        c_features: List[ConFeature],
        e_features: List[EdgFeature],
        solutions: List[List[Union[float, int]]],
    ):
        assert len(v_features) == len(c_features) == len(e_features) == len(solutions)
        self.v_features = v_features
        self.c_features = c_features
        self.e_features = e_features
        self.solutions = solutions

    @property
    def n(self):
        return len(self.solutions)

    # TODO: use consistent naming for getting the features
    @property
    def xs(self):

        n_var = []
        n_con = []
        c_v_edges = []
        v_c_edges = []
        node_features = []

        for i in range(self.n):

            # [v0, v1, v2, ... c0, c1, c2]
            var_xs = self.v_features[i].values
            con_xs = self.c_features[i].values

            # TODO: use dgl or pyg to remove the need of padding
            con_xs, var_xs = self._pad_features(con_xs, var_xs)
            n_var.append(len(var_xs))
            n_con.append(len(con_xs))
            xs = np.vstack([var_xs, con_xs])
            node_features.append(torch.as_tensor(xs, dtype=torch.float32))

            con_idxs, var_idxs = self.e_features[i].indices
            con_idxs, var_idxs = self._shift_idxs(con_idxs, var_idxs, len(var_xs))
            assert len(con_idxs) == len(var_idxs)

            cve = []
            vce = []

            n_edges = len(con_idxs)
            for i in range(n_edges):
                cve.append([con_idxs[i], var_idxs[i]])
                vce.append([var_idxs[i], con_idxs[i]])
            cve = torch.as_tensor(cve, dtype=torch.int)
            vce = torch.as_tensor(vce, dtype=torch.int)

            c_v_edges.append(cve)
            v_c_edges.append(vce)

        edge_features = [
            torch.as_tensor(f.values, dtype=torch.float32) for f in self.e_features
        ]
        return c_v_edges, v_c_edges, node_features, edge_features, n_var, n_con

    @staticmethod
    def _shift_idxs(con_idxs, var_idxs, n_vars):
        # [v0, v1, v2, ... c0, c1, c2]
        return con_idxs + n_vars, var_idxs

    @staticmethod
    def _pad_features(con_xs, var_xs):
        con_x_dim = con_xs.shape[1]
        var_x_dim = var_xs.shape[1]

        if con_x_dim < var_x_dim:
            pad_size = var_x_dim - con_x_dim
            con_xs = np.pad(
                con_xs,
                pad_width=((0, 0), (0, pad_size)),
                mode="constant",
                constant_values=0,
            )

        if var_x_dim < con_x_dim:
            pad_size = con_x_dim - var_x_dim
            var_xs = np.pad(
                var_xs,
                pad_width=((0, 0), (0, pad_size)),
                mode="constant",
                constant_values=0,
            )

        return con_xs, var_xs

    @property
    def ys(self):
        # TODO: handle other type of x
        values = []
        for i, s in enumerate(self.solutions):

            # TODO: use accessor method
            n_constr = len(self.c_features[i].values)

            # TODO: remove padding
            arr = np.array([int(v) for v in s] + [0] * n_constr)
            values.append(torch.as_tensor(arr, dtype=torch.int32))
        return values


def train(instances, model): ...
