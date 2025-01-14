from typing import List, Union

import numpy as np
import torch

from .feature import ConFeature, EdgFeature, VarFeature


class Instance:
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
        edges = ([], [])
        node_features = []
        edge_features = []

        for i in range(len(self.n)):

            # [v0, v1, v2, ... c0, c1, c2]
            var_xs = self.v_features[i].values
            con_xs = self.c_features[i].values

            # TODO: use dgl or pyg to remove the need of padding
            con_xs, var_xs = self._pad_features(con_xs, var_xs)
            xs = np.vstack([var_xs, con_xs])
            node_features.append(torch.as_tensor(xs))

            graph_edges = self.e_features.indices
            graph_edges = self._shift_idxs(*graph_edges)
            edges[0].append(torch.as_tensor(graph_edges[0]))
            edges[1].append(torch.as_tensor(graph_edges[1]))
            edge_features.append(torch.as_tensor(self.e_features.values))
        return edges, node_features, edge_features

    @staticmethod
    def _shift_idxs(con_idxs, var_idxs):
        # [v0, v1, v2, ... c0, c1, c2]
        return con_idxs + len(var_idxs), var_idxs

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
        for s in self.solutions:
            arr = np.array([int(v) for v in s])
            values.append(torch.as_tensor(arr))
        return values


def train(instances, model): ...
