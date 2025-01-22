import json
import random
from functools import partial
from typing import Callable, List, Tuple, Union

import dgl
import gurobipy as gp
import numpy as np
import torch

from .feature import ConFeature, EdgFeature, VarFeature
from .info import ModelInfo

__DEVICE_PTR = [torch.device("cuda" if torch.cuda.is_available() else "cpu")]


def SET_DEVICE(device):
    __DEVICE_PTR[0] = device


def GET_DEVICE():
    return __DEVICE_PTR[0]


class Inst:
    def __init__(
        self,
        v_features: List[VarFeature],
        c_features: List[ConFeature],
        e_features: List[EdgFeature],
        solutions: List[List[Union[float, int]]],
        distances: List[List[Union[float, int]]],
    ):
        assert len(v_features) == len(c_features) == len(e_features) == len(solutions)
        self.v_features = v_features
        self.c_features = c_features
        self.e_features = e_features
        self.solutions = solutions
        self.distances = distances

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
            # TODO: replace with hetro-graph

            arr = np.array(
                [[0, 1] if v == 1 else [1, 0] for v in s] + [[0, 0]] * n_constr
            )
            values.append(torch.as_tensor(arr, dtype=torch.int32))

        distances = []
        for i, d in enumerate(self.distances):

            # TODO: use accessor method
            n_constr = len(self.c_features[i].values)

            # TODO: remove padding
            # TODO: replace with hetro-graph
            arr = np.array(d + [[0, 0, 0]] * n_constr)
            distances.append(torch.as_tensor(arr, dtype=torch.int32))

        return values, distances


def get_train_mask(graph, ratio: float):
    """return mask for solution that can be included as hint"""
    assert 0.0 <= ratio <= 1.0
    is_var_flag_idx = 2
    var_flag = graph.ndata["feat"][:, is_var_flag_idx]
    size = list(var_flag[var_flag == 1].size())[0]
    n_include = int(round(size * ratio))
    mask = torch.zeros(size, dtype=torch.bool)
    idx = torch.randperm(size)[:n_include]
    mask[idx] = 1
    return mask


def get_solution_mask(
    mask: torch.Tensor, ratio: Union[float, Tuple[float, float]] = (0.5, 1.0)
) -> torch.Tensor:
    """get solution mask that has ratio between given ratio range that can be included as hint"""
    mask = mask.clone()
    ones_indices = torch.where(mask == 1)[0]
    ratio = (ratio, ratio) if isinstance(ratio, float) else ratio
    assert 0.0 <= ratio[0] <= 1.0 and 0.0 <= ratio[1] <= 1.0

    min_num_keep = int(round(len(ones_indices) * ratio[0]))
    max_num_keep = int(round(len(ones_indices) * ratio[1]))
    max_num_keep = max(min_num_keep, max_num_keep)
    num_keep = random.randint(min_num_keep, max_num_keep)

    if num_keep <= 0:
        mask[ones_indices] = 0
        return mask

    if num_keep >= len(ones_indices):
        return mask

    selected_indices = torch.randperm(len(ones_indices))[:num_keep]
    keep_indices = ones_indices[selected_indices]
    mask[ones_indices] = 0
    mask[keep_indices] = 1
    return mask


def get_mask_node_feature(node_feature, y, mask):
    """add mask and hint into feature"""
    node_feature_with_y = torch.hstack([node_feature, (y[:, 1] == 0).unsqueeze(1)])
    mask = torch.cat([mask, torch.zeros(len(y) - len(mask), dtype=torch.bool)])
    masked = node_feature_with_y.clone()
    masked[~mask, -1] = 0
    return torch.hstack([masked, mask.unsqueeze(1)]), mask


def build_graphs(inst):
    c_v_edges, v_c_edges, node_features, edge_features, _, _ = inst.xs
    ys, dists = inst.ys

    graphs = []
    for i in range(len(ys)):
        srcs = torch.cat([c_v_edges[i][:, 0], v_c_edges[i][:, 0]])
        dsts = torch.cat([c_v_edges[i][:, 1], v_c_edges[i][:, 1]])

        # TODO: replace with hetro-graph
        g = dgl.graph((srcs, dsts))
        g.ndata["feat"] = node_features[i]
        g.ndata["label"] = ys[i]
        g.ndata["distance"] = dists[i]
        g.edata["feat"] = torch.cat([edge_features[i], edge_features[i]])
        assert (g.in_degrees() == g.out_degrees()).all()
        graphs.append(g)

    return graphs


def build_inst(model_generator: Callable[[], gp.Model], n=1024, env=None) -> Inst:

    # if env is None:
    #     with open("gb.lic") as f:
    #         params = json.load(f)
    #         env = gp.Env(params=params)

    var_feats = []
    con_feats = []
    edg_feats = []
    solutions = []
    distances = []

    for _ in range(n):
        raw_m = model_generator()
        info = ModelInfo.from_model(raw_m)
        vf = VarFeature.from_info(info.var_info, info.obj_info)
        cf = ConFeature.from_info(info.con_info)
        ef = EdgFeature.from_info(info.con_info)

        m = raw_m if env is None else raw_m.copy(env=env)
        m.update()

        ss = []
        vs = m.getVars()
        m.optimize(partial(_collect_mip_sol, variables=vs, collection=ss))

        final_s = [v.X for v in vs]
        if ss and ss[-1] != final_s:
            ss.append(final_s)

        for s in ss:
            var_feats.append(vf)
            con_feats.append(cf)
            edg_feats.append(ef)
            solutions.append(s)
            d = []
            for v1, v2 in zip(s, ss[-1]):
                if v1 == v2:
                    d.append([0, 1, 0])
                    continue
                if v1 < v2:
                    d.append([0, 0, 1])
                    continue
                if v1 > v2:
                    d.append([1, 0, 0])
                    continue
            distances.append(d)

        raw_m.dispose()
        m.dispose()

    return Inst(var_feats, con_feats, edg_feats, solutions, distances)


def remove_redundant_nodes(g) -> None:
    to_remove = (g.in_degrees() == 0).nonzero().reshape(-1).int()
    g.remove_nodes(to_remove)


# TODO: take the objective value into consideration and weight the sample
def _collect_mip_sol(
    model: gp.Model, where: int, variables: List, collection: List
) -> None:
    if where == gp.GRB.Callback.MIPSOL:
        s = model.cbGetSolution(variables)
        collection.append(s)
