import itertools
import os
import pickle
import random
from functools import lru_cache
from typing import List

import gurobipy as gp
import numpy as np
import torch
from joblib import Parallel, delayed
from torch_geometric.data import Data, Dataset, InMemoryDataset
from tqdm import tqdm

from .graph import add_label, get_bipartite_graph
from .info import ModelInfo
from .preprocessing import constraint_valuation


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
        return 0

import os
class AugCache:
    def __init__(self, info, name, augment=None, size=5, life=10):
        self.info = info
        self.name = name

        max_retry = 100
        for i in range(max_retry):
            d = info_to_data(info)
            if d is None:
                continue
            self.data = info_to_data(info)
            self.data.instance_name = name
            break

        self.augment = augment
        self.size = size
        self.life = life
        self.augmented = [[0, None] for _ in range(size)]

    def get(self):
        if self.augment is None:
            return self.data

        idx = random.randint(-1, len(self.augmented) - 1)
        if idx == -1:
            return self.data

        counter, aug = self.augmented[idx]
        if counter == self.life:
            self.augmented[idx][1] = None
            self.augmented[idx][0] = 0

        if aug is None:
            max_retry = 10
            for i in range(max_retry):
                a = self.augment(self.info)
                d = info_to_data(a)
                if d is None:
                    # TODO: fix this
                    # print(f"augmentation {i} failed, {a.applied}")
                    continue
                d.instance_name = f"aug_{idx}_{self.name}"
                self.augmented[idx][1] = d
                break
        self.augmented[idx][0] += 1
        return self.augmented[idx][1]


import io
from contextlib import redirect_stdout
from functools import cached_property


class ModelGraphDataset(InMemoryDataset):
    def __init__(self, root, augment=None, *args, **kwargs):
        self._inst_names = self._get_inst_names(root)
        self._augment = augment
        self._augment_cache = [None for i in range(len(self._inst_names))]
        super().__init__(root, *args, **kwargs)

    def len(self):
        return len(self._inst_names)

    @property
    def inst_names(self):
        return list(self._inst_names)

    @property
    def raw_file_names(self):
        mdl_paths = [os.path.join(self.root, f"{n}.lp") for n in self._inst_names]
        sol_paths = [os.path.join(self.root, f"{n}.npz") for n in self._inst_names]
        return mdl_paths + sol_paths

    @property
    def processed_file_names(self):
        return [f"{n}.pkl" for n in self._inst_names]

    def process(self):
        chunk_size_limit = 500
        chunk = ([], [], [])
        save_paths = self.processed_paths
        for i, n in tqdm(
            list(enumerate(self._inst_names)), "(model, solution) => info"
        ):
            chunk[0].append(os.path.join(self.root, f"{n}.lp"))
            chunk[1].append(os.path.join(self.root, f"{n}.npz"))
            chunk[2].append(save_paths[i])
            if len(chunk[0]) >= chunk_size_limit or i == len(self._inst_names) - 1:
                parallel_to_info(*chunk)
                chunk = ([], [], [])

    def get(self, idx):
        if self._augment_cache[idx] is None:
            with open(self.processed_paths[idx], "rb") as f:
                info = pickle.load(f)
                name = os.path.basename(self.processed_paths[idx])
                name = os.path.splitext(name)[0]
            self._augment_cache[idx] = AugCache(info, name, self._augment)
        return idx, self._augment_cache[idx].get()

    @staticmethod
    def _get_inst_names(root):
        mdl_paths = sorted(p for p in os.listdir(root) if p.endswith(".lp"))
        sol_paths = sorted(p for p in os.listdir(root) if p.endswith(".npz"))
        assert len(mdl_paths) == len(sol_paths), (len(mdl_paths), len(sol_paths))
        assert set(mp[:-2] == sp[:-3] for mp, sp in zip(mdl_paths, sol_paths))
        lp_suffix_len = len(".lp")
        return [p[:-lp_suffix_len] for p in mdl_paths]


def to_info(lp_path, npz_path, save_path):
    with redirect_stdout(io.StringIO()):
        m = gp.read(lp_path)
        s = np.load(npz_path)["solutions"]
    info = ModelInfo.from_model(m)
    info.var_info.sols = s
    with open(save_path, "wb") as f:
        pickle.dump(info, f)


def parallel_to_info(lp_paths, npz_paths, save_paths, n_jobs=10):
    assert len(lp_paths) == len(npz_paths) == len(save_paths)
    Parallel(n_jobs=n_jobs)(
        delayed(to_info)(lp_p, npz_p, sp)
        for lp_p, npz_p, sp in zip(lp_paths, npz_paths, save_paths)
    )


def info_to_data(info: ModelInfo):
    os.info = info
    sol = info.var_info.sols
    g, _ = get_bipartite_graph(info)
    g = add_label(g, info, sol) if sol is not None else g
    data = create_data_object(g, sol is not None)
    return data


def parallel_info_to_data(infos: List[ModelInfo], jobs=10):
    to_data = lambda infos: [info_to_data(i) for i in infos]
    chunk_size = max(len(infos) // jobs, 1)
    res = Parallel(n_jobs=jobs)(
        delayed(to_data)(infos[i * chunk_size : (i + 1) * chunk_size])
        for i in range(jobs)
    )
    return list(itertools.chain(*res))


def sequential_info_to_data(infos: List[ModelInfo]):
    res = [info_to_data(i) for i in infos]
    return res


def create_data_object(graph, is_labeled=True) -> BipartiteData:
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

            assert ub[idx] >= lb[idx], (ub[idx], lb[idx])

            if is_labeled:
                w_bias = node_data["weighted_bias"]
                incumbent = node_data["incumbent"]
                value_range = abs(ub[idx] - lb[idx])

                if value_range > 1e-6:
                    norm_bias = (w_bias - lb[idx]) / value_range
                    norm_incumbent = (incumbent - lb[idx]) / value_range

                    if not (0 <= norm_bias <= 1):
                        w_bias = np.clip(w_bias, lb[idx], ub[idx])
                        norm_bias = (w_bias - lb[idx]) / value_range
                        
                    if not (0 <= norm_incumbent <= 1):
                        incumbent = np.clip(incumbent, lb[idx], ub[idx])
                        norm_incumbent = (incumbent - lb[idx]) / value_range

                    y_real[idx] = w_bias
                    y_norm_real[idx] = norm_bias
                    y_incumbent[idx] = norm_incumbent
                else:
                    y_norm_real[idx] = 1 if ub[idx] != 0 or lb[idx] != 0 else 0
                    y_incumbent[idx] = 1 if ub[idx] != 0 or lb[idx] != 0 else 0

                msg = (lb[idx], ub[idx], incumbent, w_bias)
                assert 0 <= y_incumbent[idx] <= 1 and 0 <= y_norm_real[idx] <= 1, msg

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
        os.data = data

        if violation.max() > (data.ub - data.lb).abs().max() * 0.01:
            # TODO: fix this
            print(">>>", str(violation.max()))
            return None
    return data
