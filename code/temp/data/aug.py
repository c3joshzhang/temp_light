import itertools
import random
from typing import List

import gurobipy as gp
import numpy as np
import torch
from joblib import Parallel, delayed

from .info import ConInfo, ModelInfo, VarInfo


def get_lhs_matrix(n_var: int, con_info: ConInfo) -> torch.Tensor:
    n_con = con_info.n
    shape = (n_con, n_var)

    idxs = [[], []]
    vals = []

    for con_idx in range(n_con):
        var_idxs = con_info.lhs_p[con_idx]
        var_cefs = con_info.lhs_c[con_idx]
        for var_idx, var_cef in zip(var_idxs, var_cefs):
            idxs[0].append(con_idx)
            idxs[1].append(var_idx)
            vals.append(var_cef)

    lhs = torch.sparse_coo_tensor(idxs, vals, shape)
    return lhs


def random_shift_binary_var_val(vals, var_info: VarInfo, prob: float = 0.2):
    shifted = vals.copy()
    for i, val in enumerate(vals):
        if var_info.types[i] != gp.GRB.BINARY:
            continue
        if random.random() > prob:
            continue
        shifted[i] = 1 - vals[i]
    return np.array(shifted)


def get_con_shift(lhs, dv):
    dv = dv[: np.newaxis] if len(dv.shape) == 1 else dv
    shift = lhs @ torch.as_tensor(dv).float()
    return shift.numpy().squeeze()


def get_obj_shift(ks, dv):
    dv = dv.squeeze() if len(dv.shape) == 2 else dv
    shift = sum(k * dv[i] for i, k in ks.items())
    return shift


def shift_model_info(info: ModelInfo, var_shift, con_shift, obj_shift):
    info = info.copy()
    var_shift = var_shift.squeeze() if len(var_shift.shape) == 2 else var_shift

    info.var_info.sols[:, 1:] += var_shift
    info.var_info.sols[:, 0] += obj_shift

    for i, v_shift in enumerate(var_shift):
        if v_shift == 0:
            continue

        info.var_info.lbs[i] += v_shift
        info.var_info.ubs[i] += v_shift

        if info.var_info.types[i] != gp.GRB.BINARY:
            continue

        info.var_info.lbs[i] = max(info.var_info.lbs[i], 0.0)
        info.var_info.ubs[i] = min(info.var_info.ubs[i], 1.0)

    for i, c_shift in enumerate(con_shift):
        if c_shift == 0:
            continue
        info.con_info.rhs[i] += c_shift

    return info


def augment_info(info: ModelInfo, prob=0.2, n=10):
    assert info.var_info.sols is not None, "info must contain solution at var_info.sols"
    augs = []
    for _ in range(n):
        vals = info.var_info.sols[0, 1:]
        shifted_vals = random_shift_binary_var_val(vals, info.var_info, prob=prob)
        lhs = get_lhs_matrix(info.var_info.n, info.con_info)
        var_shfit = shifted_vals - vals
        con_shift = get_con_shift(lhs, var_shfit)
        obj_shift = get_obj_shift(info.obj_info.ks, var_shfit)
        a = shift_model_info(info, var_shfit, con_shift, obj_shift)
        augs.append(a)
    return augs


def parallel_augment_info(info: ModelInfo, prob=0.2, n=10, jobs=10) -> List[ModelInfo]:
    n_per_job = n // jobs
    augs = Parallel(n_jobs=jobs)(
        delayed(augment_info)(info, prob, n_per_job) for _ in range(jobs)
    )
    return list(itertools.chain(*augs))


# def shift_model(model, var_shift, rhs_shift):
#     # ONLY USED FOR VALIDATION
#     var_shift = var_shift.squeeze() if len(var_shift.shape) == 2 else var_shift

#     shifted = model.copy()
#     vs = shifted.getVars()
#     # TODO: allow C and I variable bound change
#     for v, v_shift in zip(vs, var_shift):
#         if v_shift == 0:
#             continue
#         if v_shift > 0:
#             v.setAttr("lb", 1)
#             continue
#         if v_shift < 0:
#             v.setAttr("ub", 0)
#             continue

#     cs = shifted.getConstrs()
#     for c, c_shift in zip(cs, rhs_shift):
#         c.setAttr("rhs", c.rhs + c_shift)

#     shifted.update()
#     return shifted
