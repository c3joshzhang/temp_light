import itertools
import random
from typing import List

import gurobipy as gp
import numpy as np
import torch
from joblib import Parallel, delayed

from temp.data.info import ConInfo, ModelInfo, VarInfo


def get_lhs_matrix(n_var: int, con_info: ConInfo):
    shape = (con_info.n, n_var)

    row_idxs = [np.full(len(p), i, dtype=np.int64) for i, p in enumerate(con_info.lhs_p)]
    row_idxs = np.concatenate(row_idxs)
    col_idxs = np.concatenate(con_info.lhs_p)

    vals = np.concatenate(con_info.lhs_c)
    idxs = np.stack([row_idxs, col_idxs])
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
    lhs = lhs.to(torch.float32)
    shift = lhs @ torch.as_tensor(dv).float()
    return shift.numpy().squeeze()


def get_obj_shift(ks, dv):
    dv = dv.squeeze() if len(dv.shape) == 2 else dv
    shift = sum(k * dv[i] for i, k in ks.items())
    return shift


def shift_model_info(info: ModelInfo, var_shift, con_shift, obj_shift):
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

        info.var_info.lbs[i] = min(max(info.var_info.lbs[i], 0.0), 1.0)
        info.var_info.ubs[i] = max(min(info.var_info.ubs[i], 1.0), 0.0)

    for i, c_shift in enumerate(con_shift):
        if c_shift == 0:
            continue
        info.con_info.rhs[i] += c_shift

    return info


def add_objective_constraint(info, ratio=0.01):
    old_sols = info.var_info.sols
    sol_idx = random.randint(0, len(old_sols) - 1)
    rand_sol = old_sols[sol_idx]

    rhs_ub = rand_sol[0] * (1 + random.random() * ratio)
    rhs_lb = rand_sol[0] * (1 - random.random() * ratio)
    lhs_p = list(info.obj_info.ks)
    lhs_c = [info.obj_info.ks[i] for i in lhs_p]

    old_obj_vals = info.var_info.sols[:, 0]
    val_between_lb_ub = (old_obj_vals >= rhs_lb) & (old_obj_vals <= rhs_ub)
    new_sols = old_sols[val_between_lb_ub].copy()

    info.con_info.lhs_p.append(lhs_p)
    info.con_info.lhs_c.append(lhs_c)
    info.con_info.rhs.append(rhs_ub)
    info.con_info.types.append(info.con_info.ENUM_TO_OP["<="])

    info.con_info.lhs_p.append(lhs_p)
    info.con_info.lhs_c.append(lhs_c)
    info.con_info.rhs.append(rhs_lb)
    info.con_info.types.append(info.con_info.ENUM_TO_OP[">="])

    info.var_info.sols = new_sols
    return info


def add_redundant_constraint(info: ModelInfo, prob=0.2, ratio=0.1):
    vals = info.var_info.sols[0, 1:]
    n_redundant = int(info.con_info.n * ratio)
    n_vars_in_c = int(len(vals) * prob)
    rand_lhs_ps = np.random.choice(len(vals), (n_redundant, n_vars_in_c))
    rand_lhs_cs = np.random.random(rand_lhs_ps.shape)

    added_con_info = ConInfo([], [], [], [])
    ge = ConInfo.ENUM_TO_OP[">="]
    le = ConInfo.ENUM_TO_OP["<="]
    added_con_types = np.random.choice([ge, le], n_redundant)
    perturb_ratios = np.random.random(n_redundant)
    perturb_ratios[added_con_types == le] *= -1

    added_con_info.lhs_p = rand_lhs_ps.tolist()
    added_con_info.lhs_c = rand_lhs_cs.tolist()
    added_con_info.types = added_con_types.tolist()
    added_con_info.rhs = [None] * n_redundant
    lhs_m = get_lhs_matrix(len(vals), added_con_info)

    rhs = lhs_m @ torch.as_tensor(vals)
    rhs = (torch.as_tensor(perturb_ratios) + 1) * rhs
    added_con_info.rhs = rhs.tolist()

    info.con_info.lhs_p.extend(added_con_info.lhs_p)
    info.con_info.lhs_c.extend(added_con_info.lhs_c)
    info.con_info.rhs.extend(added_con_info.rhs)
    info.con_info.types.extend(added_con_info.types)
    info.var_info.sols = info.var_info.sols[:1]
    return info


def replace_with_eq_aux_var(info: ModelInfo, prob=0.2, ratio=0.2):
    
    new_lhs_p = []
    new_lhs_c = []
    new_op_types = []
    new_rhs = []

    n_aux_var = 0
    new_lbs = []
    new_ubs = []
    new_var_types = []

    n_old_con = info.con_info.n
    n_old_var = info.var_info.n

    aux_lhs_p = []
    aux_lhs_c = []
    to_replace = np.random.random(n_old_con)
    for i in range(n_old_con):

        if to_replace[i] > prob:
            continue

        lhs_p = info.con_info.lhs_p[i]
        lhs_c = info.con_info.lhs_c[i]

        curr_new_lhs_p = []
        curr_new_lhs_c = []

        select_prob = np.random.random(len(lhs_p))
        keep = []
        for j in range(len(lhs_p)):
            if select_prob[j] > ratio:
                keep.append(j)
                continue
            curr_new_lhs_p.append(lhs_p[j])
            curr_new_lhs_c.append(lhs_c[j])

        if len(curr_new_lhs_p) <= 1:
            continue
        
        aux_lhs_p.append(curr_new_lhs_p.copy())
        aux_lhs_c.append(curr_new_lhs_c.copy())

        aux_var_idx = n_old_var + n_aux_var
        n_aux_var += 1

        info.con_info.lhs_p[i] = [lhs_p[j] for j in keep]
        info.con_info.lhs_p[i].append(aux_var_idx)
        info.con_info.lhs_c[i] = [lhs_c[j] for j in keep]
        info.con_info.lhs_c[i].append(1.0)

        aux_var_lb = sum(info.var_info.lbs[i] for i in curr_new_lhs_p)
        aux_var_ub = sum(info.var_info.ubs[i] for i in curr_new_lhs_p)

        new_lbs.append(aux_var_lb)
        new_ubs.append(aux_var_ub)
        new_var_types.append(gp.GRB.CONTINUOUS)

        curr_new_lhs_p.append(aux_var_idx)
        curr_new_lhs_c.append(-1)

        new_lhs_p.append(curr_new_lhs_p)
        new_lhs_c.append(curr_new_lhs_c)
        new_op_types.append(info.con_info.ENUM_TO_OP[">="])
        new_rhs.append(0.0)
        
        new_lhs_p.append(curr_new_lhs_p)
        new_lhs_c.append(curr_new_lhs_c)
        new_op_types.append(info.con_info.ENUM_TO_OP["<="])
        new_rhs.append(0.0)

    info.var_info.lbs.extend(new_lbs)
    info.var_info.ubs.extend(new_ubs)
    info.var_info.types.extend(new_var_types)

    info.con_info.lhs_p.extend(new_lhs_p)
    info.con_info.lhs_c.extend(new_lhs_c)
    info.con_info.types.extend(new_op_types)
    info.con_info.rhs.extend(new_rhs)

    old_sols = info.var_info.sols
    aux_sols = get_aux_solutions(old_sols[:, 1:], aux_lhs_p, aux_lhs_c)
    info.var_info.sols = np.hstack([old_sols[:, :1], aux_sols])
    return info


def get_aux_solutions(solutions, aux_lhs_p, aux_lhs_c):
    assert len(aux_lhs_p) == len(aux_lhs_c)

    n_vars = solutions.shape[1]
    n_auxs = len(aux_lhs_p)
    shape = (n_vars, n_auxs)

    idxs = [[], []]
    vals = []

    for aux_var_idx in range(n_auxs):
        ps = aux_lhs_p[aux_var_idx]
        cs = aux_lhs_c[aux_var_idx]
        for p, c in zip(ps, cs):
            idxs[0].append(p)
            idxs[1].append(aux_var_idx)
            vals.append(c)

    n_m = solutions
    m_k = torch.sparse_coo_tensor(idxs, vals, shape)
    n_k = torch.as_tensor(solutions).float() @ m_k.float()
    return np.hstack([n_m, n_k])


def shift_solution(info: ModelInfo, prob=0.2):
    vals = info.var_info.sols[0, 1:]
    shifted_vals = random_shift_binary_var_val(vals, info.var_info, prob=prob)
    lhs = get_lhs_matrix(info.var_info.n, info.con_info)
    var_shift = shifted_vals - vals
    con_shift = get_con_shift(lhs, var_shift)
    obj_shift = get_obj_shift(info.obj_info.ks, var_shift)
    shifted = shift_model_info(info, var_shift, con_shift, obj_shift)
    shifted.sols = info.var_info.sols[:1]
    return shifted


def augment_info(info: ModelInfo):
    assert info.var_info.sols is not None, "info must contain solution at var_info.sols"
    augments = [replace_with_eq_aux_var, add_objective_constraint, add_redundant_constraint, shift_solution]
    selector = np.random.randint(int(False), int(True) + 1, len(augments), dtype=bool)
    augmented = info.copy()
    for s, a in zip(selector, augments):
        augmented = a(augmented) if s else augmented
    return augmented


def parallel_augment_info(info: ModelInfo, prob=0.2, n=10, jobs=10) -> List[ModelInfo]:
    n_per_job = n // jobs
    augs = Parallel(n_jobs=jobs)(
        delayed(augment_info)(info, prob, n_per_job) for _ in range(jobs)
    )
    return list(itertools.chain(*augs))
