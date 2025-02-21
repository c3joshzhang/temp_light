import itertools
import random
from typing import List

import gurobipy as gp
import numpy as np
import torch
from joblib import Parallel, delayed

from temp.data.info import ConInfo, ModelInfo, VarInfo


def get_lhs_matrix(n_var: int, n_con: int, lhs_p: list, lhs_c: list):
    assert len(lhs_p) == len(lhs_c) == n_con
    shape = (n_con, n_var)
    row_idxs = [np.full(len(p), i, dtype=int) for i, p in enumerate(lhs_p)]
    row_idxs = np.concatenate(row_idxs)
    col_idxs = np.concatenate(lhs_p)

    assert len(row_idxs) == len(set(zip(row_idxs, col_idxs)))
    vals = np.concatenate(lhs_c)
    idxs = np.stack([row_idxs, col_idxs])
    lhs = torch.sparse_coo_tensor(idxs, vals, shape, dtype=torch.float)
    return lhs


def random_shift_binary_var_val(vals, var_info: VarInfo, prob: float = 0.2):
    shifted = vals.copy()
    for i, val in enumerate(vals):
        if var_info.types[i] != gp.GRB.BINARY:
            continue
        if random.random() > prob:
            continue
        shifted[i] = 1 - val
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

    if not con_shift.shape:
        con_shift = [con_shift]
    for i, c_shift in enumerate(con_shift):
        if c_shift == 0:
            continue
        info.con_info.rhs[i] += c_shift

    return info


def add_objective_constraint(info, ratio=0.01):
    old_sols = info.var_info.sols
    sol_idx = random.randint(0, len(old_sols) - 1)
    rand_sol = old_sols[sol_idx]

    rhs_ub = rand_sol[0] * np.round((1 + random.random() * ratio), 3)
    rhs_lb = rand_sol[0] * np.round((1 - random.random() * ratio), 3)

    rhs_lb = min(rhs_lb, rhs_ub)
    rhs_ub = max(rhs_lb, rhs_ub)

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

    if n_redundant == 0:
        return info

    n_vars_in_c = int(len(vals) * prob)
    rand_lhs_ps = [
        np.random.choice(len(vals), n_vars_in_c, replace=False)
        for _ in range(n_redundant)
    ]
    rand_lhs_cs = 0.5 - np.random.random((n_redundant, n_vars_in_c))

    added_con_info = ConInfo([], [], [], [])
    perturb_ratios = np.round(np.random.random(n_redundant) * 0.95, 3)

    added_con_info.lhs_p = [p.tolist() for p in rand_lhs_ps]
    added_con_info.lhs_c = rand_lhs_cs.tolist()
    added_con_info.types = np.ones(n_redundant, dtype=int) * ConInfo.ENUM_TO_OP["<="]
    added_con_info.rhs = np.zeros(n_redundant)

    lhs = get_lhs_matrix(
        info.var_info.n, added_con_info.n, added_con_info.lhs_p, added_con_info.lhs_c
    )
    diff = get_lhs_rhs_diff(lhs, info.var_info.sols[0, 1:], added_con_info.rhs)
    added_con_info.rhs += diff

    pos_rhs = added_con_info.rhs > 0
    neg_rhs = ~pos_rhs
    added_con_info.rhs[pos_rhs] *= 1 + perturb_ratios[pos_rhs]
    added_con_info.rhs[neg_rhs] *= 1 - perturb_ratios[neg_rhs]
    added_con_info.rhs = added_con_info.rhs.tolist()

    info.con_info.lhs_p.extend(added_con_info.lhs_p)
    info.con_info.lhs_c.extend(added_con_info.lhs_c)
    info.con_info.rhs.extend(added_con_info.rhs)
    info.con_info.types.extend(added_con_info.types)
    info.var_info.sols = info.var_info.sols[:1].copy()
    return info


def reduce_with_fixed_solution(info: ModelInfo, ratio=0.2):
    vals = info.var_info.sols[0, 1:]
    n_fixed = int(info.var_info.n * ratio)
    if n_fixed == 0:
        return info

    fixed_vars = set(np.random.choice(info.var_info.n, n_fixed, replace=False))
    var_idx_mapping = {}
    for i in range(len(vals)):
        if i in fixed_vars:
            continue
        var_idx_mapping[i] = len(var_idx_mapping)

    new_lhs_p = []
    new_lhs_c = []
    new_rhs = []
    new_types = []

    con_info = info.con_info
    for i in range(con_info.n):
        old_lhs_p = con_info.lhs_p[i]
        old_lhs_c = con_info.lhs_c[i]
        old_rhs = con_info.rhs[i]
        old_type = con_info.types[i]

        cur_lhs_p = []
        cur_lhs_c = []
        cur_types = []
        cur_rhs = old_rhs
        for j, c in zip(old_lhs_p, old_lhs_c):
            if j in fixed_vars:
                cur_rhs -= c * vals[j]
                continue
            cur_lhs_p.append(var_idx_mapping[j])
            cur_lhs_c.append(c)

        if len(cur_lhs_p) == 0:
            continue
        
        new_lhs_p.append(cur_lhs_p)
        new_lhs_c.append(cur_lhs_c)
        new_rhs.append(cur_rhs)
        new_types.append(old_type)

    con_info.lhs_p = new_lhs_p
    con_info.lhs_c = new_lhs_c
    con_info.rhs = new_rhs
    con_info.types = new_types

    new_lbs = [None for _ in range(len(var_idx_mapping))]
    new_ubs = [None for _ in range(len(var_idx_mapping))]
    new_types = [None for _ in range(len(var_idx_mapping))]

    var_info = info.var_info
    for i in range(var_info.n):
        if i in fixed_vars:
            continue
        new_i = var_idx_mapping[i]
        new_lbs[new_i] = var_info.lbs[i]
        new_ubs[new_i] = var_info.ubs[i]
        new_types[new_i] = var_info.types[i]

    var_info.lbs = new_lbs
    var_info.ubs = new_ubs
    var_info.types = new_types

    cur_obj = info.var_info.sols[0, 0]
    new_ks = {}
    obj_info = info.obj_info
    for i, k in obj_info.ks.items():
        if i not in fixed_vars:
            new_i = var_idx_mapping[i]
            new_ks[new_i] = k
            continue
        cur_obj -= k * vals[i]
    obj_info.ks = new_ks

    cur_sols = [None for _ in range(len(var_idx_mapping))]
    for old_i, new_i in var_idx_mapping.items():
        cur_sols[new_i] = vals[old_i]
    var_info.sols = np.hstack([[cur_obj], cur_sols])[np.newaxis, :]
    return info


def re_rank_solutions(info: ModelInfo):

    if len(info.var_info.sols) <= 1:
        return info
    
    binary_vars_in_obj = [i for i in info.obj_info.ks if info.var_info.types[i] == gp.GRB.BINARY]
    if not binary_vars_in_obj:
        return info
    
    sols = info.var_info.sols
    sense = info.obj_info.sense
    cur_idx = np.random.randint(1, len(sols))
    
    top_sol = sols[0, 1:]
    cur_sol = sols[cur_idx, 1:]

    top_obj_val = sols[0, 0]
    cur_obj_val = sols[cur_idx, 0]

    top_bin_vals = top_sol[binary_vars_in_obj]
    cur_bin_vals = cur_sol[binary_vars_in_obj]

    rand_ks = 2 * (0.5 - np.random.random(len(binary_vars_in_obj)))

    top_bin_total = (top_bin_vals * rand_ks).sum()
    cur_bin_total = (cur_bin_vals * rand_ks).sum()

    if top_bin_total == cur_bin_total:
        return info

    obj_val_diff = cur_obj_val - top_obj_val
    bin_val_diff = top_bin_total - cur_bin_total
    shift_ratio = obj_val_diff / bin_val_diff

    shift_ratio *= 1 + np.random.random() * 0.1 if sense == gp.GRB.MAXIMIZE else -np.random.random() * 0.1
    rand_ks *= shift_ratio

    all_vals = sols[:, 1:]
    obj_val_diffs = (all_vals[:, binary_vars_in_obj] * rand_ks).sum(axis=1)
    sols[:, 0] += obj_val_diffs

    sort_indices = np.argsort(sols[:, 0])
    if sense == gp.GRB.MAXIMIZE:
        sort_indices = sort_indices[::-1]
    sols = sols[sort_indices]

    for i, rand_k in zip(binary_vars_in_obj, rand_ks):
        info.obj_info.ks[i] += rand_k
    info.var_info.sols = sols
    
    info.con_info.lhs_p.append([k for k in info.obj_info.ks])
    info.con_info.lhs_c.append([c for c in info.obj_info.ks.values()])
    con_typ = "<=" if sense == gp.GRB.MAXIMIZE else ">="
    info.con_info.types.append(ConInfo.ENUM_TO_OP[con_typ])
    info.con_info.rhs.append(sols[0, 0])

    return info


def replace_eq_with_double_bound(info: ModelInfo, ratio=0.2):

    new_lhs_p = []
    new_lhs_c = []
    new_types = []
    new_rhs = []
    con_info = info.con_info

    for i, prob in enumerate(np.random.random(con_info.n)):
        if prob > ratio:
            continue
        if con_info.types[i] != ConInfo.ENUM_TO_OP["=="]:
            continue
        con_info.types[i] = ConInfo.ENUM_TO_OP["<="]
        new_lhs_p.append(con_info.lhs_p[i].copy())
        new_lhs_c.append(con_info.lhs_c[i].copy())
        new_types.append(ConInfo.ENUM_TO_OP[">="])
        new_rhs.append(con_info.rhs[i])

    con_info.lhs_p.extend(new_lhs_p)
    con_info.lhs_c.extend(new_lhs_c)
    con_info.types.extend(new_types)
    con_info.rhs.extend(new_rhs)
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
    for i, p in enumerate(np.random.random(n_old_con)):

        if p > prob:
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

        aux_var_lb = 0
        aux_var_ub = 0
        for p, c in zip(curr_new_lhs_p, curr_new_lhs_c):
            c_lb = c * info.var_info.lbs[p]
            c_ub = c * info.var_info.ubs[p]
            if c_ub >= c_lb:
                aux_var_ub += c_ub
                aux_var_lb += c_lb
            else:
                aux_var_ub += c_lb
                aux_var_lb += c_ub

        aux_var_lb = max(min(aux_var_lb, aux_var_ub), -info.var_info.inf)
        aux_var_ub = min(max(aux_var_lb, aux_var_ub), +info.var_info.inf)

        new_lbs.append(aux_var_lb)
        new_ubs.append(aux_var_ub)

        new_var_types.append(gp.GRB.CONTINUOUS)
        curr_new_lhs_p.append(aux_var_idx)
        curr_new_lhs_c.append(-1)

        new_lhs_p.append(curr_new_lhs_p)
        new_lhs_c.append(curr_new_lhs_c)
        new_op_types.append(info.con_info.ENUM_TO_OP["=="])
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
    aux_sols = np.clip(aux_sols, new_lbs, new_ubs)
    aux_sols = np.hstack([old_sols[:, 1:], aux_sols])
    info.var_info.sols = np.hstack([old_sols[:, :1], aux_sols])
    return info


def get_aux_solutions(solutions, aux_lhs_p, aux_lhs_c):
    # n: number of existing variables
    # m: number of existing solutions
    # k: number of aux variables, number of aux constraints
    # n + k is the existing variables and the aux variables

    assert len(aux_lhs_p) == len(aux_lhs_c)
    n_vars = solutions.shape[1]
    n_auxs = len(aux_lhs_p)

    m_n = solutions
    k_n = get_lhs_matrix(n_vars, n_auxs, aux_lhs_p, aux_lhs_c)
    m_k = torch.as_tensor(m_n, dtype=torch.float) @ k_n.T
    return m_k


def shift_solution(info: ModelInfo, prob=0.2):
    vals = info.var_info.sols[0, 1:]
    shifted_vals = random_shift_binary_var_val(vals, info.var_info, prob=prob)
    lhs = get_lhs_matrix(
        info.var_info.n, info.con_info.n, info.con_info.lhs_p, info.con_info.lhs_c
    )
    var_shift = shifted_vals - vals
    con_shift = get_con_shift(lhs, var_shift)
    obj_shift = get_obj_shift(info.obj_info.ks, var_shift)
    shifted = shift_model_info(info, var_shift, con_shift, obj_shift)
    shifted.sols = info.var_info.sols[:1]
    return shifted


def augment_info(info: ModelInfo):
    assert info.var_info.sols is not None, "info must contain solution at var_info.sols"
    augments = [
        re_rank_solutions,
        add_redundant_constraint,
        replace_eq_with_double_bound,
        reduce_with_fixed_solution,
        add_objective_constraint,
        replace_with_eq_aux_var,
        shift_solution,
    ]
    np.random.shuffle(augments)
    select_probs = np.random.random(len(augments))
    augmented = info.copy()
    applied = []
    for p, a in zip(select_probs, augments):
        if p >= 1/len(augments):
            continue
        augmented = a(augmented)
    augmented.applied = applied
    return augmented


def parallel_augment_info(info: ModelInfo, prob=0.2, n=10, jobs=10) -> List[ModelInfo]:
    n_per_job = n // jobs
    augs = Parallel(n_jobs=jobs)(
        delayed(augment_info)(info, prob, n_per_job) for _ in range(jobs)
    )
    return list(itertools.chain(*augs))


def get_constraint_violations(lhs, vs, rhs, ops):
    # lhs = get_lhs_matrix(info.var_info.n, info.con_info)
    # rhs = np.array(info.con_info.rhs)
    # ops = np.array(info.con_info.types)
    # violations = get_constraint_violations(lhs, info.var_info.sols[0, 1:], rhs, ops)
    lt_ops = ops == ConInfo.ENUM_TO_OP["<="]
    eq_ops = ops == ConInfo.ENUM_TO_OP["=="]
    gt_ops = ops == ConInfo.ENUM_TO_OP[">="]

    diff = get_lhs_rhs_diff(lhs, vs, rhs)
    good = np.zeros_like(diff, dtype=bool)
    good[lt_ops] = diff[lt_ops] <= 0
    good[gt_ops] = diff[gt_ops] >= 0
    good[eq_ops] = diff[eq_ops] == 0
    diff[good] = 0
    return diff


def get_lhs_rhs_diff(lhs, vs, rhs):
    lhs_vs = lhs @ torch.as_tensor(vs).float()
    lhs_vs = lhs_vs.numpy()
    diff = lhs_vs - rhs
    return diff
