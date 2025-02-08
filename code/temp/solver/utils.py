import json
from tempfile import NamedTemporaryFile

import gurobipy as gp


def fix_var(inst, idxs, vals):
    assert len(idxs) == len(vals)
    bounds = {}
    vs = inst.getVars()
    for idx, val in zip(idxs, vals):
        v = vs[idx]
        bounds[idx] = (v.lb, v.ub)
        v.setAttr("lb", val)
        v.setAttr("ub", val)
    inst.update()
    return bounds


def unfix_var(inst, idxs, bounds):
    assert len(idxs) == len(bounds)
    vs = inst.getVars()
    for i, (lb, ub) in zip(idxs, bounds):
        vs[i].setAttr("lb", lb)
        vs[i].setAttr("ub", ub)


def get_iis_vars(inst):
    try:
        inst.computeIIS()
    except Exception as e:
        print(e)
        if "Cannot compute IIS on a feasible model" in str(e):
            return set()
        raise e

    with NamedTemporaryFile(suffix=".ilp", mode="w+") as f:
        inst.write(f.name)
        f.seek(0)
        return set(f.read().split())


def set_starts(inst, starts):
    vs = inst.getVars()
    for i, s in starts.items():
        vs[i].setAttr("lb", s)


def solve_inst(inst):
    vs = inst.getVars()
    inst.optimize()
    return inst.getAttr("X", vs)


def repair(inst, fixed: set, bounds: dict):
    old_iis_method = getattr(inst, "IISMethod", -1)
    inst.setParam("IISMethod", 0)

    vs = inst.getVars()
    ns = inst.getAttr("varName", vs)
    name_to_idx = {n: i for i, n in enumerate(ns)}

    freed = set()
    while iis_var_names := get_iis_vars(inst):
        for n in iis_var_names:
            if n not in name_to_idx:
                continue

            var_idx = name_to_idx[n]
            if var_idx not in fixed:
                continue

            if var_idx in freed:
                continue

            lb, ub = bounds[var_idx]
            vs[var_idx].lb = lb
            vs[var_idx].ub = ub
            freed.add(var_idx)

    inst.setParam("IISMethod", old_iis_method)
    return freed


def with_lic(m, path="gb.lic"):
    with open(path) as f:
        env = gp.Env(params=json.load(f))
    return m.copy(env=env)
