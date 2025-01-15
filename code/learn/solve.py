from typing import Dict, List, Optional

import gurobipy as gp

from learn.info import ConInfo, ModelInfo


def build_partial_model(
    info: ModelInfo,
    model=None,
    const_vars: Optional[Dict[str, float]] = None,
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
