from typing import Dict

import gurobipy as gp


def _handle_inf(v):
    # if v == float('inf'):
    #     return 1e10
    # if v == -float('inf'):
    #     return -1e10
    return v


class VarInfo:

    def __init__(self, lbs, ubs, types):
        assert len(lbs) == len(ubs) == len(types)
        self.lbs = [_handle_inf(l) for l in lbs]
        self.ubs = [_handle_inf(u) for u in ubs]
        self.types = types

    def __repr__(self):
        info_str = []
        for i in range(self.n):
            info_str.append((self.lbs[i], self.ubs[i], self.types[i]))
        info_str = ", ".join(str(v) for v in info_str)
        return f"[{info_str}]"

    @property
    def n(self):
        return len(self.lbs)

    def subset(self, ids):
        ids = list(set(ids))
        assert min(ids) >= 0 and max(ids) < self.n
        sub_lbs = [self.lbs[i] for i in ids]
        sub_ubs = [self.ubs[i] for i in ids]
        sub_types = [self.types[i] for i in ids]
        new_old_mapping = {new_i: old_i for new_i, old_i in enumerate(ids)}
        return type(self)(sub_lbs, sub_ubs, sub_types), new_old_mapping


class ConInfo:

    ENUM_TO_OP = {"<=": 1, ">=": 2, "==": 3}
    OP_TO_ENUM = {1: "<=", 2: ">=", 3: "=="}

    def __init__(self, lhs_p, lhs_c, rhs, types):
        self.lhs_p = lhs_p
        self.lhs_c = lhs_c
        self.rhs = [_handle_inf(r) for r in rhs]
        self.types = types

    def __repr__(self):
        info_str = []
        for i in range(self.n):
            info_str.append(
                (
                    self.lhs_p[i],
                    self.lhs_c[i],
                    self.OP_TO_ENUM[self.types[i]],
                    self.rhs[i],
                )
            )
        info_str = ", ".join(str(v) for v in info_str)
        return f"[{info_str}]"

    @property
    def n(self):
        return len(self.rhs)

    def subset(self, ids):
        ids = list(set(ids))
        new_old_mapping = {new_i: old_i for new_i, old_i in enumerate(ids)}
        old_new_mapping = {old_i: new_i for new_i, old_i in enumerate(ids)}
        must_include = set(ids)
        sub_lhs_p = []
        sub_lhs_c = []
        sub_rhs = []
        sub_types = []
        for i in range(self.n):
            if not all(j in must_include for j in self.lhs_p[i]):
                continue

            sub_lhs_p.append([old_new_mapping[j] for j in self.lhs_p[i]])
            sub_lhs_c.append(self.lhs_c[i])
            sub_rhs.append(self.rhs[i])
            sub_types.append(self.types[i])
        return type(self)(sub_lhs_p, sub_lhs_c, sub_rhs, sub_types), new_old_mapping

    def expand(self, ids, ratio_threshold=0.5):
        ids = set(ids)
        expand_ids = ids.copy()
        for i in range(self.n):
            cnt = sum(j in ids for j in self.lhs_p[i])
            if len(self.lhs_p[i]) * ratio_threshold <= cnt:
                expand_ids.update(self.lhs_p[i])
        return list(expand_ids)


class ObjInfo:

    def __init__(self, ks: Dict[int, float], sense: int):
        self.ks = ks
        self.sense = sense

    def __repr__(self):
        return f"[{self.ks}, {self.sense}]"

    def subset(self, ids):
        new_old_mapping = {new_i: old_i for new_i, old_i in enumerate(ids)}
        new_ks = {i: self.ks[i] for i in ids if i in self.ks}
        return type(self)(new_ks, self.sense), new_old_mapping


class ModelInfo:
    def __init__(self, var_info: VarInfo, con_info: ConInfo, obj_info: ObjInfo):
        self.var_info = var_info
        self.con_info = con_info
        self.obj_info = obj_info

    def __repr__(self):
        return f"{self.var_info}\n{self.con_info}\n{self.obj_info}"

    @property
    def n(self):
        return self.var_info.n

    @property
    def m(self):
        return self.con_info.n

    @property
    def k(self):
        return [len(idxs) for idxs in self.con_info.lhs_p]

    @property
    def site(self):
        return self.con_info.lhs_p

    @property
    def value(self):
        return self.con_info.lhs_c

    @property
    def constraint(self):
        return self.con_info.rhs

    @property
    def constraint_type(self):
        return self.con_info.types

    @property
    def coefficient(self):
        return self.obj_info.ks

    @property
    def lower_bound(self):
        return self.var_info.lbs

    @property
    def upper_bound(self):
        return self.var_info.ubs

    @property
    def value_type(self):
        return self.var_info.types

    @staticmethod
    def _parse_var_info(model) -> VarInfo:
        vs = model.getVars()
        lbs = [v.lb for v in vs]
        ubs = [v.ub for v in vs]
        typs = [v.vtype for v in vs]
        info = VarInfo(lbs, ubs, typs)
        return info

    @staticmethod
    def _parse_con_info(model) -> ConInfo:
        cs = model.getConstrs()
        vs = model.getVars()
        var_map = {v.index: i for i, v in enumerate(vs)}

        rhs = []
        lhs_c = []
        lhs_p = []
        types = []

        for c in cs:
            op_enum = c.sense + "="
            types.append(ConInfo.ENUM_TO_OP[op_enum])
            rhs.append(c.rhs)
            row = model.getRow(c)
            lhs_p.append([var_map[row.getVar(j).index] for j in range(row.size())])
            lhs_c.append([row.getCoeff(j) for j in range(row.size())])
        return ConInfo(lhs_p, lhs_c, rhs, types)

    @staticmethod
    def _parse_obj_info(model) -> ObjInfo:
        vs = model.getVars()
        var_map = {v.index: i for i, v in enumerate(vs)}

        expr = model.getObjective()
        sense = model.ModelSense

        ks = {}
        for i in range(expr.size()):
            v = expr.getVar(i)
            ks[var_map[v.index]] = expr.getCoeff(i)
        return ObjInfo(ks, sense)

    @classmethod
    def from_model(cls, model):
        var_info = cls._parse_var_info(model)
        con_info = cls._parse_con_info(model)
        obj_info = cls._parse_obj_info(model)
        return cls(var_info, con_info, obj_info)

    def subset(self, ids):
        ids = list(set(ids))
        print(len(ids))
        ids = self.con_info.expand(ids, 0.5)
        print(len(ids))
        print("-" * 78)
        sub_var_info, new_old_mapping = self.var_info.subset(ids)
        sub_con_info, new_old_mapping = self.con_info.subset(ids)
        sub_obj_info, new_old_mapping = self.obj_info.subset(ids)
        return type(self)(sub_var_info, sub_con_info, sub_obj_info), new_old_mapping
