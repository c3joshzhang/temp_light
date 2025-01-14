from typing import Dict

import gurobipy as gp


class VarInfo:

    def __init__(self, lbs, ubs, types):
        assert len(lbs) == len(ubs) == len(types)
        self.lbs = lbs
        self.ubs = ubs
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


class ConInfo:

    ENUM_TO_OP = {"<=": 1, ">=": 2, "==": 3}
    OP_TO_ENUM = {1: "<=", 2: ">=", 3: "=="}

    def __init__(self, lhs_p, lhs_c, rhs, types):
        self.lhs_p = lhs_p
        self.lhs_c = lhs_c
        self.rhs = rhs
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


class ObjInfo:

    def __init__(self, ks: Dict[int, float], sense: int):
        self.ks = ks
        self.sense = sense

    def __repr__(self):
        return f"[{self.ks}, {self.sense}]"


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
