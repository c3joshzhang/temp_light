import random
from typing import List

import numpy as np

from .info import ConInfo, ObjInfo, VarInfo


class VarFeature:

    def __init__(self, values):
        raw_values = np.array(values)
        pos_encode = np.array([[random.random()] for _ in range(len(values))])
        self._values = np.hstack([raw_values, pos_encode])

    @property
    def values(self) -> np.ndarray:
        return self._values

    @classmethod
    def from_info(cls, v_info: VarInfo, o_info: ObjInfo):
        values = []
        for i in range(v_info.n):
            v = [o_info.ks.get(i, 0)]
            # [is_constraint, is_variable]
            v.extend([0, 1])
            v.extend([v_info.lbs[i], v_info.ubs[i]])
            v.extend(cls._type_encode(v_info.types[i]))
            values.append(v)
        return cls(values)

    @staticmethod
    def _type_encode(type_str) -> List[int]:
        # B, I, C
        if type_str == "B":
            return [1, 0, 0]
        if type_str == "I":
            return [0, 1, 0]
        if type_str == "C":
            return [0, 0, 1]


class ConFeature:

    def __init__(self, values):
        raw_values = np.array(values)
        pos_encode = np.array([[random.random()] for _ in range(len(values))])
        self._values = np.hstack([raw_values, pos_encode])

    @property
    def values(self) -> np.ndarray:
        return self._values

    @classmethod
    def from_info(cls, info: ConInfo):
        values = []
        for i in range(info.n):
            v = [info.rhs[i]]
            # [is_constraint, is_variable]
            v.extend([1, 0])
            v.extend(cls._type_encode(info.types[i]))
            values.append(v)
        return cls(values)

    @staticmethod
    def _type_encode(type_int) -> List[int]:
        encode = [0, 0, 0]
        encode[type_int - 1] = 1
        return encode


# TODO: use presolve and relax to collect more features
class PreFeature: ...


class EdgFeature:

    def __init__(self, srcs, dsts, vals):
        self._srcs = np.array(srcs)
        self._dsts = np.array(dsts)
        self._vals = np.array(vals)

    @property
    def indices(self):
        return self._srcs, self._dsts

    @property
    def values(self):
        return self._vals

    @classmethod
    def from_info(cls, info: ConInfo):
        con_srcs = []
        var_dsts = []
        vals = []
        for i in range(info.n):
            for j, n in enumerate(info.lhs_p[i]):
                con_srcs.append(i)
                var_dsts.append(n)
                vals.append(info.lhs_c[i][j])
        return cls(con_srcs, var_dsts, vals)
