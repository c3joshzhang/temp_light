from typing import List

import gurobipy as gp


class VarInfo:
    def __init__(self, vs, lbs, ubs, types, const):
        assert len(vs) == len(lbs) == len(ubs) == len(types) == len(const)
        self.n = len(self.lbs)
        self.vs = vs
        self.lbs = lbs
        self.ubs = ubs
        self.types = types
        self.const = const


class ConInfo:

    LT = 0
    EQ = 1
    GT = 2

    def __init__(self, v_info, lhs_p, lhs_c, rhs, types, const):
        self.v_info = v_info
        self.n = len(rhs)
        self.lhs_p = lhs_p
        self.lhs_c = lhs_c
        self.rhs = rhs
        self.types = types
        self.const = const


class ObjInfo:

    def __init__(self, v_info, coeffs, sense):
        self.v_info = v_info
        self.coeffs = coeffs
        self.sense = sense


def relax_solve(var_info, con_info, obj_info, build_kwargs):

    model = gp.Model(**build_kwargs)
    model.feasRelaxS(relaxobjtype=0, minrelax=False, vrelax=False, crelax=True)

    vs = []
    for i in range(var_info.n):
        if var_info.const[i]:
            vs.append(var_info.vs[i])
            continue
        v = model.addVar(
            lb=var_info.lbs[i], ub=var_info.ubs[i], vtype=var_info.types[i]
        )
        vs.append(v)

    for i in range(con_info.n):

        if con_info.const[i]:
            continue

        const_by_var = all(var_info.const[j] for j in con_info.lhs_p[i])
        if const_by_var:
            continue

        vs_in_con = [vs[j] for j in con_info.lhs_p[i]]
        ks_in_con = con_info.lhs_c[i]
        lhs = sum(k * v for k, v in zip(ks_in_con, vs_in_con))
        rhs = con_info.rhs[i]

        if con_info.types[i] == ConInfo.LT:
            model.addConstr(lhs <= rhs)
        elif con_info.types[i] == ConInfo.EQ:
            model.addConstr(lhs == rhs)
        elif con_info.types[i] == ConInfo.GR:
            model.addConstr(lhs >= rhs)
        else:
            raise NotImplementedError()

    obj_val = sum(vs[i] * obj_info.coeffs[i] for i in range(len(vs)))
    model.setObjective(obj_val, obj_info.sense)

    model.optimize()
    # TODO: cast B or I to int
    return [v.X for v in vs], model.ObjVal


def fennel_partition(): ...
def cross(): ...
def large_neighborhood_search(): ...


def solve_exact(
    n,
    m,
    k,
    site,
    value,
    constraint,
    constraint_type,
    coefficient,
    time_limit,
    obj_type,
    now_sol,
    now_col,
    constr_flag,
    lower_bound,
    upper_bound,
    value_type,
):
    """
    Function Explanation:
    This function solves a problem instance using the SCIP solver based on the provided parameters.

    Parameter Explanation:
    - n: The number of decision variables in the problem instance.
    - m: The number of constraints in the problem instance.
    - k: k[i] indicates the number of decision variables in the i-th constraint.
    - site: site[i][j] indicates which decision variable the j-th decision variable in the i-th constraint is.
    - value: value[i][j] represents the coefficient of the j-th decision variable in the i-th constraint.
    - constraint: constraint[i] represents the right-hand side value of the i-th constraint.
    - constraint_type: constraint_type[i] indicates the type of the i-th constraint, where 1 represents <= and 2 represents >=.
    - coefficient: coefficient[i] indicates the coefficient of the i-th decision variable in the objective function.
    - time_limit: The maximum solving time.
    - obj_type: Specifies whether the problem is a maximization or minimization problem.
    - now_sol: The current solution.
    - now_col: Dimensionality reduction flags for decision variables.
    - constr_flag: Dimensionality reduction flags for constraints.

    - lower_bound: Lower bounds for decision variables.
    - upper_bound: Upper bounds for decision variables.
    - value_type: The type of decision variables (e.g., integer or continuous variables).
    """
    # Get the start time
    begin_time = time.time()

    # Define the solver model
    with open("gb.lic") as f:
        env = gp.Env(params=json.load(f))

    model = gp.Model("Gurobi", env=env)
    model.feasRelaxS(0, False, False, True)

    # Set up variable mappings
    site_to_new = {}
    new_to_site = {}
    new_num = 0

    # Define new_num decision variables x[]
    x = []
    for i in range(n):
        if now_col[i] == 1:
            site_to_new[i] = new_num
            new_to_site[new_num] = i
            new_num += 1
            if value_type[i] == "B":
                x.append(
                    model.addVar(lb=lower_bound[i], ub=upper_bound[i], vtype=GRB.BINARY)
                )
            elif value_type[i] == "C":
                x.append(
                    model.addVar(
                        lb=lower_bound[i], ub=upper_bound[i], vtype=GRB.CONTINUOUS
                    )
                )
            else:
                x.append(
                    model.addVar(
                        lb=lower_bound[i], ub=upper_bound[i], vtype=GRB.INTEGER
                    )
                )

    # Set the objective function and optimization goal (maximize/minimize)
    coeff = 0
    for i in range(n):
        if now_col[i] == 1:
            coeff += x[site_to_new[i]] * coefficient[i]
        else:
            coeff += now_sol[i] * coefficient[i]

    if obj_type == "maximize":
        model.setObjective(coeff, GRB.MAXIMIZE)
    else:
        model.setObjective(coeff, GRB.MINIMIZE)

    # Add m constraints
    for i in range(m):
        if constr_flag[i] == 0:
            continue
        constr = 0
        flag = 0
        for j in range(k[i]):
            if now_col[site[i][j]] == 1:
                constr += x[site_to_new[site[i][j]]] * value[i][j]
                flag = 1
            else:
                constr += now_sol[site[i][j]] * value[i][j]

        if flag == 1:
            if constraint_type[i] == 1:
                model.addConstr(constr <= constraint[i])
            else:
                model.addConstr(constr >= constraint[i])
        else:
            if constraint_type[i] == 1:
                if constr > constraint[i]:
                    print("QwQ")
                    print(constr, constraint[i])
                    # print(now_col)
            else:
                if constr < constraint[i]:
                    print("QwQ")
                    print(constr, constraint[i])
                    # print(now_col)

    # Set the maximum solving time
    model.setParam("TimeLimit", max(time_limit - (time.time() - begin_time), 0))

    # Optimize the solution
    model.optimize()
    # print(time.time() - begin_time)
    try:
        new_sol = []
        for i in range(n):
            if now_col[i] == 0:
                new_sol.append(now_sol[i])
            else:
                if value_type[i] == "C":
                    new_sol.append(x[site_to_new[i]].X)
                else:
                    new_sol.append((int)(x[site_to_new[i]].X))

        return new_sol, model.ObjVal
    except:
        return -1, -1
