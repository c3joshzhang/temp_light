import os
from typing import Callable, List

import gurobipy as gp
import numpy as np
from joblib import Parallel, delayed


def generate_problem(
    generator: Callable[[], gp.Model], save_path: str, prefix: str, n: int
):
    for i in range(n):
        model = generator()
        model.write(os.path.join(save_path, f"{prefix}_{i}.lp"))


def parallel_generate_problem(
    generator: Callable[[], gp.Model], save_path: str, n_insts: int, n_jobs: int
):
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    n_insts_per_job = n_insts // n_jobs
    Parallel(n_jobs=n_jobs)(
        delayed(generate_problem)(generator, save_path, i, n_insts_per_job)
        for i in range(n_jobs)
    )


def generate_solutions(model_paths: List[str], n=10, lic=None):
    for model_path in model_paths:
        try:
            print(f"READING AND SOLVING FOR {model_path}")
            print("*"*100)
            model = gp.read(model_path, env=gp.Env(params=lic) if lic else None)
            model.setParam("OutputFlag", 0)

            model.setParam("NoRelHeurTime", 10)
            model.setParam("TimeLimit", 120)
            model.setParam("MIPGap", 1e-3)

            model.setParam("PoolSolutions", n)
            model.setParam("PoolSearchMode", 2)
            model.optimize()

            vs = model.getVars()
            obj_val_and_sols = []

            if model.SolCount == 0:
                print(f"NO SOLUTION FOUND FOR {model_path}: DELETED")
                print("^"*100)
                os.remove(model_path)
                continue

            for i in range(model.SolCount):
                # TODO: setting solution number actually takes long time try to optimize
                # TODO: add function to round by tolerance
                model.params.SolutionNumber = i
                obj_val = model.PoolObjVal
                s = [obj_val] + [v.Xn for v in vs]
                obj_val_and_sols.append(s)

            with open(model_path.replace(".lp", ".npz"), "wb") as f:
                np.savez(f, solutions=obj_val_and_sols)
                print(f"SOLUTION SAVED {model_path}")
                print("^"*100)
            
        except Exception as e:
            print(f"FAILED {model_path}: {e}: DELETED")
            print("!"*100)
            os.remove(model_path)

        model.dispose()

def parallel_generate_solutions(model_path: str, n_jobs: int, *args, **kwargs):
    model_files_paths = [p for p in os.listdir(model_path) if p.endswith(".lp")]
    model_files_paths = [os.path.join(model_path, p) for p in model_files_paths]

    chunk_size = max((len(model_files_paths) // n_jobs), 1)
    Parallel(n_jobs=n_jobs)(
        delayed(generate_solutions)(
            model_files_paths[i * chunk_size : (i + 1) * chunk_size], *args, **kwargs
        )
        for i in range(n_jobs)
    )


def sequential_generate_solutions(model_path, *args, **kwargs):
    model_files_paths = [p for p in os.listdir(model_path) if p.endswith(".lp")]
    model_files_paths.sort()
    model_files_paths = [os.path.join(model_path, p) for p in model_files_paths]
    generate_solutions(model_files_paths, *args, **kwargs)
