import os
import glob
from pathlib import Path
import json
import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count
import time
import argparse
from typing import List, Dict
import pickle
import gurobipy as gp
from gurobipy import GRB
from multiprocessing import Process

from problem import setcover


import os
from typing import Callable


def generate_inst_in_process(
    generator: Callable[[], gp.Model], path: str, process_id: int, n_per_process: int
):
    process_path = os.path.join(path, str(process_id))
    process_path.mkdir(exist_ok=True)

    for i in range(n_per_process):
        try:
            model = generator()
            filename = f"instance_{process_id}_{i}.lp"
            model.write(os.path.join(process_path, filename))
        except Exception as e:
            print(f"Error in process {process_id} instance {i}: {str(e)}")


def generate_instances(
    generator: Callable[[], gp.Model], path: str, n_instances: int, n_processes: int
):
    """
    Generate problem instances in parallel and save them as LP files.

    Args:
        generator: Function that generates a Gurobi model
        path: Path to save the instances
        n_instances: Total number of instances to generate
        n_processes: Number of parallel processes to use
    """

    processes = []
    for i in range(n_processes):
        p = Process(
            target=generate_inst_in_process, args=(generator, path, i, n_instances)
        )
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    # all_files = []
    # for i in range(n_processes):
    #     temp_dir = Path(path).joinpath(f"temp_{i}")
    #     if temp_dir.exists():
    #         all_files.extend(list(temp_dir.glob("*.lp")))

    # # Sort files by their original number
    # all_files.sort(key=lambda x: int(x.stem.split('_')[1]))

    # # Rename files sequentially
    # for i, file_path in enumerate(all_files):
    #     new_name = Path(path).joinpath(f"instance_{i:05d}.lp")
    #     file_path.rename(new_name)

    # # Clean up temporary directories
    # for i in range(n_processes):
    #     temp_dir = Path(path).joinpath(f"temp_{i}")
    #     if temp_dir.exists():
    #         temp_dir.rmdir()

    # print(f"Generated {len(all_files)} instances in {path}")


generate_instances(setcover, "temp", 5, 5)
