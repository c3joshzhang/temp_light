import gurobipy as gp
import numpy as np
from gurobipy import GRB


def setcover(
    n_rows: int = 100,
    n_cols: int = 200,
    density: float = 0.1,
    min_cost: int = 5,
    max_cost: int = 15,
    seed: int = None,
) -> gp.Model:
    """
    Generate a random set cover instance using Gurobi.

    Args:
        n_rows: Number of elements to be covered (constraints)
        n_cols: Number of sets available (variables)
        density: Probability of an element being in a set
        min_cost: Minimum cost for each set
        max_cost: Maximum cost for each set
        seed: Random seed for reproducibility

    Returns:
        model: Gurobi model representing the set cover instance
    """
    if seed is not None:
        np.random.seed(seed)

    # Initialize model
    model = gp.Model("setcover")

    # Generate random binary matrix (1 if element i is in set j)
    matrix = np.random.binomial(1, density, size=(n_rows, n_cols))

    # Generate random costs for each set
    costs = np.random.randint(min_cost, max_cost + 1, size=n_cols)

    # Create binary variables for each set
    vars = model.addVars(
        n_cols, vtype=GRB.BINARY, name=[f"x{j}" for j in range(n_cols)]
    )

    # Add constraints: each element must be covered by at least one selected set
    for i in range(n_rows):
        model.addConstr(
            gp.quicksum(matrix[i, j] * vars[j] for j in range(n_cols)) >= 1,
            name=f"cover_{i}",
        )

    # Set objective: minimize total cost
    model.setObjective(
        gp.quicksum(costs[j] * vars[j] for j in range(n_cols)), GRB.MINIMIZE
    )

    # Store the problem data as model attributes for potential later use
    model._matrix = matrix
    model._costs = costs
    model._n_rows = n_rows
    model._n_cols = n_cols
    model._density = density
    model.update()
    return model
