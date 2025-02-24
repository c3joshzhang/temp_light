import gurobipy as gp
import numpy as np
from gurobipy import GRB


def setcover(
    n_rows: int = 128,
    n_cols: int = 256,
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


def lot_sizing(
    n_periods: int = 32,
    n_items: int = 32,
    final_demand_low: int = 10,
    final_demand_high: int = 100,
    slack_low: int = 10,
    slack_high: int = 50,
    prod_cost_low: int = 1,
    prod_cost_high: int = 10,
    hold_cost_low: int = 1,
    hold_cost_high: int = 5,
    setup_cost_low: int = 50,
    setup_cost_high: int = 200,
    bom_ratio_low: int = 1,
    bom_ratio_high: int = 3,
    scale: float = 1e-3,
    cap_factor_final: float = 1.5,  # extra capacity multiplier for final product
    cap_factor: float = 1.2,        # extra capacity multiplier for intermediate items
    mo_factor: float = 0.8,         # multiplier to relax the minimum order requirement
    batch_levels: list = None,      # list of item indices that must be produced in batches
    batch_sizes: list = None,       # if provided as a list of length n_items, batch_sizes[i] is the batch size for item i
    final_inv_factor: float = 2.0,   # penalty multiplier for final inventory
    seed: int = None,
) -> gp.Model:
    """
    Generate a multi-level capacitated lot-sizing instance with a BOM and batch production.
    
    Structure:
      - There are n_items items arranged in a chain:
            * Item (n_items-1) is the final product and receives random external demand.
            * Items 0,..., n_items-2 are intermediate items (with zero external demand).
      - For each period t:
            * The final product's external demand is generated uniformly in [final_demand_low, final_demand_high].
            * For the final product, raw capacity is defined as:
                     cap_raw[t] = final_demand[t] + (final_demand[t+1] if t < n_periods-1 else 0) + slack_final[t],
                     mo_raw[t]  = final_demand[t] + (final_demand[t+1] if t < n_periods-1 else 0).
              Then, capacity[t, n_items-1] = cap_raw[t] * cap_factor_final and
                     min_order[t, n_items-1] = mo_raw[t].
      - For lower items (i = n_items-2 downto 0), capacity and minimum order are computed as:
                     capacity[t, i] = bom_ratio[i+1] * capacity[t, i+1] * cap_factor,
                     min_order[t, i] = bom_ratio[i+1] * min_order[t, i+1] * mo_factor.
      - The arrays demand, capacity, and min_order are scaled by 'scale'.
    
    Decision variables for each period t and item i:
      - x[t,i]: production quantity (continuous, ≥ 0)
      - I[t,i]: inventory level at end of period t (continuous, ≥ 0)
      - y[t,i]: binary setup variable (1 if production is set up in period t for item i)
    
    Batch Production:
      - For any item i in batch_levels, production must occur in multiples of a given batch size.
        An auxiliary integer variable z[t,i] is introduced so that:
              x[t,i] = batch_size[i] * z[t,i].
    
    Inventory Balance:
      - For the final product (i = n_items-1):
            t = 0:  x[0,i] - I[0,i] = external demand[0]
            t ≥ 1:  I[t-1,i] + x[t,i] - I[t,i] = external demand[t]
      - For intermediate items (i = 0,..., n_items-2):
            t = 0:  x[0,i] - I[0,i] = bom_ratio[i+1] * x[0,i+1]
            t ≥ 1:  I[t-1,i] + x[t,i] - I[t,i] = bom_ratio[i+1] * x[t,i+1]
    
    Production Bounds:
         x[t,i] ≤ capacity[t,i] * y[t,i]
         x[t,i] ≥ min_order[t,i] * y[t,i]
    
    Note:
      - Final inventory is now free (I[t,i] ≥ 0) even in period n_periods-1.
      - An extra penalty cost is added in the objective for final period inventory:
            final_inv_factor * hold_cost[i] * I[n_periods-1,i].
    
    Args:
        n_periods: Number of time periods.
        n_items: Number of items (BOM levels).
        final_demand_low, final_demand_high: Range for external demand for the final product.
        slack_low, slack_high: Range for additional slack (for the final product).
        prod_cost_low, prod_cost_high: Production cost per unit range (per item).
        hold_cost_low, hold_cost_high: Holding cost per unit range (per item).
        setup_cost_low, setup_cost_high: Fixed setup cost range (per item).
        bom_ratio_low, bom_ratio_high: Range for BOM ratios for items 1,..., n_items-1.
        scale: Scaling factor applied to demand, capacity, and minimum order.
        cap_factor_final: Multiplier to boost capacity of the final product.
        cap_factor: Multiplier to boost capacity for intermediate items.
        mo_factor: Multiplier to relax the minimum order requirement.
        batch_levels: List of item indices (0 <= i < n_items) for which production is batched.
                      (Default is None, meaning no items are forced into batch production.)
        batch_sizes: If provided as a list (or array) of length n_items, then for any i in batch_levels,
                     x[t,i] must be an integer multiple of batch_sizes[i]. If not provided, for each i in
                     batch_levels a random batch size between 5 and 20 is generated.
        final_inv_factor: Multiplier to penalize final period inventory (in addition to normal holding cost).
        seed: Random seed for reproducibility.
        
    Returns:
        model: Gurobi model representing the instance.
    """
    if seed is not None:
        np.random.seed(seed)
    
    # If no batch_levels provided, use empty list.
    if batch_levels is None:
        batch_levels = np.random.choice(n_items, int(0.2 * n_items), replace=False).tolist()
    
    # Process batch_sizes.
    if batch_levels and batch_sizes is None:
        batch_sizes = {i: np.random.randint(2, 10) for i in batch_levels}
    elif batch_levels and isinstance(batch_sizes, list):
        batch_sizes = {i: batch_sizes[i] for i in batch_levels}
    
    # Generate external demand for the final product (item index n_items-1)
    final_demand = np.random.randint(final_demand_low, final_demand_high + 1, size=n_periods)
    
    # Build demand matrix: only final product gets external demand.
    demand = np.zeros((n_periods, n_items), dtype=float)
    demand[:, n_items - 1] = final_demand.astype(float)
    
    # Generate BOM ratios for items 1,..., n_items-1.
    bom_ratio = np.zeros(n_items, dtype=int)
    bom_ratio[0] = 0
    for i in range(1, n_items):
        bom_ratio[i] = np.random.randint(bom_ratio_low, bom_ratio_high + 1)
    
    # For final product, compute raw slack, capacity, and minimum order per period.
    slack_final = np.random.randint(slack_low, slack_high + 1, size=n_periods)
    capacity = np.empty((n_periods, n_items), dtype=float)
    min_order = np.empty((n_periods, n_items), dtype=float)
    for t in range(n_periods):
        next_demand = final_demand[t + 1] if t < n_periods - 1 else 0
        cap_raw = final_demand[t] + next_demand + slack_final[t]
        mo_raw = final_demand[t] + next_demand
        capacity[t, n_items - 1] = cap_raw * cap_factor_final
        min_order[t, n_items - 1] = mo_raw
    # For lower levels, back-solve capacity and min_order.
    for i in range(n_items - 2, -1, -1):
        for t in range(n_periods):
            capacity[t, i] = bom_ratio[i + 1] * capacity[t, i + 1] * cap_factor
            min_order[t, i] = bom_ratio[i + 1] * min_order[t, i + 1] * mo_factor
    
    # Apply scaling.
    demand *= scale
    capacity *= scale
    min_order *= scale
    
    # Generate cost parameters per item.
    prod_cost = np.random.randint(prod_cost_low, prod_cost_high + 1, size=n_items)
    hold_cost = np.random.randint(hold_cost_low, hold_cost_high + 1, size=n_items)
    setup_cost = np.random.randint(setup_cost_low, setup_cost_high + 1, size=n_items)
    
    model = gp.Model("lot_sizing")
    
    # Decision variables.
    x = model.addVars(n_periods, n_items, lb=0, vtype=GRB.CONTINUOUS, name="x")
    I = model.addVars(n_periods, n_items, lb=0, vtype=GRB.CONTINUOUS, name="I")
    y = model.addVars(n_periods, n_items, vtype=GRB.BINARY, name="y")
    
    # For batched items, add auxiliary integer variable z[t,i] and linking constraint.
    z = {}
    for i in batch_levels:
        for t in range(n_periods):
            z[t, i] = model.addVar(lb=0, vtype=GRB.INTEGER, name=f"z_{t}_{i}")
    model.update()
    for i in batch_levels:
        for t in range(n_periods):
            model.addConstr(x[t, i] == batch_sizes[i] * z[t, i],
                            name=f"batch_link_{t}_item_{i}")
    
    # Inventory balance constraints.
    for i in range(n_items):
        for t in range(n_periods):
            if t == 0:
                if i == n_items - 1:
                    model.addConstr(x[0, i] - I[0, i] == demand[0, i],
                                    name=f"balance_0_item_{i}")
                else:
                    model.addConstr(x[0, i] - I[0, i] == bom_ratio[i+1] * x[0, i+1],
                                    name=f"balance_0_item_{i}")
            else:
                if i == n_items - 1:
                    model.addConstr(I[t-1, i] + x[t, i] - I[t, i] == demand[t, i],
                                    name=f"balance_{t}_item_{i}")
                else:
                    model.addConstr(I[t-1, i] + x[t, i] - I[t, i] == bom_ratio[i+1] * x[t, i+1],
                                    name=f"balance_{t}_item_{i}")
    
    # (Allow final inventory to be positive; remove the zero–final-inventory constraint.)
    for i in range(n_items):
        model.addConstr(I[n_periods-1, i] >= 0, name=f"final_inventory_item_{i}")

    # Capacity and minimum order constraints.
    for i in range(n_items):
        for t in range(n_periods):
            model.addConstr(x[t, i] <= capacity[t, i] * y[t, i],
                            name=f"cap_upper_{t}_item_{i}")
            model.addConstr(x[t, i] >= min_order[t, i] * y[t, i],
                            name=f"min_order_{t}_item_{i}")
    
    # Objective: minimize total cost over all periods and items, including an extra penalty on final inventory.
    # The extra final inventory penalty is final_inv_factor * hold_cost[i] for inventory in period n_periods-1.
    actual_obj = (gp.quicksum(
            prod_cost[i] * x[t, i] + hold_cost[i] * I[t, i] + setup_cost[i] * y[t, i]
            for t in range(n_periods) for i in range(n_items)
        )
        + gp.quicksum(
            final_inv_factor * hold_cost[i] * I[n_periods-1, i] for i in range(n_items)
        )
    )

    # tie breaker
    binary_vars = [v for v in model.getVars()]
    random_cef = np.random.random(len(binary_vars))
    random_obj = sum(v * c for v, c in zip(binary_vars, random_cef))
    model.setObjective(
        actual_obj + random_obj,
        GRB.MINIMIZE,
    )
    
    # Store instance data.
    model._demand = demand
    model._capacity = capacity
    model._min_order = min_order
    model._bom_ratio = bom_ratio
    model._prod_cost = prod_cost
    model._hold_cost = hold_cost
    model._setup_cost = setup_cost
    model._n_periods = n_periods
    model._n_items = n_items
    model._scale = scale
    if batch_levels:
        model._batch_levels = batch_levels
        model._batch_sizes = batch_sizes
    
    model.update()
    return model
