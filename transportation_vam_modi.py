"""
Transportation Problem: Vogel's Approximation Method (VAM) + MODI Method
=========================================================================

Case study :

              D1    D2    D3    D4   | Supply
        O1     4     6     8     6   |   50
        O2     3     5     2     5   |   60
        O3     3     9     6     5   |   25
      -------------------------------
Demand        30    40    50    15   |  135 (= total supply)

Step 1: Use VAM to obtain an initial basic feasible solution (IBFS).
Step 2: Use MODI (u-v method) to test the IBFS for optimality and,
        if needed, iteratively improve the allocation until the
        optimal (minimum-cost) solution is reached.
"""

import numpy as np
import copy

np.set_printoptions(suppress=True)

sources = ["O1", "O2", "O3"]
destinations = ["D1", "D2", "D3", "D4"]

cost = np.array([
    [4, 6, 8, 6],
    [3, 5, 2, 5],
    [3, 9, 6, 5],
], dtype=float)

supply = [50, 60, 25]
demand = [30, 40, 50, 15]

assert sum(supply) == sum(demand), "Problem must be balanced."


# ---------------------------------------------------------------------
# STEP 1: Vogel's Approximation Method (VAM)
# ---------------------------------------------------------------------
def vam(cost, supply, demand):
    cost = cost.copy()
    supply = supply.copy()
    demand = demand.copy()
    m, n = cost.shape
    allocation = np.zeros((m, n))
    row_done = [False] * m
    col_done = [False] * n
    BIG = 1e9

    step = 1
    while not all(row_done) and not all(col_done):
        # Compute row penalties
        row_pen = []
        for i in range(m):
            if row_done[i]:
                row_pen.append(-1)
                continue
            vals = sorted(cost[i][j] for j in range(n) if not col_done[j])
            if len(vals) >= 2:
                row_pen.append(vals[1] - vals[0])
            elif len(vals) == 1:
                row_pen.append(vals[0])
            else:
                row_pen.append(-1)

        # Compute column penalties
        col_pen = []
        for j in range(n):
            if col_done[j]:
                col_pen.append(-1)
                continue
            vals = sorted(cost[i][j] for i in range(m) if not row_done[i])
            if len(vals) >= 2:
                col_pen.append(vals[1] - vals[0])
            elif len(vals) == 1:
                col_pen.append(vals[0])
            else:
                col_pen.append(-1)

        max_row = max(row_pen)
        max_col = max(col_pen)

        row_pen_disp = [round(float(x), 2) for x in row_pen]
        col_pen_disp = [round(float(x), 2) for x in col_pen]
        print(f"\nStep {step}: Row penalties = {row_pen_disp}, Col penalties = {col_pen_disp}")

        if max_row >= max_col:
            i = row_pen.index(max_row)
            j = min((j for j in range(n) if not col_done[j]), key=lambda j: cost[i][j])
        else:
            j = col_pen.index(max_col)
            i = min((i for i in range(m) if not row_done[i]), key=lambda i: cost[i][j])

        qty = min(supply[i], demand[j])
        allocation[i][j] = qty
        print(f"  Allocate {qty} units to cell ({sources[i]}, {destinations[j]}) "
              f"[cost = {cost[i][j]}]")

        supply[i] -= qty
        demand[j] -= qty

        if supply[i] == 0:
            row_done[i] = True
        if demand[j] == 0:
            col_done[j] = True

        step += 1

    total_cost = float((allocation * (cost)).sum())
    return allocation, total_cost


print("=" * 70)
print("STEP 1: VOGEL'S APPROXIMATION METHOD (VAM) - Initial Basic Feasible Solution")
print("=" * 70)
allocation, ibfs_cost = vam(cost, supply, demand)

print("\nInitial Allocation (VAM):")
print("\t" + "\t".join(destinations))
for i, row in enumerate(allocation):
    print(sources[i] + "\t" + "\t".join(f"{v:.0f}" for v in row))
print(f"\nTotal Transportation Cost (IBFS from VAM) = {ibfs_cost:.2f}")


# ---------------------------------------------------------------------
# STEP 2: MODI (Modified Distribution) Method - test/optimize
# ---------------------------------------------------------------------
def modi_method(cost, allocation):
    m, n = cost.shape
    allocation = allocation.copy()
    iteration = 1

    while True:
        print(f"\n--- MODI Iteration {iteration} ---")
        basic_cells = [(i, j) for i in range(m) for j in range(n) if allocation[i][j] > 0]

        # Handle degeneracy: need m+n-1 basic cells
        needed = m + n - 1
        if len(basic_cells) < needed:
            # add an epsilon allocation to the lowest-cost independent empty cell
            for i in range(m):
                for j in range(n):
                    if allocation[i][j] == 0 and (i, j) not in basic_cells:
                        allocation[i][j] = 1e-6
                        basic_cells.append((i, j))
                        print(f"  Degeneracy handled: added epsilon at ({sources[i]},{destinations[j]})")
                        break
                if len(basic_cells) >= needed:
                    break

        # Solve for u_i, v_j using u_i + v_j = c_ij for basic cells
        u = [None] * m
        v = [None] * n
        u[0] = 0
        changed = True
        while changed:
            changed = False
            for (i, j) in basic_cells:
                if u[i] is not None and v[j] is None:
                    v[j] = cost[i][j] - u[i]
                    changed = True
                elif v[j] is not None and u[i] is None:
                    u[i] = cost[i][j] - v[j]
                    changed = True

        print(f"  u = {['%.2f'%x if x is not None else None for x in u]}")
        print(f"  v = {['%.2f'%x if x is not None else None for x in v]}")

        # Compute opportunity cost (Cij - ui - vj) for non-basic cells
        delta = np.full((m, n), np.nan)
        for i in range(m):
            for j in range(n):
                if (i, j) not in basic_cells:
                    delta[i][j] = cost[i][j] - u[i] - v[j]

        print("  Opportunity costs (delta) for non-basic cells:")
        for i in range(m):
            row_str = []
            for j in range(n):
                if np.isnan(delta[i][j]):
                    row_str.append("  -  ")
                else:
                    row_str.append(f"{delta[i][j]:5.2f}")
            print("   " + "\t".join(row_str))

        # Optimality check: all deltas >= 0
        min_delta = np.nanmin(delta)
        if min_delta >= -1e-6 or np.isnan(min_delta):
            print("\n  All opportunity costs >= 0 -> OPTIMAL SOLUTION REACHED.")
            break

        # Entering cell = most negative delta
        enter_i, enter_j = np.unravel_index(np.nanargmin(delta), delta.shape)
        print(f"  Entering cell: ({sources[enter_i]}, {destinations[enter_j]}) "
              f"with delta = {delta[enter_i][enter_j]:.2f}")

        # Build closed loop (stepping-stone) starting at entering cell
        loop = find_closed_loop(allocation, enter_i, enter_j)
        print(f"  Closed loop: {[(sources[i], destinations[j]) for i, j in loop]}")

        # Alternate +/- around loop; theta = min allocation at '-' positions
        minus_cells = loop[1::2]
        theta = min(allocation[i][j] for i, j in minus_cells)
        print(f"  Theta (units to reallocate) = {theta:.2f}")

        for idx, (i, j) in enumerate(loop):
            if idx % 2 == 0:
                allocation[i][j] += theta
            else:
                allocation[i][j] -= theta

        iteration += 1
        if iteration > 25:
            print("Stopping: too many MODI iterations.")
            break

    return allocation


def find_closed_loop(allocation, start_i, start_j):
    """Find a closed loop for the stepping-stone/MODI method starting
    at (start_i, start_j), alternating horizontal/vertical moves through
    currently basic (occupied) cells."""
    m, n = allocation.shape
    occupied = [(i, j) for i in range(m) for j in range(n) if allocation[i][j] > 0]
    occupied.append((start_i, start_j))

    def get_loop(path, turn):
        last = path[-1]
        if len(path) > 3 and last == path[0]:
            return path
        if turn == 'row':
            candidates = [c for c in occupied if c[0] == last[0] and c != last]
        else:
            candidates = [c for c in occupied if c[1] == last[1] and c != last]
        for nxt in candidates:
            if nxt == path[0] and len(path) >= 3:
                return path + [nxt]
            if nxt in path:
                continue
            result = get_loop(path + [nxt], 'col' if turn == 'row' else 'row')
            if result:
                return result
        return None

    loop = get_loop([(start_i, start_j)], 'row')
    if loop is None:
        loop = get_loop([(start_i, start_j)], 'col')
    return loop[:-1]  # drop the repeated starting cell


print("\n" + "=" * 70)
print("STEP 2: MODI (MODIFIED DISTRIBUTION) METHOD - Optimality Test")
print("=" * 70)
optimal_allocation = modi_method(cost, allocation)

# Clean tiny epsilon values
optimal_allocation[optimal_allocation < 1e-3] = 0

optimal_cost = float((optimal_allocation * cost).sum())

print("\nFinal Optimal Allocation:")
print("\t" + "\t".join(destinations))
for i, row in enumerate(optimal_allocation):
    print(sources[i] + "\t" + "\t".join(f"{v:.0f}" for v in row))

print(f"\n================ FINAL RESULT ================")
print(f"Minimum Total Transportation Cost = {optimal_cost:.2f}")
