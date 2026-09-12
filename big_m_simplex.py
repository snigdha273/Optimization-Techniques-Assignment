"""
Big-M Simplex Method
=====================
Problem :

    Minimize   Z = 4x1 + x2
    Subject to:
        3x1 +  x2  = 3
        4x1 + 3x2 >= 6
         x1 + 2x2 <= 4
        x1, x2 >= 0

Standard form conversion:
    Constraint 1 (=)  : needs an artificial variable A1
    Constraint 2 (>=) : subtract surplus s1, add artificial variable A2
    Constraint 3 (<=) : add slack s2

    3x1 +  x2 + A1                     = 3
    4x1 + 3x2      - s1      + A2      = 6
     x1 + 2x2            + s2          = 4

Big-M objective (minimization):
    Minimize Z = 4x1 + x2 + M*A1 + M*A2
    (M is a very large positive number)

This program builds the initial simplex tableau, then performs the
Big-M simplex iterations until optimality, printing every tableau.
"""

import numpy as np

np.set_printoptions(precision=3, suppress=True)

M = 1000  # a sufficiently large number to represent Big-M

# Variable order: x1, x2, s1(surplus), A1, A2, s2(slack)
var_names = ["x1", "x2", "s1", "A1", "A2", "s2"]

# Objective coefficients (minimization): x1, x2, s1, A1, A2, s2
c = np.array([4, 1, 0, M, M, 0], dtype=float)

# Constraint matrix (rows = constraints)
A = np.array([
    [3, 1, 0, 1, 0, 0],   # 3x1 + x2 + A1 = 3
    [4, 3, -1, 0, 1, 0],  # 4x1 + 3x2 - s1 + A2 = 6
    [1, 2, 0, 0, 0, 1],   # x1 + 2x2 + s2 = 4
], dtype=float)

b = np.array([3, 6, 4], dtype=float)

# Basic variables initially: A1, A2, s2  -> columns index 3, 4, 5
basis = [3, 4, 5]

n_vars = len(var_names)
n_cons = A.shape[0]


def print_tableau(A, b, c, basis, iteration):
    print(f"\n--- Iteration {iteration} ---")
    header = "Basis\t" + "\t".join(var_names) + "\tRHS"
    print(header)
    for i in range(n_cons):
        row = f"{var_names[basis[i]]}\t" + "\t".join(f"{v:6.2f}" for v in A[i]) + f"\t{b[i]:6.2f}"
        print(row)
    # Zj row and Cj - Zj row
    Zj = np.array([sum(c[basis[i]] * A[i][j] for i in range(n_cons)) for j in range(n_vars)])
    Cj_Zj = c - Zj
    Zval = sum(c[basis[i]] * b[i] for i in range(n_cons))
    print("Zj\t" + "\t".join(f"{v:6.2f}" for v in Zj) + f"\t{Zval:6.2f}")
    print("Cj-Zj\t" + "\t".join(f"{v:6.2f}" for v in Cj_Zj))
    return Cj_Zj


def big_m_simplex(A, b, c, basis):
    iteration = 0
    A = A.copy()
    b = b.copy()
    basis = basis.copy()

    while True:
        Cj_Zj = print_tableau(A, b, c, basis, iteration)

        # Optimality check (minimization): stop when all Cj-Zj >= 0
        if np.all(Cj_Zj >= -1e-9):
            print("\nOptimality reached: all (Cj - Zj) >= 0")
            break

        # Entering variable: most negative Cj - Zj
        entering = np.argmin(Cj_Zj)

        # Ratio test for leaving variable
        ratios = []
        for i in range(n_cons):
            if A[i][entering] > 1e-9:
                ratios.append(b[i] / A[i][entering])
            else:
                ratios.append(np.inf)
        ratios = np.array(ratios)

        if np.all(np.isinf(ratios)):
            print("Problem is unbounded.")
            return None, None, None

        leaving_row = np.argmin(ratios)
        pivot = A[leaving_row][entering]

        print(f"Entering variable: {var_names[entering]}, "
              f"Leaving variable: {var_names[basis[leaving_row]]}, "
              f"Pivot element: {pivot:.2f}")

        # Pivot operation
        A[leaving_row] = A[leaving_row] / pivot
        b[leaving_row] = b[leaving_row] / pivot
        for i in range(n_cons):
            if i != leaving_row:
                factor = A[i][entering]
                A[i] = A[i] - factor * A[leaving_row]
                b[i] = b[i] - factor * b[leaving_row]

        basis[leaving_row] = entering
        iteration += 1

        if iteration > 20:
            print("Stopping: too many iterations.")
            break

    return A, b, basis


if __name__ == "__main__":
    A_final, b_final, basis_final = big_m_simplex(A, b, c, basis)

    if A_final is not None:
        print("\n================ OPTIMAL SOLUTION ================")
        solution = {name: 0.0 for name in var_names}
        for i, bi in enumerate(basis_final):
            solution[var_names[bi]] = b_final[i]

        for name in var_names:
            print(f"{name} = {solution[name]:.3f}")

        # Check no artificial variable remains positive in basis (feasibility)
        if solution["A1"] > 1e-6 or solution["A2"] > 1e-6:
            print("\nWarning: Artificial variable is positive -> No feasible solution exists.")
        else:
            Z_optimal = 4 * solution["x1"] + 1 * solution["x2"]
            print(f"\nOptimal Objective Value: Z = 4*x1 + x2 = {Z_optimal:.3f}")
