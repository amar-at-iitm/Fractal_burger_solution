from __future__ import annotations

import os
import numpy as np
import wandb

from fractal_burger_solver import (
    build_fractal_second_derivative,
    pointwise_fractal,
)
from mqqi.problems import cole_analytic_solution
from fractal_chen_wu_sweep_config import sweep_config

# Authenticate with Weights & Biases (uses WANDB_API_KEY environment variable or cached login)
wandb_key = os.environ.get("WANDB_API_KEY")
if wandb_key:
    wandb.login(key=wandb_key)
else:
    try:
        wandb.login()
    except Exception as exc:
        print(f"wandb login notice: {exc}")

# ============================================================================
# Parameters: Chen & Wu (2006) Example 1 (burger_equation.tex)
# ============================================================================
a = 0.0
b = 1.0
R = 10.0          # Reynolds number (Eq. 3.1)
m = 100           # Number of spatial subdivisions (x_0, x_1, ..., x_m)
h = (b - a) / m   # Mesh size h = 0.01 (h_i = h = 1/m, Section 4)
c = 0.0072        # Multiquadric shape parameter lambda (Eq. 2.6)
tau = 0.001       # Time step size (Section 4)
T = 1.0           # Final simulation time

n = int(round((b - a) / h))
if not np.isclose(a + n * h, b):
    raise ValueError("h must divide b-a exactly.")
Nt = int(round(T / tau))

# Spatial grid nodes: x_j = a + j * h,  j = 0, 1, ..., m
x = np.linspace(a, b, n + 1)


# Initial condition (Eq. 4.1 in burger_equation.tex): u(x, 0) = sin(pi * x)
def initial_condition(x_coords: np.ndarray) -> np.ndarray:
    u = np.sin(np.pi * x_coords)
    u[0] = 0.0
    u[-1] = 0.0
    return u


u0 = initial_condition(x)

# Exact Cole-Hopf analytical benchmark solution at t = T
u_exact = cole_analytic_solution(x, T, R)


# ============================================================================
# Step 1: Precompute Multiquadric Basis Function Derivative Tables
# (Equations 2.6 and 2.4 in burger_equation.tex)
# ============================================================================
def build_basis_first_derivatives(x_nodes: np.ndarray, shape_c: float = c) -> np.ndarray:
    # from eqaution 2.4 and 2.6
    m_len = len(x_nodes) - 1
    dphi = np.zeros((m_len + 1, m_len + 1), dtype=float)

    for j in range(m_len + 1):
        xj = x_nodes[j]
        # Interior and left boundary centers: k = 0, ..., m - 1
        for k in range(m_len):
            xk = x_nodes[k]
            diff_x = xj - xk
            dphi[j, k] = diff_x / np.sqrt(diff_x**2 + shape_c**2)

        # Right boundary center k = m using Eq. (2.4): phi_m'(x_j) = phi_0'(x_j) - 2
        dphi[j, m_len] = dphi[j, 0] - 2.0

    return dphi


def build_basis_second_derivatives(
    x_nodes: np.ndarray,
    shape_c: float = c,
    fractal_dd: dict | None = None,
) -> np.ndarray:
    # from equation 2.4 and 2.6
    m_len = len(x_nodes) - 1
    d2phi = np.zeros((m_len + 1, m_len + 1), dtype=float)

    if fractal_dd is None:
        # Classical multiquadric second derivative
        for j in range(m_len + 1):
            xj = x_nodes[j]
            for k in range(m_len):
                xk = x_nodes[k]
                diff_x = xj - xk
                d2phi[j, k] = (shape_c**2) / (np.sqrt(diff_x**2 + shape_c**2) ** 3)
            # Boundary closure Eq. (2.4): phi_m''(x_j) = phi_0''(x_j)
            d2phi[j, m_len] = d2phi[j, 0]
    else:
        # Fractalized multiquadric second derivative (pointwise IFS evaluation)
        for j in range(m_len + 1):
            xj = x_nodes[j]
            for k in range(m_len):
                xk = x_nodes[k]
                diff_x = xj - xk
                d2phi[j, k] = pointwise_fractal(diff_x, fractal_dd)
            # Boundary closure Eq. (2.4): phi_m''(x_j) = phi_0''(x_j)
            d2phi[j, m_len] = d2phi[j, 0]

    return d2phi


# Precompute classical first-derivative basis table phi_k'(x_j)
dphi_table = build_basis_first_derivatives(x, c)


# ============================================================================
# Step 2: Direct Mathematical Time-Stepping Solver (Without A_mat and B_mat)
# ============================================================================
def solve_chen_wu_direct(
    u_init: np.ndarray,
    dphi: np.ndarray,
    d2phi: np.ndarray,
    h_step: float,
    time_step: float,
    reynolds: float,
    num_steps: int,
) -> np.ndarray:

    m_len = len(u_init) - 1
    u = u_init.copy()

    ux = np.zeros(m_len + 1, dtype=float)
    uxx = np.zeros(m_len + 1, dtype=float)
    g = np.ones(m_len + 1, dtype=float)

    for _ in range(num_steps):
        # --------------------------------------------------------------------
        # Part A: First divided differences of current solution u^n
        # delta_u_k = u_{k+1}^n - u_k^n,  k = 0, ..., m - 1
        # --------------------------------------------------------------------
        delta_u = u[1:] - u[:-1]

        # --------------------------------------------------------------------
        # Part B: Direct calculation of (u_x)_j and (u_xx)_j for each node x_j
        #
        # Eq. (3.5):
        # (u_x)_j = sum_{k=0}^{m-1} [ (phi_k'(x_j) - phi_{k+1}'(x_j)) / (2*h) ] * (u_{k+1} - u_k)
        #
        # Eq. (3.6):
        # (u_xx)_j = sum_{k=0}^{m-1} [ (phi_k''(x_j) - phi_{k+1}''(x_j)) / (2*h) ] * (u_{k+1} - u_k)
        # --------------------------------------------------------------------
        for j in range(m_len + 1):
            # Kernel difference vectors across centers k = 0, ..., m - 1:
            diff_dphi_j = dphi[j, :m_len] - dphi[j, 1 : m_len + 1]
            diff_d2phi_j = d2phi[j, :m_len] - d2phi[j, 1 : m_len + 1]

            # Direct summation over k = 0 to m - 1:
            ux[j] = np.sum(0.5 * (diff_dphi_j / h_step) * delta_u)
            uxx[j] = np.sum(0.5 * (diff_d2phi_j / h_step) * delta_u)

        # --------------------------------------------------------------------
        # Part C: Upwind dispersion switch function g_j (Equation 3.3)
        #
        # g_j^n = max{0, 1 + min{0, sign((u_x)_j^n * (u_x)_k^n)}}
        # where k = j - sign(u_j^n)
        # --------------------------------------------------------------------
        for j in range(1, m_len):
            sgn_u = np.sign(u[j])
            k_switch = int(j - sgn_u)
            prod_ux = ux[j] * ux[k_switch]
            sgn_prod = np.sign(prod_ux)
            g[j] = max(0.0, 1.0 + min(0.0, sgn_prod))

        # --------------------------------------------------------------------
        # Part D: Explicit Forward Euler time-stepping update (Equation 3.4)
        #
        # u_j^{n+1} = u_j^n - tau * u_j^n * (u_x)_j^n * g_j^n + (tau / R) * (u_xx)_j^n
        # --------------------------------------------------------------------
        for j in range(1, m_len):
            convective_term = u[j] * ux[j] * g[j]
            diffusive_term = (1.0 / reynolds) * uxx[j]
            u[j] = u[j] - time_step * convective_term + time_step * diffusive_term

        # --------------------------------------------------------------------
        # Part E: Homogeneous Dirichlet boundary conditions (Equation 4.2)
        # u(0, t) = 0,  u(1, t) = 0
        # --------------------------------------------------------------------
        u[0] = 0.0
        u[m_len] = 0.0

        # Numerical stability check for CFL blowup (NaN / Inf)
        if np.isnan(u[1:m_len]).any() or np.isinf(u[1:m_len]).any():
            break

    return u


best_Linf_error = float("inf")
best_RMS_error = float("inf")


# ============================================================================
# Step 3: Fractal Optimization Objective for WandB Sweeps
# ============================================================================
def fractal_optimization():
    global best_Linf_error, best_RMS_error

    wandb.init(settings=wandb.Settings(init_timeout=3000))
    config = wandb.config
    f_alpha1 = config.f_alpha1
    f_alpha2 = config.f_alpha2
    f_alpha3 = config.f_alpha3

    run_name = f"direct_alpha1-{f_alpha1}_alpha2-{f_alpha2}_alpha3-{f_alpha3}"
    wandb.run.name = run_name

    # Symmetric 6-subinterval scaling vector on difference domain [-1, 1]
    f_alpha = [f_alpha1, f_alpha2, f_alpha3, f_alpha3, f_alpha2, f_alpha1]
    n_subintervals = len(f_alpha)
    n_iter = 8

    # Precompute fractal multiquadric second derivative table
    fractal_dd = build_fractal_second_derivative(
        c=c,
        f_alpha=f_alpha,
        n_iter=n_iter,
        a_domain=-1.0,
        b_domain=1.0,
        n_sub=n_subintervals,
    )

    # Build direct basis second derivative table d2phi[j, k] = (phi^alpha)''_k(x_j)
    d2phi_table = build_basis_second_derivatives(x, shape_c=c, fractal_dd=fractal_dd)

    # Solve Burgers' equation directly step-by-step
    f_u_num = solve_chen_wu_direct(
        u_init=u0,
        dphi=dphi_table,
        d2phi=d2phi_table,
        h_step=h,
        time_step=tau,
        reynolds=R,
        num_steps=Nt,
    )

    if np.isnan(f_u_num).any() or np.isinf(f_u_num).any():
        Linf_error = float("inf")
        RMS_error = float("inf")
    else:
        f_abs_err = np.abs(f_u_num - u_exact)
        Linf_error = float(np.max(f_abs_err))
        RMS_error = float(np.sqrt(np.mean(f_abs_err**2)))

    if Linf_error < best_Linf_error:
        best_Linf_error = Linf_error
        best_RMS_error = RMS_error
        print(f" New best Linf error: {best_Linf_error:.6e}")
        print(f" New best RMS error:  {best_RMS_error:.6e}")

        file_name = f"best_results_chen_wu_mqqi_direct_T_{T}.txt"
        with open(file_name, "w") as f:
            f.write(f"Classical MQQI at time T={T}:\n")
            f.write(f"Best Linf error: {best_Linf_error:.6e}\n")
            f.write(f"Best RMS error:  {best_RMS_error:.6e}\n")
            f.write(f"Parameters: alpha1={f_alpha1}, alpha2={f_alpha2}, alpha3={f_alpha3}\n")
            f.write(f"Full f_alpha vector: {f_alpha}\n")

    wandb.log({
        "alpha1": f_alpha1,
        "alpha2": f_alpha2,
        "alpha3": f_alpha3,
        "RMS_error": RMS_error,
        "Linf_error": Linf_error,
    })

    wandb.finish()


# ============================================================================
# Main Entry Point
# ============================================================================
if __name__ == "__main__":
    project_name = os.environ.get("WANDB_PROJECT", "Burger_Chen_Wu_fractal_optimization")
    sweep_id = wandb.sweep(sweep_config, project=project_name)
    wandb.agent(sweep_id, function=fractal_optimization)
    print("Direct loop sweep complete")
