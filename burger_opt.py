"""WandB Sweep Optimization for Fractal L_W2 Operator on Burgers' Equation (Classical Initial Condition).

This script optimizes the fractal scaling parameters f_alpha for the second-derivative
multiquadric quasi-interpolation operator L_W2^alpha using Weights & Biases sweeps.

Problem: 1D Viscous Burgers' Equation
    u_t + u * u_x = (1 / R) * u_xx,   x in [0, 1], t > 0
    u(x, 0) = sin(pi * x)            (Classical initial condition)
    u(0, t) = u(1, t) = 0            (Homogeneous Dirichlet boundaries)

Exact Solution: Cole-Hopf analytical solution (Cole, 1951).
"""

from __future__ import annotations

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import wandb

from burger_solver import (
    build_ux_matrix,
    build_classical_spatial_operator,
    step_classical_burgers,
)
from fractal_burger_solver import (
    build_fractal_second_derivative,
    build_fractal_spatial_operator,
    step_fractal_burgers,
)
from mqqi.problems import cole_analytic_solution

# Import sweep configuration
from fractal_sweep_config import sweep_config

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
# Benchmark Problem Parameters: Chen & Wu (2006) Case 1
# ============================================================================
a = 0.0
b = 1.0
R = 10.0          # Reynolds number
m = 100           # Number of spatial subdivisions
h = (b - a) / m   # Mesh size h = 0.01
K = 10            # Center subsampling stride N = m / K = 10
s = 0.8           # RBF shape parameter
c = 0.0072        # Multiquadric shape parameter
tau = 0.001       # Time step size
T = 1.0           # Final simulation time

n = round((b - a) / h)
if not np.isclose(a + n * h, b):
    raise ValueError("h must divide b-a exactly.")
n = int(n)

if n % K != 0:
    raise ValueError("n must be divisible by K because N = n / K.")

N = n // K
Nt = int(round(T / tau))

x = np.linspace(a, b, n + 1)

# Interpolation center indices k_j, j = 1, ..., N
k_idx = np.concatenate(([1], K * np.arange(1, N - 1), [n - 1])).astype(int)
xk = x[k_idx]

# Classical initial condition at t=0
def initial_condition(x_coords: np.ndarray) -> np.ndarray:
    u = np.sin(np.pi * x_coords)
    u[0] = 0.0
    u[-1] = 0.0
    return u

u0 = initial_condition(x)

# Precompute first-derivative MQQI operator matrix
A_ux = build_ux_matrix(x, c)

# Compute Cole's exact analytical solution at t = T
print(f"Computing Cole's exact analytical solution at T={T}, R={R}...")
u_exact = cole_analytic_solution(x, T, R)

# ============================================================================
# Classical Baseline Evaluation
# ============================================================================
print("Computing Classical L_W2 baseline...")
M_xx_classical = build_classical_spatial_operator(x, xk, k_idx, s, c)
u_classical = u0.copy()
for _ in range(Nt):
    u_classical = step_classical_burgers(u_classical, tau, R, A_ux, M_xx_classical)

err_classical = np.abs(u_classical - u_exact)
classical_linf = float(np.max(err_classical))
classical_rms = float(np.sqrt(np.mean(err_classical**2)))

print(f"Classical Baseline: Linf = {classical_linf:.6e}, RMS = {classical_rms:.6e}")
print("=" * 78)

best_Linf_error = float("inf")
best_RMS_error = float("inf")


# ============================================================================
# Fractal Optimization Objective for WandB
# ============================================================================
def fractal_optimization() -> None:
    global best_Linf_error, best_RMS_error

    wandb.init(settings=wandb.Settings(init_timeout=3000))
    config = wandb.config

    f_alpha1 = getattr(config, "f_alpha1", 0.0)
    f_alpha2 = getattr(config, "f_alpha2", 0.0)
    f_alpha3 = getattr(config, "f_alpha3", 0.0)

    run_name = f"alpha1-{f_alpha1}_alpha2-{f_alpha2}_alpha3-{f_alpha3}"
    wandb.run.name = run_name

    # Symmetric 6-subinterval scaling vector on [-1, 1]:
    # Exterior subintervals remain unperturbed (0.0), interior subintervals
    # regularize regions with steep convective shock gradients.
    f_alpha = [f_alpha1, f_alpha2, f_alpha3, f_alpha3, f_alpha2, f_alpha1]

    n_subintervals = len(f_alpha)
    n_iter = 8


    # Precompute fractal second derivative of multiquadric RBF
    fractal_dd = build_fractal_second_derivative(c=c, f_alpha=f_alpha, n_iter=n_iter, a_domain=-1.0, b_domain=1.0, n_sub=n_subintervals)

    # Build condensed fractal spatial operator M_xx^fractal:
    # M_xx_fractal @ u evaluates d2_fractal_L_W2 at all grid points simultaneously
    # with exact mathematical equivalence and 500x speedup over pointwise nested loops.
    M_xx_fractal = build_fractal_spatial_operator(x, xk, k_idx, s, fractal_dd)

    # Initial condition at t=0
    f_U = u0.copy()

    # Time stepping to t = T
    for _ in range(Nt):
        f_U = step_fractal_burgers(f_U, tau, R, A_ux, M_xx_fractal)

    f_u_num = f_U
    f_err = f_u_num - u_exact
    f_abs_err = np.abs(f_err)

    Linf_error = float(np.max(f_abs_err))
    RMS_error = float(np.sqrt(np.mean(f_abs_err**2)))

    linf_reduction_pct = 100.0 * (classical_linf - Linf_error) / classical_linf
    rms_reduction_pct = 100.0 * (classical_rms - RMS_error) / classical_rms
    is_superior = bool(Linf_error < classical_linf)

    if Linf_error < best_Linf_error:
        best_Linf_error = Linf_error
        best_RMS_error = RMS_error
        print(f"--> New best Linf error: {best_Linf_error:.6e}")
        print(f"--> New best RMS error:  {best_RMS_error:.6e}")
        print(f"--> Accuracy improvement: Linf {linf_reduction_pct:+.2f}%, RMS {rms_reduction_pct:+.2f}%")

        file_name = f"best_results_burger_T_{T}.txt"
        with open(file_name, "w") as f_out:
            f_out.write(f"Burgers Equation (Classical Initial Condition) at time T={T}:\n")
            f_out.write(f"Best Fractal Linf error: {best_Linf_error:.6e}\n")
            f_out.write(f"Best Fractal RMS error:  {best_RMS_error:.6e}\n")
            f_out.write(f"Classical Baseline Linf: {classical_linf:.6e}\n")
            f_out.write(f"Classical Baseline RMS:  {classical_rms:.6e}\n")
            f_out.write(f"Linf Error Reduction:    {linf_reduction_pct:+.2f}%\n")
            f_out.write(f"RMS Error Reduction:     {rms_reduction_pct:+.2f}%\n")
            f_out.write(f"Optimal Parameters: alpha1={f_alpha1}, alpha2={f_alpha2}, alpha3={f_alpha3}\n")
            f_out.write(f"Full f_alpha vector: {f_alpha}\n")

    wandb.log({
        "alpha1": f_alpha1,
        "alpha2": f_alpha2,
        "alpha3": f_alpha3,
        "RMS_error": RMS_error,
        "Linf_error": Linf_error,
        "classical_Linf": classical_linf,
        "classical_RMS": classical_rms,
        "linf_reduction_pct": linf_reduction_pct,
        "rms_reduction_pct": rms_reduction_pct,
        "is_superior": is_superior,
    })

    wandb.finish()


# ============================================================================
# Run WandB Sweep
# ============================================================================
if __name__ == "__main__":
    project_name = os.environ.get("WANDB_PROJECT", "Burger_fractal_optimization")
    sweep_id = wandb.sweep(sweep_config, project=project_name)
    wandb.agent(sweep_id, function=fractal_optimization)
    print("Sweep complete.")

