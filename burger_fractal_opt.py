"""WandB Sweep Optimization for Dual-Fractal Burgers' Equation (Fractal Initial Condition + Fractal L_W2).

This script optimizes the fractal scaling parameters f_alpha for the second-derivative
multiquadric quasi-interpolation operator L_W2^alpha when the initial condition is ALSO
an alpha-fractal function (mirroring the setup of SG_fractal_opt.py).

Problem: 1D Viscous Burgers' Equation with Fractal Initial Condition
    u_t + u * u_x = (1 / R) * u_xx,   x in [0, 1], t > 0
    u(x, 0) = f^beta(x)              (Alpha-fractalized sine profile)
    u(0, t) = u(1, t) = 0            (Homogeneous Dirichlet boundaries)

Exact Solution: Exact Cole-Hopf analytical solution for fractal initial profile,
derived via Fourier-Bessel spectral expansion of the linearized heat equation
with high-resolution numerical quadrature.
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
    rbf_matrix,
    second_divided_difference,
    build_ux_matrix,
    compute_switch_function,
    build_classical_spatial_operator,
    step_classical_burgers,
)
from fractal_burger_solver import (
    d2_fractal_L_W2,
    ddphi,
    H5_dd,
    pointwise_fractal,
    build_fractal_second_derivative,
    build_fractal_spatial_operator,
    step_fractal_burgers,
)
from alpha_fractal_function import alpha_fractalize, alpha_fractalize_second_derivative

# Import sweep configuration
try:
    from fractal_sweep_config import sweep_config
except ImportError:
    sweep_config = {
        "method": "grid",
        "metric": {"name": "Linf_error", "goal": "minimize"},
        "parameters": {
            "f_alpha1": {"values": [0.0, -0.0001, -0.0002, -0.0003]},
            "f_alpha2": {"values": [0.0, -0.00005, -0.0001, -0.0002]},
            "f_alpha3": {"values": [0.0, -0.0001, -0.0002, -0.0003]},
        },
    }

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

# ============================================================================
# Fractal Initial Condition Construction
# ============================================================================
# Base smooth function f(x) = sin(pi * x)
def f(x_val):
    y = np.sin(np.pi * x_val)
    return np.where(np.isclose(y, 0.0, atol=1e-12), 0.0, y)

# Base perturbation function g(x) satisfying g(0) = f(0) = 0 and g(1) = f(1) = 0
def g(x_val):
    return x_val * (1.0 - x_val)

# Fractal scaling vector for initial condition (beta)
f_beta = [0.005, 0.0025, 0.0025, 0.005]
subintervals = len(f_beta)
n_stages = 4

print("Constructing alpha-fractal initial profile...")
sine_pi_fractal = alpha_fractalize(f, g, a, b, subintervals, f_beta, n_stages)

# Evaluate fractal initial condition on spatial grid
u0_fractal = np.array([pointwise_fractal(xi, sine_pi_fractal) for xi in x])
u0_fractal[0] = 0.0
u0_fractal[-1] = 0.0

# Precompute first-derivative MQQI operator matrix
A_ux = build_ux_matrix(x, c)

# ============================================================================
# Analytical Solution via Cole-Hopf Fourier-Quadrature for Fractal Profile
# ============================================================================
def general_cole_analytic(
    x_eval: np.ndarray,
    t_eval: float,
    Re: float,
    fractal_dict: dict,
    n_terms: int = 120,
    n_quad: int = 10000,
) -> np.ndarray:
    """Exact analytical Cole-Hopf solution for arbitrary initial condition u0."""
    eta = np.linspace(0.0, 1.0, n_quad)
    deta = 1.0 / (n_quad - 1)
    u0_vals = pointwise_fractal(eta, fractal_dict)
    u0_vals[0] = 0.0
    u0_vals[-1] = 0.0

    # Cumulative integral U0(eta) = int_0^eta u0(xi) dxi
    U0 = np.zeros(n_quad)
    U0[1:] = np.cumsum(0.5 * (u0_vals[:-1] + u0_vals[1:]) * deta)

    integrand_base = np.exp(-(Re / 2.0) * U0)
    weights = np.ones(n_quad)
    weights[0] = 0.5
    weights[-1] = 0.5

    a0 = 2.0 * np.sum(integrand_base * weights) * deta
    num = np.zeros_like(x_eval, dtype=float)
    den = np.full_like(x_eval, 0.5 * a0, dtype=float)

    for n_idx in range(1, n_terms + 1):
        cos_n_eta = np.cos(n_idx * np.pi * eta)
        an = 2.0 * np.sum(integrand_base * cos_n_eta * weights) * deta
        decay = np.exp(-(n_idx**2 * np.pi**2 * t_eval) / Re)
        if decay < 1e-18:
            break
        num += n_idx * an * decay * np.sin(n_idx * np.pi * x_eval)
        den += an * decay * np.cos(n_idx * np.pi * x_eval)

    return (2.0 * np.pi / Re) * (num / den)


print(f"Computing exact analytical Cole-Hopf solution for fractal u0 at T={T}...")
u_exact = general_cole_analytic(x, T, R, sine_pi_fractal)

# ============================================================================
# Classical Baseline Evaluation on Fractal Initial Profile
# ============================================================================
print("Computing Classical L_W2 baseline on fractal initial profile...")
M_xx_classical = build_classical_spatial_operator(x, xk, k_idx, s, c)
u_classical = u0_fractal.copy()
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

    # Symmetric 10-subinterval scaling vector on [-2, 2]:
    # Core contracts where convective gradients are steepest:
    f_alpha = [0.0, 0.0, f_alpha1, f_alpha2, f_alpha3, f_alpha3, f_alpha2, f_alpha1, 0.0, 0.0]

    n_subintervals = len(f_alpha)
    n_iter = 3

    # Precompute fractal second derivative of multiquadric RBF
    fractal_dd = build_fractal_second_derivative(c=c, f_alpha=f_alpha, n_iter=n_iter)

    # Build condensed fractal spatial operator M_xx^fractal:
    # M_xx_fractal @ u evaluates d2_fractal_L_W2 at all grid points simultaneously
    # with exact mathematical equivalence and 500x speedup over pointwise nested loops.
    M_xx_fractal = build_fractal_spatial_operator(x, xk, k_idx, s, fractal_dd)

    # Initial condition at t=0: fractal initial condition
    f_U = u0_fractal.copy()

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

        file_name = f"best_results_burger_fractal_T_{T}.txt"
        with open(file_name, "w") as f_out:
            f_out.write(f"Burgers Equation (Fractal Initial Condition) at time T={T}:\n")
            f_out.write(f"Best Fractal Linf error: {best_Linf_error:.6e}\n")
            f_out.write(f"Best Fractal RMS error:  {best_RMS_error:.6e}\n")
            f_out.write(f"Classical Baseline Linf: {classical_linf:.6e}\n")
            f_out.write(f"Classical Baseline RMS:  {classical_rms:.6e}\n")
            f_out.write(f"Linf Error Reduction:    {linf_reduction_pct:+.2f}%\n")
            f_out.write(f"RMS Error Reduction:     {rms_reduction_pct:+.2f}%\n")
            f_out.write(f"Optimal Parameters: alpha1={f_alpha1}, alpha2={f_alpha2}, alpha3={f_alpha3}\n")
            f_out.write(f"Full f_alpha vector: {f_alpha}\n")
            f_out.write(f"Initial condition beta: {f_beta}\n")

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
    project_name = os.environ.get("WANDB_PROJECT", "Burger_dual_fractal_optimization")
    sweep_id = wandb.sweep(sweep_config, project=project_name)
    wandb.agent(sweep_id, function=fractal_optimization)
    print("Sweep complete.")

