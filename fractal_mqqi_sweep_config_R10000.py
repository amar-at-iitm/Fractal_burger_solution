"""WandB Hyperparameter Sweep Configuration for Chen & Wu (2006) MQQI: Reynolds Number R = 10,000 (Table 3).

This module defines the full admissible parameter search space for scaling coefficients
(f_alpha1, f_alpha2, f_alpha3) for Burgers' equation in the extreme high Reynolds number regime (R = 10,000):
    - Table 3 Benchmark: Chen & Wu (2006), quoting Christie & Mitchell (1978) and Hon & Mao (1998)
    - Domain: [0, 1], m = 72 subdivisions (h = 1/72 ≈ 0.01388889)
    - Time-stepping: Explicit Forward Euler, tau = 0.001 (10^-3), final time T = 1.0
    - Multiquadric shape parameter: c = lambda = 1.43e-4 (Table 3 in Chen & Wu 2006)
    - Initial condition: u(x, 0) = sin(pi * x)
    - Boundary conditions: u(0, t) = u(1, t) = 0

Mathematical & Physical Principles for R = 10,000:
--------------------------------------------------
1. Banach C^2 Smoothness Ceiling (Universal):
   - Difference domain [-1, 1] with N_sub = 6 subintervals (a_i = 1/6).
   - Read-Bajraktarevic operator on C^2: tilde{alpha}_i^(2) = alpha_i / a_i^2 = 36 * alpha_i.
   - Strict contractivity: max_i |alpha_i| < 1/36 ≈ 0.027778.
   - All parameter bounds are strictly within (-0.02778, +0.02778).

2. Forward Euler Diffusive Stability Ceiling:
   - Stability condition: (tau / R) * |lambda_max(M_xx)| <= 2.0  ==>  |lambda_max(M_xx)| <= 2 * R / tau = 20,000,000.0.
   - Baseline unperturbed MQQI (alpha = 0) has |lambda_max| ≈ 1,006,511.8.
   - The stability ratio is (tau / R) * rho ≈ 0.10065 (only 5.0% of the 2.0 ceiling; margin of 19.87x).
   - Diffusive instability is virtually non-existent; physical convection overwhelmingly dominates.
   - Core peak parameter f_alpha3 is unconstrained by diffusion up to the C^2 limit (+0.0250).

3. Convective Boundary Stabilization & Shock Resolution:
   - At R = 10,000, physical viscosity is nu = 1/R = 10^-4.
   - A sharp internal layer (shock wave) forms and propagates downstream toward the boundary x = 1.
   - Standard central differences and classical RBF interpolations produce aggressive non-physical Gibbs ringing.
   - f_alpha1 (Boundary tails, |z| in [2/3, 1]): Negative contraction [-0.025, -0.005] provides critical localized tail flattening,
     acting as an adaptive numerical boundary damper that quenches downstream oscillations without global artificial viscosity.
   - f_alpha2 (Intermediate transition, |z| in [1/3, 2/3]): Smooth transition layer in [-0.010, +0.015].
   - f_alpha3 (Core peak, |z| in [0, 1/3]): Positive values [+0.010, +0.025] sharpen the resolution of the moving shock front.
"""

# =============================================================================
# Optimal / Benchmark Configuration for R = 10,000 (Table 3)
# =============================================================================
sweep_config_best = {
    "method": "grid",
    "metric": {"name": "Linf_error", "goal": "minimize"},
    "parameters": {
        "f_alpha1": {"values": [-0.01500]},
        "f_alpha2": {"values": [0.00400]},
        "f_alpha3": {"values": [0.01800]},
    },
}

# =============================================================================
# Full Discrete Grid Sweep Configuration for R = 10,000 (23 points per parameter)
# =============================================================================
sweep_config = {
    "method": "grid",
    "metric": {"name": "Linf_error", "goal": "minimize"},
    "parameters": {
        # f_alpha1: Boundary subintervals (|z| in [2/3, 1])
        # Crucial for damping Gibbs ringing at downstream boundary x = 1
        "f_alpha1": {
            "values": [
                -0.02500,
                -0.02200,
                -0.02000,
                -0.01800,
                -0.01600,
                -0.01400,
                -0.01200,
                -0.01000,
                -0.00800,
                -0.00600,
                -0.00400,
                -0.00200,
                -0.00100,
                0.0,
                0.00100,
                0.00200,
                0.00300,
                0.00400,
                0.00500,
                0.00600,
                0.00700,
                0.00850,
                0.01000,
            ]
        },
        # f_alpha2: Intermediate transition subintervals (|z| in [1/3, 2/3])
        "f_alpha2": {
            "values": [
                -0.01000,
                -0.00600,
                -0.00300,
                -0.00100,
                0.0,
                0.00100,
                0.00200,
                0.00300,
                0.00400,
                0.00500,
                0.00600,
                0.00700,
                0.00800,
                0.00900,
                0.01000,
                0.01100,
                0.01200,
                0.01250,
                0.01300,
                0.01350,
                0.01400,
                0.01450,
                0.01500,
            ]
        },
        # f_alpha3: Core interior subintervals centered at z = 0 (|z| in [0, 1/3])
        # Sharpens shock front curvature resolution
        "f_alpha3": {
            "values": [
                -0.00200,
                0.0,
                0.00200,
                0.00400,
                0.00600,
                0.00800,
                0.01000,
                0.01200,
                0.01300,
                0.01400,
                0.01500,
                0.01600,
                0.01700,
                0.01800,
                0.01900,
                0.02000,
                0.02100,
                0.02200,
                0.02300,
                0.02350,
                0.02400,
                0.02450,
                0.02500,
            ]
        },
    },
}

# =============================================================================
# Continuous Bayesian Optimization Configuration for R = 10,000
# =============================================================================
sweep_config_bayes = {
    "method": "bayes",
    "metric": {"name": "Linf_error", "goal": "minimize"},
    "parameters": {
        "f_alpha1": {"distribution": "uniform", "min": -0.02500, "max": 0.01000},
        "f_alpha2": {"distribution": "uniform", "min": -0.01000, "max": 0.01500},
        "f_alpha3": {"distribution": "uniform", "min": -0.00200, "max": 0.02500},
    },
}
