"""WandB Hyperparameter Sweep Configuration for Chen & Wu (2006) MQQI: Reynolds Number R = 100.

This module defines the full admissible parameter search space for scaling coefficients
(f_alpha1, f_alpha2, f_alpha3) for Burgers' equation at R = 100:
    - Domain: [0, 1], m = 100 subdivisions (h = 0.01)
    - Time-stepping: Explicit Forward Euler, tau = 0.001, final time T = 1.0
    - Multiquadric shape parameter: c = lambda = 0.0029 (Table 2 & Fig. 2 in Chen & Wu 2006)
    - Initial condition: u(x, 0) = sin(pi * x)
    - Boundary condition: u(0, t) = u(1, t) = 0

Mathematical & Physical Principles for R = 100:
-----------------------------------------------
1. Banach C^2 Smoothness Ceiling (Universal):
   - Difference domain [-1, 1] with N_sub = 6 subintervals (a_i = 1/6).
   - Read-Bajraktarevic operator on C^2: tilde{alpha}_i^(2) = alpha_i / a_i^2 = 36 * alpha_i.
   - Strict contractivity: max_i |alpha_i| < 1/36 ≈ 0.027778.
   - All parameter bounds are strictly within (-0.02778, +0.02778).

2. Forward Euler Diffusive Stability Ceiling:
   - Stability condition: (tau / R) * |lambda_max(M_xx)| <= 2.0  ==>  |lambda_max(M_xx)| <= 2 * R / tau = 200,000.0.
   - Baseline unperturbed MQQI (alpha = 0) has |lambda_max| ≈ 66,289.6.
   - The stability ratio is (tau / R) * rho ≈ 0.6629 (only 33.1% of the 2.0 limit).
   - Unlike R = 10 (where the baseline was at 91.8% of the ceiling), R = 100 provides ample stability margin,
     permitting central peak parameter f_alpha3 to expand all the way up to +0.0250 without inducing numerical blowup.

3. Physical Dynamics & Parameter Roles:
   - At R = 100, nonlinear convective steepening pushes the wave front toward x = 1, forming an asymmetric waveform.
   - f_alpha1 (Boundary tails, |z| in [2/3, 1]): Negative values [-0.020, -0.005] damp spurious boundary reflections
     at the steep outflow gradient near x = 1.
   - f_alpha2 (Intermediate transition, |z| in [1/3, 2/3]): Acts as a buffer connecting boundary damping to core curvature.
   - f_alpha3 (Core peak, |z| in [0, 1/3]): Positive values [+0.012, +0.022] sharpen local Multiquadric peak curvature,
     compensating for first-difference truncation deficit and reducing Linf error significantly.
"""

# =============================================================================
# Optimal / Benchmark Configuration for R = 100
# =============================================================================
sweep_config_best = {
    "method": "grid",
    "metric": {"name": "Linf_error", "goal": "minimize"},
    "parameters": {
        "f_alpha1": {"values": [-0.01000]},
        "f_alpha2": {"values": [0.00500]},
        "f_alpha3": {"values": [0.01800]},
    },
}

# =============================================================================
# Full Discrete Grid Sweep Configuration for R = 100 (23 points per parameter)
# =============================================================================
sweep_config = {
    "method": "grid",
    "metric": {"name": "Linf_error", "goal": "minimize"},
    "parameters": {
        # f_alpha1: Boundary subintervals (|z| in [2/3, 1])
        # Full admissible range from convective limit to positive tail
        "f_alpha1": {
            "values": [
                -0.02500,
                -0.02200,
                -0.02000,
                -0.01800,
                -0.01500,
                -0.01300,
                -0.01100,
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
        # Wide positive range enabled by relaxed diffusive stability ceiling
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
# Continuous Bayesian Optimization Configuration for R = 100
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

