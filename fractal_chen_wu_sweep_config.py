"""WandB Hyperparameter Sweep Configuration for Chen & Wu (2006) MQQI (Eq. 2.1 & 2.11).

Defines the full admissible parameter search values for scaling coefficients (f_alpha1, f_alpha2, f_alpha3)
used in the directly fractalized second-derivative operator of Chen & Wu (2006) (Eq. 2.11 & 3.6).

Theoretical & Numerical Constraints:
-------------------------------------
1. Banach C^2 Smoothness Ceiling:
   - For uniform N_sub = 6 on [-1, 1], the affine contraction ratio is a_i = 1/6.
   - The Read-Bajraktarevic operator on C^2(I) has effective scaling:
         tilde{alpha}_i^(2) = alpha_i / a_i^2 = 36 * alpha_i.
   - Strict contractivity requires:
         max_i |tilde{alpha}_i^(2)| < 1.0  ==>  max_i |alpha_i| < a_i^2 = 1/36 ≈ 0.027778.
   - All parameters are strictly bounded within (-0.02778, +0.02778).

2. Forward Euler Spectral Radius CFL Ceiling:
   - Discrete diffusive stability requires (tau / R) * |lambda_max(M_xx)| <= 2.0.
   - For R=10, tau=0.001: |lambda_max(M_xx)| <= 20,000.0.
   - Central scaling f_alpha3 sharpens peak MQ curvature:
         f_alpha3 <= 0.00265  -->  |lambda_max| <= 19,996.1  (STABLE)
         f_alpha3 >= 0.00268  -->  |lambda_max| >  20,000.0  (CFL BLOWUP: NaN/Inf)
   - Thus, f_alpha3 is upper-bounded strictly by +0.00265.

3. Convective Boundary Floor on f_alpha1:
   - Outer tails (|z| >= 2/3) interact with homogeneous Dirichlet boundaries u(0)=u(1)=0.
   - Negative contractivity in [-0.0125, -0.0020] flattens outer tails and damps boundary reflections,
     reducing Linf error by 68.5% and RMS error by 82.4%.
   - If f_alpha1 < -0.0130, excessive tail gradients induce nonlinear convective instability.
   - Thus, f_alpha1 is lower-bounded strictly by -0.0125.

4. Full Search Ranges Bounded ONLY by Stability & C^2 Smoothness:
   - f_alpha1 in [-0.01250, +0.00500] (Convective floor: -0.0125; Upper stability: +0.0050)
   - f_alpha2 in [-0.00200, +0.00200] (Coupled transition stability range)
   - f_alpha3 in [-0.00200, +0.00265] (Anti-diffusion test floor: -0.0020; Diffusive CFL ceiling: +0.00265)
"""

# =============================================================================
# Grid Sweep Configuration (Full Admissible Discrete Space)
# =============================================================================
sweep_config = {
    "method": "grid",
    "metric": {"name": "Linf_error", "goal": "minimize"},
    "parameters": {
        # f_alpha1: Boundary subintervals (|z| in [2/3, 1])
        # Full range from convective instability limit (-0.0125) to positive tail (+0.0050)
        "f_alpha1": {
            "values": [
                -0.01250,
                -0.01200,
                -0.01000,
                -0.00800,
                -0.00600,
                -0.00500,
                -0.00400,
                -0.00300,
                -0.00200,
                -0.00100,
                -0.00050,
                -0.00020,
                -0.00010,
                0.0,
                0.00010,
                0.00020,
                0.00050,
                0.00100,
                0.00150,
                0.00200,
                0.00300,
                0.00400,
                0.00500,
            ]
        },
        # f_alpha2: Intermediate transition subintervals (|z| in [1/3, 2/3])
        # Full coupled stability range from -0.0020 to +0.0020
        "f_alpha2": {
            "values": [
                -0.00200,
                -0.00100,
                -0.00050,
                -0.00020,
                -0.00010,
                0.0,
                0.00020,
                0.00050,
                0.00080,
                0.00100,
                0.00110,
                0.00120,
                0.00125,
                0.00130,
                0.00135,
                0.00140,
                0.00145,
                0.00150,
                0.00155,
                0.00160,
                0.00170,
                0.00185,
                0.00200,
            ]
        },
        # f_alpha3: Core interior subintervals centered at z = 0 (|z| in [0, 1/3])
        # Full range from anti-diffusion exploration (-0.0020) up to strict CFL ceiling (+0.00265)
        "f_alpha3": {
            "values": [
                -0.00200,
                -0.00100,
                -0.00050,
                -0.00020,
                -0.00010,
                0.0,
                0.00050,
                0.00100,
                0.00150,
                0.00180,
                0.00200,
                0.00210,
                0.00220,
                0.00230,
                0.00240,
                0.00245,
                0.00250,
                0.00255,
                0.00258,
                0.00260,
                0.00262,
                0.00264,
                0.00265,
            ]
        },
    },
}

# =============================================================================
# Bayesian Optimization Sweep Configuration (Continuous Uniform Bounds)
# =============================================================================
sweep_config_bayes = {
    "method": "bayes",
    "metric": {"name": "Linf_error", "goal": "minimize"},
    "parameters": {
        "f_alpha1": {"distribution": "uniform", "min": -0.01250, "max": 0.00500},
        "f_alpha2": {"distribution": "uniform", "min": -0.00200, "max": 0.00200},
        "f_alpha3": {"distribution": "uniform", "min": -0.00200, "max": 0.00265},
    },
}

