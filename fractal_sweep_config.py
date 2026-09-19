"""WandB Hyperparameter Sweep Configuration for Burgers Fractal Optimization.

Defines fine-grained parameter search values for scaling coefficients (f_alpha1, f_alpha2, f_alpha3)
used in the fractalized L_W2 multiquadric quasi-interpolation operator.

Stability & Performance Notes:
- All values are strictly within the Forward Euler CFL stability boundary (|alpha| <= 0.0010 for N_sub=6 on [-1, 1]).
- The grid is densely concentrated in the optimal micro-scale regularizing zone ([-0.00005, -0.00050])
  where localized negative contractivity regularizes convective shock steepening.
- Contains 23 finely calibrated values per parameter (at least 20 values).
"""

sweep_config = {
    "method": "grid",
    "metric": {"name": "Linf_error", "goal": "minimize"},
    "parameters": {
        "f_alpha1": {
            "values": [
                0.0,
                -0.00002,
                -0.00005,
                -0.00008,
                -0.00010,
                -0.00012,
                -0.00015,
                -0.00018,
                -0.00020,
                -0.00025,
                -0.00030,
                -0.00035,
                -0.00040,
                -0.00045,
                -0.00050,
                -0.00060,
                -0.00070,
                -0.00080,
                -0.00090,
                -0.00100,
                0.00002,
                0.00005,
                0.00010,
            ]
        },
        "f_alpha2": {
            "values": [
                0.0,
                -0.00002,
                -0.00005,
                -0.00008,
                -0.00010,
                -0.00012,
                -0.00015,
                -0.00018,
                -0.00020,
                -0.00025,
                -0.00030,
                -0.00035,
                -0.00040,
                -0.00045,
                -0.00050,
                -0.00060,
                -0.00070,
                -0.00080,
                -0.00090,
                -0.00100,
                0.00002,
                0.00005,
                0.00010,
            ]
        },
        "f_alpha3": {
            "values": [
                0.0,
                -0.00002,
                -0.00005,
                -0.00008,
                -0.00010,
                -0.00012,
                -0.00015,
                -0.00018,
                -0.00020,
                -0.00025,
                -0.00030,
                -0.00035,
                -0.00040,
                -0.00045,
                -0.00050,
                -0.00060,
                -0.00070,
                -0.00080,
                -0.00090,
                -0.00100,
                0.00002,
                0.00005,
                0.00010,
            ]
        },
    },
}
