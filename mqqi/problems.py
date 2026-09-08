"""
Benchmark problem configurations, analytical solutions, and reference paper data
for the Burgers' equation experiments in Chen & Wu (2006).
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional
import numpy as np


@dataclass
class BenchmarkCase:
    """
    Configuration specification for a Burgers' equation test problem.
    """
    name: str
    R: float
    m: int
    tau: float
    lam: float
    u0_func: Callable[[np.ndarray], np.ndarray]
    t_end: float
    record_times: List[float]
    description: str
    paper_reference: str
    x_span: tuple = (0.0, 1.0)


def initial_condition_1(x: np.ndarray) -> np.ndarray:
    """Initial condition: u(x, 0) = sin(pi * x)."""
    return np.sin(np.pi * x)


def initial_condition_2(x: np.ndarray) -> np.ndarray:
    """Initial condition: u(x, 0) = sin(2 * pi * x) + 0.5 * sin(pi * x)."""
    return np.sin(2.0 * np.pi * x) + 0.5 * np.sin(np.pi * x)


def cole_analytic_solution(
    x: np.ndarray, t: float, R: float, n_terms: int = 120, n_quad: int = 2000
) -> np.ndarray:
    """
    Computes Cole's exact analytical solution (Cole, 1951) for Burgers' equation:
        u_t + u * u_x = (1 / R) * u_xx
        u(x, 0) = sin(pi * x),  u(0, t) = u(1, t) = 0.

    Using the Cole-Hopf transformation:
        u(x, t) = (2 * pi / R) * [sum_n n * exp(-n^2 * pi^2 * t / R) * a_n * sin(n * pi * x)] /
                                  [a_0 / 2 + sum_n exp(-n^2 * pi^2 * t / R) * a_n * cos(n * pi * x)]
    where:
        a_0 = 2 * int_0^1 exp(-R / (2 * pi) * (1 - cos(pi * eta))) d eta
        a_n = 2 * int_0^1 exp(-R / (2 * pi) * (1 - cos(pi * eta))) * cos(n * pi * eta) d eta

    Parameters
    ----------
    x : np.ndarray
        Spatial evaluation points in [0, 1].
    t : float
        Evaluation time t > 0.
    R : float
        Reynolds number.
    n_terms : int, default=120
        Number of Fourier series terms.
    n_quad : int, default=2000
        Number of quadrature nodes for computing Fourier-Bessel coefficients.

    Returns
    -------
    u : np.ndarray
        Analytical solution values at coordinates x.
    """
    eta = np.linspace(0.0, 1.0, n_quad)
    deta = 1.0 / (n_quad - 1)

    integrand_base = np.exp(-(R / (2.0 * np.pi)) * (1.0 - np.cos(np.pi * eta)))

    # Trapezoidal integration weights
    weights = np.ones(n_quad)
    weights[0] = 0.5
    weights[-1] = 0.5

    # a_0 = 2 * int_0^1 ...
    a0 = 2.0 * np.sum(integrand_base * weights) * deta

    num = np.zeros_like(x, dtype=float)
    den = np.full_like(x, 0.5 * a0, dtype=float)

    for n in range(1, n_terms + 1):
        cos_n_eta = np.cos(n * np.pi * eta)
        an = 2.0 * np.sum(integrand_base * cos_n_eta * weights) * deta

        decay = np.exp(-(n**2 * np.pi**2 * t) / R)
        if decay < 1e-18:
            break

        num += n * an * decay * np.sin(n * np.pi * x)
        den += an * decay * np.cos(n * np.pi * x)

    return (2.0 * np.pi / R) * (num / den)


# Predefined benchmark configurations
PROBLEM_CASES: Dict[str, BenchmarkCase] = {
    "fig1": BenchmarkCase(
        name="Fig1_R10",
        R=10.0,
        m=100,
        tau=1e-3,
        lam=0.0072,
        u0_func=initial_condition_1,
        t_end=1.0,
        record_times=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        description="Figure 1 & Table 1: R = 10, h = 1/100, tau = 1e-3, lambda = 0.0072",
        paper_reference="Figure 1 & Table 1",
    ),
    "fig2": BenchmarkCase(
        name="Fig2_R100",
        R=100.0,
        m=100,
        tau=1e-3,
        lam=0.0029,
        u0_func=initial_condition_1,
        t_end=1.0,
        record_times=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        description="Figure 2 & Table 2: R = 100, h = 1/100, tau = 1e-3, lambda = 0.0029",
        paper_reference="Figure 2 & Table 2",
    ),
    "fig3": BenchmarkCase(
        name="Fig3_R10000",
        R=10000.0,
        m=72,
        tau=1e-3,
        lam=1.43e-4,
        u0_func=initial_condition_1,
        t_end=1.0,
        record_times=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        description="Figure 3 & Table 3: R = 10000, h = 1/72, tau = 1e-3, lambda = 1.43e-4",
        paper_reference="Figure 3 & Table 3",
    ),
    "fig4": BenchmarkCase(
        name="Fig4_Shock_R10000",
        R=10000.0,
        m=80,
        tau=1e-3,
        lam=1.2e-4,
        u0_func=initial_condition_2,
        t_end=2.0,
        record_times=[0.2, 0.6, 1.0, 1.4, 2.0],
        description="Figure 4: Shock propagation, R = 10000, h = 1/80, tau = 1e-3, lambda = 1.2e-4",
        paper_reference="Figure 4",
    ),
}

# Reference paper table values at t = 1.0
PAPER_TABLE_1_DATA = {
    # x: (Analytic, Hon_Mao, MQQI_Paper)
    0.10: (0.0663, 0.0664, 0.071238),
    0.20: (0.1312, 0.1313, 0.134310),
    0.30: (0.1928, 0.1928, 0.193390),
    0.40: (0.2480, 0.2481, 0.245380),
    0.50: (0.2919, 0.2919, 0.285170),
    0.60: (0.3161, 0.3159, 0.304730),
    0.70: (0.3081, 0.3079, 0.292880),
    0.80: (0.2537, 0.2534, 0.237840),
    0.90: (0.1461, 0.1459, 0.135420),
}

PAPER_TABLE_2_DATA = {
    # x: (Analytic, Hon_Mao, MQQI_Paper)
    0.10: (0.0754, 0.0755, 0.078675),
    0.20: (0.1506, 0.1507, 0.152020),
    0.30: (0.2257, 0.2257, 0.225540),
    0.40: (0.3003, 0.3003, 0.299040),
    0.50: (0.3744, 0.3744, 0.372260),
    0.60: (0.4478, 0.4478, 0.444840),
    0.70: (0.5203, 0.5202, 0.516430),
    0.80: (0.5915, 0.5913, 0.586220),
    0.90: (0.6600, 0.6607, 0.629560),
}

PAPER_TABLE_3_DATA = {
    # x: (Christie, Hon_Mao, MQQI_Paper)
    0.056: (0.0422, 0.0424, 0.042395),
    0.111: (0.0843, 0.0843, 0.084337),
    0.167: (0.1263, 0.1263, 0.126190),
    0.222: (0.1684, 0.1684, 0.167960),
    0.278: (0.2103, 0.2103, 0.209570),
    0.333: (0.2522, 0.2522, 0.251290),
    0.389: (0.2939, 0.2939, 0.292810),
    0.444: (0.3355, 0.3355, 0.334200),
    0.500: (0.3769, 0.3769, 0.375440),
    0.556: (0.4182, 0.4182, 0.414680),
    0.611: (0.4592, 0.4592, 0.457290),
    0.667: (0.5000, 0.4999, 0.497830),
    0.722: (0.5404, 0.5404, 0.538060),
    0.778: (0.5806, 0.5805, 0.577940),
    0.833: (0.6203, 0.6201, 0.617390),
    0.889: (0.6596, 0.6600, 0.656350),
    0.944: (0.6983, 0.6957, 0.694750),
}


def get_problem_case(case_name: str) -> BenchmarkCase:
    """Retrieve benchmark case by key name (fig1, fig2, fig3, fig4)."""
    case_key = case_name.lower()
    if case_key not in PROBLEM_CASES:
        raise KeyError(
            f"Case '{case_name}' not found. Available cases: {list(PROBLEM_CASES.keys())}"
        )
    return PROBLEM_CASES[case_key]

