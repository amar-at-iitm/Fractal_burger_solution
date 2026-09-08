"""
MQQI (Multiquadric Quasi-Interpolation) Burgers' Equation Solver
Based on the paper:
Chen, R., & Wu, Z. (2006). Applying multiquadric quasi-interpolation to solve
Burgers equation. Applied Mathematics and Computation, 172(1), 472-484.
"""

from .solver import MQQIBurgersSolver
from .problems import (
    BenchmarkCase,
    get_problem_case,
    cole_analytic_solution,
    PROBLEM_CASES,
)
from .plotting import (
    plot_paper_figure_1_to_3,
    plot_paper_figure_4,
    plot_comparison_with_analytic,
)

__all__ = [
    "MQQIBurgersSolver",
    "BenchmarkCase",
    "get_problem_case",
    "cole_analytic_solution",
    "PROBLEM_CASES",
    "plot_paper_figure_1_to_3",
    "plot_paper_figure_4",
    "plot_comparison_with_analytic",
]

