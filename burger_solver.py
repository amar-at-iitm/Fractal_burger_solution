"""Classical Multiquadric Quasi-Interpolation (Wu-Schaback L_W2) Solver for Burgers' Equation.

This module implements the classical Wu-Schaback multiquadric (MQ)
quasi-interpolation operator L_W2 and its second derivative, combined with
the Chen & Wu (2006) MQQI first spatial derivative and upwind dispersion-damping
switch function to solve the viscous 1D Burgers' equation:

    u_t + u * u_x = (1 / R) * u_xx,   x in [0, 1], t > 0
    with u(0, t) = u(1, t) = 0.
"""

from typing import Callable, Dict, List, Optional, Tuple, Union
import numpy as np

__all__ = [
    "varphi",
    "phi",
    "d2_phi",
    "psi",
    "d2_psi",
    "second_divided_difference",
    "build_divided_difference_matrix",
    "rbf_matrix",
    "rbf_correction",
    "error_function",
    "L_W2",
    "d2_L_W2",
    "d2_psi_matrix",
    "build_classical_spatial_operator",
    "build_ux_matrix",
    "compute_switch_function",
    "step_classical_burgers",
    "solve_classical_burgers",
]


def varphi(r: np.ndarray, s: float) -> np.ndarray:
    """RBF kernel function: varphi(r) = s^2 / (s^2 + r^2)^(3/2)."""
    return s**2 / (s**2 + r**2) ** 1.5


def phi(a: Union[float, np.ndarray], b: float, c: float) -> Union[float, np.ndarray]:
    """Multiquadric radial basis function: phi(a, b) = sqrt(c^2 + (a - b)^2)."""
    return np.sqrt(c**2 + (a - b) ** 2)


def d2_phi(a: Union[float, np.ndarray], b: float, c: float) -> Union[float, np.ndarray]:
    """Second derivative of MQ RBF: phi''(a, b) = c^2 / (c^2 + (a - b)^2)^(3/2)."""
    return c**2 / (c**2 + (a - b) ** 2) ** 1.5


def psi(a: float, j: int, x: np.ndarray, c: float) -> float:
    """Evaluates the Wu-Schaback quasi-interpolation basis function psi_j(a)."""
    n = len(x) - 1

    if j == 0:
        return 0.5 + (phi(a, x[1], c) - (a - x[0])) / (2.0 * (x[1] - x[0]))

    if j == 1:
        return (
            (phi(a, x[2], c) - phi(a, x[1], c)) / (2.0 * (x[2] - x[1]))
            - (phi(a, x[1], c) - (a - x[0])) / (2.0 * (x[1] - x[0]))
        )

    if j == n - 1:
        return (
            ((x[n] - a) - phi(a, x[n - 1], c)) / (2.0 * (x[n] - x[n - 1]))
            - (phi(a, x[n - 1], c) - phi(a, x[n - 2], c)) / (2.0 * (x[n - 1] - x[n - 2]))
        )

    if j == n:
        return 0.5 + (phi(a, x[n - 1], c) - (x[n] - a)) / (2.0 * (x[n] - x[n - 1]))

    return (
        (phi(a, x[j + 1], c) - phi(a, x[j], c)) / (2.0 * (x[j + 1] - x[j]))
        - (phi(a, x[j], c) - phi(a, x[j - 1], c)) / (2.0 * (x[j] - x[j - 1]))
    )


def d2_psi(a: float, j: int, x: np.ndarray, c: float) -> float:
    """Second derivative of the Wu-Schaback basis function psi_j''(a)."""
    n = len(x) - 1

    if j == 0:
        return d2_phi(a, x[1], c) / (2.0 * (x[1] - x[0]))

    if j == 1:
        return (
            (d2_phi(a, x[2], c) - d2_phi(a, x[1], c)) / (2.0 * (x[2] - x[1]))
            - d2_phi(a, x[1], c) / (2.0 * (x[1] - x[0]))
        )

    if j == n - 1:
        return (
            -d2_phi(a, x[n - 1], c) / (2.0 * (x[n] - x[n - 1]))
            - (d2_phi(a, x[n - 1], c) - d2_phi(a, x[n - 2], c)) / (2.0 * (x[n - 1] - x[n - 2]))
        )

    if j == n:
        return d2_phi(a, x[n - 1], c) / (2.0 * (x[n] - x[n - 1]))

    return (
        (d2_phi(a, x[j + 1], c) - d2_phi(a, x[j], c)) / (2.0 * (x[j + 1] - x[j]))
        - (d2_phi(a, x[j], c) - d2_phi(a, x[j - 1], c)) / (2.0 * (x[j] - x[j - 1]))
    )


def second_divided_difference(x: np.ndarray, f: np.ndarray, kj: int) -> float:
    """Computes the 3-point non-uniform second divided difference at interior index kj."""
    if kj <= 0 or kj >= len(x) - 1:
        raise ValueError("kj must be a strictly interior grid index.")

    d1 = np.diff(f) / np.diff(x)
    d2 = 2.0 * np.diff(d1) / (x[2:] - x[:-2])
    return float(d2[kj - 1])


def build_divided_difference_matrix(x: np.ndarray, k_idx: np.ndarray) -> np.ndarray:
    """Constructs the second divided difference matrix D of shape (len(k_idx), len(x))."""
    matrix = np.zeros((len(k_idx), len(x)), dtype=float)
    for row, kj in enumerate(k_idx):
        xm = x[kj - 1]
        x0 = x[kj]
        xp = x[kj + 1]
        denom = (x0 - xm) * (xp - x0) * (xp - xm)

        matrix[row, kj + 1] = 2.0 * (x0 - xm) / denom
        matrix[row, kj] = -2.0 * (xp - xm) / denom
        matrix[row, kj - 1] = 2.0 * (xp - x0) / denom
    return matrix


def rbf_matrix(xk: np.ndarray, s: float) -> np.ndarray:
    """Returns the RBF interpolation matrix A_ij = varphi(|x_{k_i} - x_{k_j}|, s)."""
    r = np.abs(xk[:, None] - xk[None, :])
    return varphi(r, s)


def rbf_correction(a: float, xk: np.ndarray, alpha: np.ndarray, s: float) -> float:
    """Evaluates the RBF correction R(a) = sum_j alpha_j sqrt(s^2 + (a - x_{k_j})^2)."""
    return float(np.sum(alpha * np.sqrt(s**2 + (a - xk) ** 2)))


def error_function(i: int, x: np.ndarray, f: np.ndarray, xk: np.ndarray, alpha: np.ndarray, s: float) -> float:
    """Returns the residual error e(x_i) = f(x_i) - R(x_i)."""
    return float(f[i] - rbf_correction(x[i], xk, alpha, s))


def L_W2(i: int, x: np.ndarray, f: np.ndarray, xk: np.ndarray, alpha: np.ndarray, s: float, c: float) -> float:
    """Evaluates the Wu-Schaback quasi-interpolant (L_W2 f)(x_i)."""
    rbf_part = rbf_correction(x[i], xk, alpha, s)
    e_vals = np.array([error_function(p, x, f, xk, alpha, s) for p in range(len(x))])
    d = np.array([psi(x[i], j, x, c) for j in range(len(x))])
    ld_part = float(e_vals @ d)
    return rbf_part + ld_part


def d2_L_W2(i: int, x: np.ndarray, f: np.ndarray, xk: np.ndarray, alpha: np.ndarray, s: float, c: float) -> float:
    """Evaluates the second derivative of the Wu-Schaback operator (L_W2 f)''(x_i).

    (L_W2 f)''(x_i) = sum_j alpha_j varphi(x_i - x_{k_j}, s) + sum_p e(x_p) psi_p''(x_i).
    """
    xx = x[i]
    rbf_part = float(np.sum(alpha * varphi(xx - xk, s)))
    e_vals = np.array([error_function(p, x, f, xk, alpha, s) for p in range(len(x))])
    ld_part = sum(e_vals[p] * d2_psi(xx, p, x, c) for p in range(len(x)))
    return rbf_part + ld_part


def d2_psi_matrix(x: np.ndarray, c: float) -> np.ndarray:
    """Precomputes the full matrix of second derivatives Psi''[i, j] = psi_j''(x_i)."""
    n = len(x) - 1
    psi_mat = np.empty((len(x), len(x)), dtype=float)

    for row, xx in enumerate(x):
        vals = np.empty(len(x), dtype=float)
        vals[0] = d2_phi(xx, x[1], c) / (2.0 * (x[1] - x[0]))
        vals[1] = (
            (d2_phi(xx, x[2], c) - d2_phi(xx, x[1], c)) / (2.0 * (x[2] - x[1]))
            - d2_phi(xx, x[1], c) / (2.0 * (x[1] - x[0]))
        )

        for j in range(2, n - 1):
            vals[j] = (
                (d2_phi(xx, x[j + 1], c) - d2_phi(xx, x[j], c)) / (2.0 * (x[j + 1] - x[j]))
                - (d2_phi(xx, x[j], c) - d2_phi(xx, x[j - 1], c)) / (2.0 * (x[j] - x[j - 1]))
            )

        vals[n - 1] = (
            -d2_phi(xx, x[n - 1], c) / (2.0 * (x[n] - x[n - 1]))
            - (d2_phi(xx, x[n - 1], c) - d2_phi(xx, x[n - 2], c)) / (2.0 * (x[n - 1] - x[n - 2]))
        )
        vals[n] = d2_phi(xx, x[n - 1], c) / (2.0 * (x[n] - x[n - 1]))
        psi_mat[row, :] = vals

    return psi_mat


def build_classical_spatial_operator(
    x: np.ndarray,
    xk: np.ndarray,
    k_idx: np.ndarray,
    s: float,
    c: float,
) -> np.ndarray:
    """Constructs the condensed linear operator matrix M_xx such that (u_xx) = M_xx @ u.

    Mathematical Derivation:
        e = u - R @ alpha
        alpha = A_rbf^{-1} @ D @ u
        u_xx = Psi'' @ e + Phi @ alpha
             = [Psi'' + (Phi - Psi'' @ R) @ A_rbf^{-1} @ D] @ u
    """
    A_rbf = rbf_matrix(xk, s)
    A_inv = np.linalg.inv(A_rbf)
    D = build_divided_difference_matrix(x, k_idx)
    Phi = varphi(x[:, None] - xk[None, :], s)
    R_mat = np.sqrt(s**2 + (x[:, None] - xk[None, :]) ** 2)
    Psi_dd = d2_psi_matrix(x, c)

    return Psi_dd + (Phi - Psi_dd @ R_mat) @ A_inv @ D


def build_ux_matrix(x: np.ndarray, c: float) -> np.ndarray:
    """Precomputes Chen & Wu (2006) MQQI first spatial derivative matrix A_ux.

    A_ux has shape (m + 1, m) such that u_x = A_ux @ (u[1:] - u[:-1]).
    """
    m = len(x) - 1
    h = (x[-1] - x[0]) / m
    diff = x[:, None] - x[:m][None, :]
    denom = np.sqrt(diff**2 + c**2)

    dphi = np.zeros((m + 1, m + 1), dtype=float)
    dphi[:, :m] = diff / denom
    dphi[:, m] = dphi[:, 0] - 2.0

    return 0.5 * (dphi[:, :m] - dphi[:, 1 : m + 1]) / h


def compute_switch_function(u: np.ndarray, ux: np.ndarray) -> np.ndarray:
    """Computes the upwind dispersion-damping switch function g_j^n (Eq. 3.3 in Chen & Wu):

    g_j^n = max{0, 1 + min{0, sign((u_x)_j^n * (u_x)_k^n)}}, where k = j - sign(u_j^n).
    """
    m = len(u) - 1
    sgn_u = np.sign(u).astype(int)
    j_indices = np.arange(m + 1)
    k_indices = np.clip(j_indices - sgn_u, 0, m)

    prod_sign = np.sign(ux * ux[k_indices])
    return np.maximum(0.0, 1.0 + np.minimum(0.0, prod_sign))


def step_classical_burgers(
    u: np.ndarray,
    tau: float,
    R: float,
    A_ux: np.ndarray,
    M_xx: np.ndarray,
) -> np.ndarray:
    """Performs one forward time step using classical L_W2 spatial derivatives."""
    m = len(u) - 1
    delta_u = u[1:] - u[:-1]
    ux = A_ux @ delta_u
    g = compute_switch_function(u, ux)
    uxx = M_xx @ u

    u_next = u.copy()
    u_next[1:m] = (
        u[1:m]
        - tau * u[1:m] * ux[1:m] * g[1:m]
        + (tau / R) * uxx[1:m]
    )
    u_next[0] = 0.0
    u_next[m] = 0.0
    return u_next


def solve_classical_burgers(
    u0: Union[Callable[[np.ndarray], np.ndarray], np.ndarray],
    x: np.ndarray,
    tau: float,
    R: float,
    t_end: float,
    s: float,
    c: float,
    K: int = 10,
    record_times: Optional[List[float]] = None,
) -> Tuple[np.ndarray, Dict[float, np.ndarray]]:
    """Integrates Burgers' equation from t = 0 to t = t_end using classical L_W2."""
    m = len(x) - 1
    N = m // K
    k_idx = np.concatenate(([1], K * np.arange(1, N - 1), [m - 1])).astype(int)
    xk = x[k_idx]

    A_ux = build_ux_matrix(x, c)
    M_xx = build_classical_spatial_operator(x, xk, k_idx, s, c)

    if callable(u0):
        u = u0(x)
    else:
        u = np.array(u0, dtype=float, copy=True)

    u[0] = 0.0
    u[m] = 0.0

    if record_times is None:
        record_times = [0.0, float(t_end)]

    step_to_time: Dict[int, float] = {
        int(round(t_target / tau)): t_target for t_target in record_times
    }

    solutions: Dict[float, np.ndarray] = {}
    if 0 in step_to_time:
        solutions[step_to_time[0]] = u.copy()

    total_steps = int(round(t_end / tau))
    for step_idx in range(1, total_steps + 1):
        u = step_classical_burgers(u, tau, R, A_ux, M_xx)
        if step_idx in step_to_time:
            solutions[step_to_time[step_idx]] = u.copy()

    for t_target in record_times:
        if t_target not in solutions:
            solutions[t_target] = u.copy()

    return x, solutions

