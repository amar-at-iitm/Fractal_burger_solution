"""Fractal Multiquadric Quasi-Interpolation (Fractal L_W2) Solver for Burgers' Equation.

This module implements the alpha-fractal multiquadric (MQ) quasi-interpolation
operator L_W2^alpha and its second derivative, constructed using a 5th-degree
Hermite polynomial base function H5 that matches the MQ basis function and its
first two derivatives at the boundary nodes.
"""

from typing import Callable, Dict, List, Optional, Tuple, Union
import numpy as np

from alpha_fractal_function import alpha_fractalize_second_derivative
from burger_solver import (
    varphi,
    error_function,
    rbf_matrix,
    build_divided_difference_matrix,
    build_ux_matrix,
    compute_switch_function,
)

__all__ = [
    "phi",
    "dphi",
    "ddphi",
    "H5",
    "H5_dd",
    "build_fractal_second_derivative",
    "pointwise_fractal",
    "d2_fractal_psi",
    "d2_fractal_L_W2",
    "d2_fractal_psi_matrix",
    "build_fractal_spatial_operator",
    "step_fractal_burgers",
    "solve_fractal_burgers",
]


def phi(z: Union[float, np.ndarray], c: float = 0.0072) -> Union[float, np.ndarray]:
    """Multiquadric (MQ) radial basis function: phi(z) = sqrt(c^2 + z^2)."""
    return np.sqrt(c**2 + z**2)


def dphi(z: Union[float, np.ndarray], c: float = 0.0072) -> Union[float, np.ndarray]:
    """First derivative of the MQ RBF: phi'(z) = z / sqrt(c^2 + z^2)."""
    return z / np.sqrt(c**2 + z**2)


def ddphi(z: Union[float, np.ndarray], c: float = 0.0072) -> Union[float, np.ndarray]:
    """Second derivative of the MQ RBF: phi''(z) = c^2 / (c^2 + z^2)^(3/2)."""
    return c**2 / (c**2 + z**2) ** 1.5


def H5(
    z: Union[float, np.ndarray],
    x1: float = -2.0,
    xN: float = 2.0,
    c: float = 0.0072,
) -> np.ndarray:
    """5th-degree Hermite polynomial base function H5(z).

    Interpolates phi, phi', and phi'' at the boundary points x1 and xN.
    """
    z_arr = np.asarray(z, dtype=float)
    dx = xN - x1

    phi1 = phi(x1, c)
    phiN = phi(xN, c)
    phi1d = dphi(x1, c)
    phiNd = dphi(xN, c)
    phi1dd = ddphi(x1, c)
    phiNdd = ddphi(xN, c)

    h1 = (phiN - phi1 - phi1d * dx - 0.5 * phi1dd * dx**2) / dx**3
    h2 = (3.0 * (phi1 - phiN) + 2.0 * (phi1d + 0.5 * phiNd) * dx + 0.5 * phi1dd * dx**2) / dx**4
    h3 = (6.0 * (phiN - phi1) - 3.0 * (phi1d + phiNd) * dx + 0.5 * (phiNdd - phi1dd) * dx**2) / dx**5

    dz = z_arr - x1
    return (
        phi1
        + phi1d * dz
        + 0.5 * phi1dd * dz**2
        + h1 * dz**3
        + h2 * dz**3 * (z_arr - xN)
        + h3 * dz**3 * (z_arr - xN) ** 2
    )


def H5_dd(
    z: Union[float, np.ndarray],
    x1: float = -2.0,
    xN: float = 2.0,
    c: float = 0.0072,
) -> np.ndarray:
    """Second derivative of the 5th-degree Hermite polynomial base function H5''(z)."""
    z_arr = np.asarray(z, dtype=float)
    dx = xN - x1

    phi1 = phi(x1, c)
    phiN = phi(xN, c)
    phi1d = dphi(x1, c)
    phiNd = dphi(xN, c)
    phi1dd = ddphi(x1, c)
    phiNdd = ddphi(xN, c)

    h1 = (phiN - phi1 - phi1d * dx - 0.5 * phi1dd * dx**2) / dx**3
    h2 = (3.0 * (phi1 - phiN) + 2.0 * (phi1d + 0.5 * phiNd) * dx + 0.5 * phi1dd * dx**2) / dx**4
    h3 = (6.0 * (phiN - phi1) - 3.0 * (phi1d + phiNd) * dx + 0.5 * (phiNdd - phi1dd) * dx**2) / dx**5

    dz = z_arr - x1
    w = z_arr - xN

    return (
        phi1dd
        + 6.0 * h1 * dz
        + h2 * (6.0 * dz * w + 6.0 * dz**2)
        + h3 * (6.0 * dz * w**2 + 12.0 * dz**2 * w + 2.0 * dz**3)
    )


def build_fractal_second_derivative(
    c: float,
    f_alpha: Union[float, List[float], np.ndarray],
    n_iter: int = 3,
    a_domain: float = -2.0,
    b_domain: float = 2.0,
    n_sub: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """Constructs the precomputed alpha-fractalized MQ second derivative table.

    Parameters
    ----------
    c : float
        MQ shape parameter.
    f_alpha : float, list, or array
        Fractal scaling parameters alpha_k.
    n_iter : int, default=3
        Number of fractal recursion stages.
    a_domain, b_domain : float, default=(-2.0, 2.0)
        Domain encompassing all difference coordinates (x_i - x_j).
    n_sub : int, optional
        Number of subintervals. If None, inferred from len(f_alpha).
    """
    if n_sub is None:
        n_sub = len(f_alpha) if hasattr(f_alpha, "__len__") else 10

    f2_func = lambda z: ddphi(z, c=c)
    g2_func = lambda z: H5_dd(z, x1=a_domain, xN=b_domain, c=c)

    return alpha_fractalize_second_derivative(
        f2=f2_func,
        g2=g2_func,
        a=a_domain,
        b=b_domain,
        n_sub=n_sub,
        in_alpha=f_alpha,
        n_iter=n_iter,
        dict=True,
    )


def pointwise_fractal(z: Union[float, np.ndarray], fractal_dd: Dict[str, np.ndarray]) -> Union[float, np.ndarray]:
    """Evaluates the fractal second derivative at query coordinate(s) z via linear interpolation."""
    p = fractal_dd["partition"]
    v = fractal_dd["values"]
    return np.interp(z, p, v)


def d2_fractal_psi(a: float, j: int, x: np.ndarray, fractal_dd: Dict[str, np.ndarray]) -> float:
    """Second derivative of the fractal basis function psi_j''(a)."""
    n = len(x) - 1

    if j == 0:
        return float(pointwise_fractal(a - x[1], fractal_dd) / (2.0 * (x[1] - x[0])))

    if j == 1:
        return float(
            (pointwise_fractal(a - x[2], fractal_dd) - pointwise_fractal(a - x[1], fractal_dd)) / (2.0 * (x[2] - x[1]))
            - pointwise_fractal(a - x[1], fractal_dd) / (2.0 * (x[1] - x[0]))
        )

    if j == n - 1:
        return float(
            -pointwise_fractal(a - x[n - 1], fractal_dd) / (2.0 * (x[n] - x[n - 1]))
            - (pointwise_fractal(a - x[n - 1], fractal_dd) - pointwise_fractal(a - x[n - 2], fractal_dd))
            / (2.0 * (x[n - 1] - x[n - 2]))
        )

    if j == n:
        return float(pointwise_fractal(a - x[n - 1], fractal_dd) / (2.0 * (x[n] - x[n - 1])))

    return float(
        (pointwise_fractal(a - x[j + 1], fractal_dd) - pointwise_fractal(a - x[j], fractal_dd)) / (2.0 * (x[j + 1] - x[j]))
        - (pointwise_fractal(a - x[j], fractal_dd) - pointwise_fractal(a - x[j - 1], fractal_dd)) / (2.0 * (x[j] - x[j - 1]))
    )


def d2_fractal_L_W2(
    i: int,
    x: np.ndarray,
    f: np.ndarray,
    xk: np.ndarray,
    alpha: np.ndarray,
    s: float,
    fractal_dd: Dict[str, np.ndarray],
) -> float:
    """Evaluates the second derivative of the fractal quasi-interpolant (L_W2^alpha f)''(x_i)."""
    xx = x[i]
    rbf_part = float(np.sum(alpha * varphi(xx - xk, s)))
    e_vals = np.array([error_function(p, x, f, xk, alpha, s) for p in range(len(x))])
    ld_part = sum(e_vals[p] * d2_fractal_psi(xx, p, x, fractal_dd) for p in range(len(x)))
    return rbf_part + ld_part


def d2_fractal_psi_matrix(x: np.ndarray, fractal_dd: Dict[str, np.ndarray]) -> np.ndarray:
    """Precomputes the full matrix of fractal second derivatives Psi^alpha''[i, j] = psi_j^alpha''(x_i)."""
    p = fractal_dd["partition"]
    v = fractal_dd["values"]
    n = len(x) - 1
    psi_mat = np.empty((len(x), len(x)), dtype=float)

    def q(z: np.ndarray) -> np.ndarray:
        return np.interp(z, p, v)

    for row, xx in enumerate(x):
        vals = np.empty(len(x), dtype=float)
        vals[0] = q(xx - x[1]) / (2.0 * (x[1] - x[0]))
        vals[1] = (
            (q(xx - x[2]) - q(xx - x[1])) / (2.0 * (x[2] - x[1]))
            - q(xx - x[1]) / (2.0 * (x[1] - x[0]))
        )

        for j in range(2, n - 1):
            vals[j] = (
                (q(xx - x[j + 1]) - q(xx - x[j])) / (2.0 * (x[j + 1] - x[j]))
                - (q(xx - x[j]) - q(xx - x[j - 1])) / (2.0 * (x[j] - x[j - 1]))
            )

        vals[n - 1] = (
            -q(xx - x[n - 1]) / (2.0 * (x[n] - x[n - 1]))
            - (q(xx - x[n - 1]) - q(xx - x[n - 2])) / (2.0 * (x[n - 1] - x[n - 2]))
        )
        vals[n] = q(xx - x[n - 1]) / (2.0 * (x[n] - x[n - 1]))
        psi_mat[row, :] = vals

    return psi_mat


def build_fractal_spatial_operator(
    x: np.ndarray,
    xk: np.ndarray,
    k_idx: np.ndarray,
    s: float,
    fractal_dd: Dict[str, np.ndarray],
) -> np.ndarray:
    """Constructs the condensed linear operator matrix M_xx^fractal such that (u_xx)_fractal = M_xx^fractal @ u."""
    A_rbf = rbf_matrix(xk, s)
    A_inv = np.linalg.inv(A_rbf)
    D = build_divided_difference_matrix(x, k_idx)
    Phi = varphi(x[:, None] - xk[None, :], s)
    R_mat = np.sqrt(s**2 + (x[:, None] - xk[None, :]) ** 2)
    Psi_fractal_dd = d2_fractal_psi_matrix(x, fractal_dd)

    return Psi_fractal_dd + (Phi - Psi_fractal_dd @ R_mat) @ A_inv @ D


def step_fractal_burgers(
    u: np.ndarray,
    tau: float,
    R: float,
    A_ux: np.ndarray,
    M_xx_fractal: np.ndarray,
) -> np.ndarray:
    """Performs one forward time step using fractal L_W2 spatial derivatives."""
    m = len(u) - 1
    delta_u = u[1:] - u[:-1]
    ux = A_ux @ delta_u
    g = compute_switch_function(u, ux)
    uxx = M_xx_fractal @ u

    u_next = u.copy()
    u_next[1:m] = (
        u[1:m]
        - tau * u[1:m] * ux[1:m] * g[1:m]
        + (tau / R) * uxx[1:m]
    )
    u_next[0] = 0.0
    u_next[m] = 0.0
    return u_next


def solve_fractal_burgers(
    u0: Union[Callable[[np.ndarray], np.ndarray], np.ndarray],
    x: np.ndarray,
    tau: float,
    R: float,
    t_end: float,
    s: float,
    c: float,
    f_alpha: Union[float, List[float], np.ndarray],
    n_iter: int = 3,
    K: int = 10,
    record_times: Optional[List[float]] = None,
) -> Tuple[np.ndarray, Dict[float, np.ndarray]]:
    """Integrates Burgers' equation from t = 0 to t = t_end using fractal L_W2."""
    m = len(x) - 1
    N = m // K
    k_idx = np.concatenate(([1], K * np.arange(1, N - 1), [m - 1])).astype(int)
    xk = x[k_idx]

    A_ux = build_ux_matrix(x, c)
    fractal_dd = build_fractal_second_derivative(c=c, f_alpha=f_alpha, n_iter=n_iter)
    M_xx_fractal = build_fractal_spatial_operator(x, xk, k_idx, s, fractal_dd)

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
        u = step_fractal_burgers(u, tau, R, A_ux, M_xx_fractal)
        if step_idx in step_to_time:
            solutions[step_to_time[step_idx]] = u.copy()

    for t_target in record_times:
        if t_target not in solutions:
            solutions[t_target] = u.copy()

    return x, solutions
