"""
Core Multiquadric Quasi-Interpolation (MQQI) Numerical Solver for Burgers' Equation.

Governing Equation:
    u_t + u * u_x = (1 / R) * u_xx,   x in [x0, xm], t > 0
    with boundary conditions: u(x0, t) = u(xm, t) = 0.

Spatial discretization uses the derivatives of the univariate Multiquadric (MQ)
quasi-interpolation developed by Chen & Wu (2006).
Time stepping uses explicit forward Euler with an upwind dispersion-damping switch g_j^n.
"""

from typing import Callable, Dict, List, Optional, Tuple, Union
import numpy as np


class MQQIBurgersSolver:
    """
    Multiquadric Quasi-Interpolation (MQQI) Solver for the 1D viscous Burgers' equation.

    Parameters
    ----------
    m : int
        Number of spatial subintervals (number of grid points is m + 1).
    tau : float
        Time step size (delta t).
    lam : float
        Multiquadric shape parameter (lambda).
    R : float
        Reynolds number.
    x_span : Tuple[float, float], default=(0.0, 1.0)
        Spatial domain interval [x0, xm].
    """

    def __init__(
        self,
        m: int,
        tau: float,
        lam: float,
        R: float,
        x_span: Tuple[float, float] = (0.0, 1.0),
    ):
        self.m = int(m)
        self.tau = float(tau)
        self.lam = float(lam)
        self.R = float(R)
        self.x0, self.xm = float(x_span[0]), float(x_span[1])

        # Spatial grid
        self.x = np.linspace(self.x0, self.xm, self.m + 1)
        self.h = (self.xm - self.x0) / self.m

        # Precompute spatial differentiation matrices A and B
        self._precompute_differentiation_matrices()

    def _precompute_differentiation_matrices(self) -> None:
        """
        Precomputes the spatial quasi-interpolation derivative operator matrices:
            A[j, k] = 0.5 * (phi'_k(x_j) - phi'_{k+1}(x_j)) / (x_{k+1} - x_k)
            B[j, k] = 0.5 * (phi''_k(x_j) - phi''_{k+1}(x_j)) / (x_{k+1} - x_k)
        for j = 0, ..., m and k = 0, ..., m - 1.

        Basis functions:
            phi_k(x) = sqrt((x - x_k)^2 + lambda^2) for k = 0, ..., m - 1
            phi_m(x) = phi_0(x) - 2x + xm + x0
        """
        m = self.m
        x = self.x
        lam = self.lam
        h = self.h

        # Compute differences: (x_j - x_k) for j=0..m, k=0..m-1
        X_j = x[:, np.newaxis]          # Shape (m+1, 1)
        X_k = x[:m][np.newaxis, :]      # Shape (1, m)
        diff = X_j - X_k                # Shape (m+1, m)
        denom = np.sqrt(diff**2 + lam**2)

        # Allocate phi' and phi'' arrays for k = 0, ..., m
        dphi = np.zeros((m + 1, m + 1), dtype=float)
        d2phi = np.zeros((m + 1, m + 1), dtype=float)

        # For k = 0, ..., m - 1:
        # phi'_k(x) = (x - x_k) / sqrt((x - x_k)^2 + lam^2)
        # phi''_k(x) = lam^2 / ((x - x_k)^2 + lam^2)^(3/2)
        dphi[:, :m] = diff / denom
        d2phi[:, :m] = (lam**2) / (denom**3)

        # For k = m:
        # phi_m(x) = phi_0(x) - 2x + xm + x0
        # phi'_m(x) = phi'_0(x) - 2
        # phi''_m(x) = phi''_0(x)
        dphi[:, m] = dphi[:, 0] - 2.0
        d2phi[:, m] = d2phi[:, 0]

        # Assemble operator matrices:
        # A: (m+1, m) for first spatial derivative
        # B: (m+1, m) for second spatial derivative
        self.A = 0.5 * (dphi[:, :m] - dphi[:, 1 : m + 1]) / h
        self.B = 0.5 * (d2phi[:, :m] - d2phi[:, 1 : m + 1]) / h

    def compute_spatial_derivatives(
        self, u: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Approximates the first and second spatial derivatives u_x and u_xx
        using the precomputed MQQI operators.

        Parameters
        ----------
        u : np.ndarray
            Values of u on the grid, length m + 1.

        Returns
        -------
        ux : np.ndarray
            Approximated first derivative (u_x)_j, length m + 1.
        uxx : np.ndarray
            Approximated second derivative (u_xx)_j, length m + 1.
        """
        # delta_u_k = u_{k+1} - u_k for k = 0, ..., m - 1
        delta_u = u[1:] - u[:-1]
        ux = self.A @ delta_u
        uxx = self.B @ delta_u
        return ux, uxx

    def compute_switch_function(
        self, u: np.ndarray, ux: np.ndarray
    ) -> np.ndarray:
        """
        Computes the dispersion-damping switch function g_j^n (Equation 3.3 in paper):
            g_j^n = max{0, 1 + min{0, sign((u_x)_j^n * (u_x)_k^n)}}
        where k = j - sign(u_j^n).

        Parameters
        ----------
        u : np.ndarray
            Current solution values on the grid.
        ux : np.ndarray
            First spatial derivative on the grid.

        Returns
        -------
        g : np.ndarray
            Binary switch values (0 or 1) at each grid point.
        """
        sgn_u = np.sign(u).astype(int)
        j_indices = np.arange(self.m + 1)
        k_indices = np.clip(j_indices - sgn_u, 0, self.m)

        prod_sign = np.sign(ux * ux[k_indices])
        g = np.maximum(0.0, 1.0 + np.minimum(0.0, prod_sign))
        return g

    def step(self, u: np.ndarray) -> np.ndarray:
        """
        Performs one forward time step according to Equation (3.4):
            u_j^{n+1} = u_j^n - tau * u_j^n * (u_x)_j^n * g_j^n + (tau / R) * (u_xx)_j^n
        with Dirichlet boundary conditions u_0 = u_m = 0.

        Parameters
        ----------
        u : np.ndarray
            Solution at time step n.

        Returns
        -------
        u_next : np.ndarray
            Solution at time step n + 1.
        """
        ux, uxx = self.compute_spatial_derivatives(u)
        g = self.compute_switch_function(u, ux)

        u_next = u.copy()
        # Update interior points
        u_next[1 : self.m] = (
            u[1 : self.m]
            - self.tau * u[1 : self.m] * ux[1 : self.m] * g[1 : self.m]
            + (self.tau / self.R) * uxx[1 : self.m]
        )
        # Apply boundary conditions
        u_next[0] = 0.0
        u_next[self.m] = 0.0
        return u_next

    def solve(
        self,
        u0: Union[Callable[[np.ndarray], np.ndarray], np.ndarray],
        t_end: float,
        record_times: Optional[List[float]] = None,
    ) -> Tuple[np.ndarray, Dict[float, np.ndarray]]:
        """
        Integrates Burgers' equation from t = 0 to t = t_end and records
        the solution at requested time snapshots.

        Parameters
        ----------
        u0 : Callable or np.ndarray
            Initial condition function u0(x) or array of initial values on grid.
        t_end : float
            Final simulation time.
        record_times : list of float, optional
            List of target times to record the solution. If None, records
            the initial condition and the final solution.

        Returns
        -------
        x : np.ndarray
            Spatial coordinate array.
        solutions : dict
            Mapping from recorded time t -> solution array u(x, t).
        """
        if callable(u0):
            u = u0(self.x)
        else:
            u = np.array(u0, dtype=float, copy=True)

        # Enforce boundary condition on initial state
        u[0] = 0.0
        u[self.m] = 0.0

        if record_times is None:
            record_times = [0.0, float(t_end)]

        # Map target times to the closest integer step index
        step_to_time: Dict[int, float] = {}
        for t_target in record_times:
            step_idx = int(round(t_target / self.tau))
            step_to_time[step_idx] = t_target

        solutions: Dict[float, np.ndarray] = {}
        if 0 in step_to_time:
            solutions[step_to_time[0]] = u.copy()

        total_steps = int(round(t_end / self.tau))

        for step_idx in range(1, total_steps + 1):
            u = self.step(u)
            if step_idx in step_to_time:
                solutions[step_to_time[step_idx]] = u.copy()

        # Ensure all requested times are populated
        for t_target in record_times:
            if t_target not in solutions:
                step_idx = int(round(t_target / self.tau))
                if step_idx >= total_steps:
                    solutions[t_target] = u.copy()

        return self.x, solutions

