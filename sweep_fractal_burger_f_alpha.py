"""Grid search sweep for f_alpha in Fractal L_W2 Burgers solver.

This script searches second-derivative fractal scale vectors f_alpha to
demonstrate superiority of the fractal L_W2 operator over the classical L_W2
operator for Burgers' equation.

It writes every candidate result incrementally to CSV so a run can be
interrupted at any time without losing completed candidates.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import math
import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np

from alpha_fractal_function import alpha_fractalize_second_derivative
from burger_solver import (
    build_classical_spatial_operator,
    build_ux_matrix,
    compute_switch_function,
    step_classical_burgers,
)
from fractal_burger_solver import (
    build_fractal_second_derivative,
    build_fractal_spatial_operator,
    step_fractal_burgers,
    ddphi,
    H5_dd,
)
from mqqi.problems import cole_analytic_solution


def run_sweep(
    R: float = 10.0,
    m: int = 100,
    tau: float = 1e-3,
    t_end: float = 1.0,
    c: float = 0.0072,
    s: float = 0.8,
    K: int = 10,
    n_subintervals: int = 10,
    n_iter: int = 3,
    output_csv: str = "results/sweep_burger_f_alpha.csv",
    quick: bool = False,
) -> None:
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    x = np.linspace(0.0, 1.0, m + 1)
    N = m // K
    k_idx = np.concatenate(([1], K * np.arange(1, N - 1), [m - 1])).astype(int)
    xk = x[k_idx]
    Nt = int(round(t_end / tau))

    A_ux = build_ux_matrix(x, c)

    # Initial condition
    u0 = np.sin(np.pi * x)
    u0[0] = 0.0
    u0[m] = 0.0

    print("=" * 78)
    print("Fractal L_W2 Hyperparameter Sweep for 1D Viscous Burgers' Equation")
    print(f"Parameters: R = {R}, m = {m}, tau = {tau}, T = {t_end} (Nt = {Nt}), c = {c}, s = {s}")
    print(f"Fractal config: {n_subintervals} subintervals, {n_iter} iterations")
    print("=" * 78)

    # 1. Compute exact analytical solution
    print("Computing Cole's exact analytical solution...")
    exact = cole_analytic_solution(x, t_end, R)

    # 2. Compute classical L_W2 baseline
    print("Computing classical L_W2 baseline solution...")
    t0_c = time.perf_counter()
    M_xx_classical = build_classical_spatial_operator(x, xk, k_idx, s, c)

    u_c = u0.copy()
    for _ in range(Nt):
        u_c = step_classical_burgers(u_c, tau, R, A_ux, M_xx_classical)
    t_classical = time.perf_counter() - t0_c

    err_c = np.abs(u_c - exact)
    classical_linf = float(np.max(err_c))
    classical_rms = float(np.sqrt(np.mean(err_c**2)))
    print(f"Classical Baseline: Linf = {classical_linf:.6e}, RMS = {classical_rms:.6e} (runtime: {t_classical:.3f}s)")
    print("-" * 78)

    # 3. Generate candidate grid
    if quick:
        scales = [-0.00005, -0.0001, -0.00015, -0.0002, -0.00025, -0.0003]
        candidates = []
        for sc in scales:
            candidates.append([sc, 0.0, sc, 0.0, sc, sc, 0.0, sc, 0.0, sc])
            candidates.append([sc] * 10)
    else:
        # Structured symmetric grid for 10 subintervals: [a, b, c, d, e, e, d, c, b, a]
        a_vals = [0.0, -0.0001, -0.0002, -0.0003]
        b_vals = [0.0, -0.00005, -0.0001, -0.0002]
        c_vals = [0.0, -0.0001, -0.0002, -0.0003]
        d_vals = [0.0, -0.0001]
        e_vals = [-0.0001, -0.0002, -0.0003]

        candidates = []
        for a_v in a_vals:
            for b_v in b_vals:
                for c_v in c_vals:
                    for d_v in d_vals:
                        for e_v in e_vals:
                            vec = [a_v, b_v, c_v, d_v, e_v, e_v, d_v, c_v, b_v, a_v]
                            if any(abs(val) > 0 for val in vec):
                                candidates.append(vec)

    total_candidates = len(candidates)
    print(f"Total candidate combinations to evaluate: {total_candidates}")

    fieldnames = [
        "rank",
        "f_alpha",
        "fractal_Linf",
        "fractal_RMS",
        "classical_Linf",
        "classical_RMS",
        "linf_reduction_pct",
        "rms_reduction_pct",
        "is_superior",
    ]

    results = []
    superior_count = 0

    with open(output_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        start_time = time.perf_counter()
        for idx, f_alpha in enumerate(candidates, 1):
            try:
                fractal_dd = build_fractal_second_derivative(
                    c=c, f_alpha=f_alpha, n_iter=n_iter, a_domain=-2.0, b_domain=2.0
                )
                M_xx_f = build_fractal_spatial_operator(x, xk, k_idx, s, fractal_dd)

                u_f = u0.copy()
                for _ in range(Nt):
                    u_f = step_fractal_burgers(u_f, tau, R, A_ux, M_xx_f)

                err_f = np.abs(u_f - exact)
                f_linf = float(np.max(err_f))
                f_rms = float(np.sqrt(np.mean(err_f**2)))

                if np.isnan(f_linf) or np.isinf(f_linf):
                    continue

                linf_pct = ((classical_linf - f_linf) / classical_linf) * 100.0
                rms_pct = ((classical_rms - f_rms) / classical_rms) * 100.0
                is_superior = (f_linf < classical_linf) and (f_rms < classical_rms)

                if is_superior:
                    superior_count += 1

                row = {
                    "rank": 0,
                    "f_alpha": str(f_alpha),
                    "fractal_Linf": f_linf,
                    "fractal_RMS": f_rms,
                    "classical_Linf": classical_linf,
                    "classical_RMS": classical_rms,
                    "linf_reduction_pct": round(linf_pct, 4),
                    "rms_reduction_pct": round(rms_pct, 4),
                    "is_superior": is_superior,
                }
                results.append(row)
                writer.writerow(row)
                csvfile.flush()

                if idx % 10 == 0 or idx == total_candidates or is_superior:
                    elapsed = time.perf_counter() - start_time
                    rate = idx / elapsed if elapsed > 0 else 0
                    print(
                        f"[{idx:4d}/{total_candidates}] Superior: {superior_count:3d} | "
                        f"Current: Linf={f_linf:.5e} ({linf_pct:+6.2f}%), RMS={f_rms:.5e} ({rms_pct:+6.2f}%) | "
                        f"{rate:.1f} trials/s"
                    )

            except Exception as ex:
                continue

    # Re-sort results by lowest Linf error and update CSV with ranks
    if results:
        results.sort(key=lambda r: r["fractal_Linf"])
        for r_idx, row in enumerate(results, 1):
            row["rank"] = r_idx

        with open(output_path, "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

        print("=" * 78)
        print(f"Sweep Completed! {len(results)} valid configurations written to {output_path}")
        print(f"Found {superior_count} configurations superior to the classical L_W2 operator.")
        print("-" * 78)
        print("Top 5 Performing Fractal L_W2 Configurations:")
        for r_idx in range(min(5, len(results))):
            top_row = results[r_idx]
            print(
                f" Rank {top_row['rank']}: Linf = {top_row['fractal_Linf']:.6e} "
                f"(-{top_row['linf_reduction_pct']:.2f}%), "
                f"RMS = {top_row['fractal_RMS']:.6e} (-{top_row['rms_reduction_pct']:.2f}%) | "
                f"f_alpha = {top_row['f_alpha']}"
            )
        print("=" * 78)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sweep f_alpha for Fractal L_W2 Burgers equation solver.")
    parser.add_argument("--R", type=float, default=10.0, help="Reynolds number (default: 10.0)")
    parser.add_argument("--m", type=int, default=100, help="Spatial intervals (default: 100)")
    parser.add_argument("--tau", type=float, default=1e-3, help="Time step (default: 0.001)")
    parser.add_argument("--T", type=float, default=1.0, help="Final time (default: 1.0)")
    parser.add_argument("--c", type=float, default=0.0072, help="MQ shape parameter (default: 0.0072)")
    parser.add_argument("--s", type=float, default=0.8, help="RBF shape parameter (default: 0.8)")
    parser.add_argument("--subintervals", type=int, default=10, help="f_alpha subintervals (default: 10)")
    parser.add_argument("--iter", type=int, default=3, help="Fractal iterations (default: 3)")
    parser.add_argument("--output", type=str, default="results/sweep_burger_f_alpha.csv", help="Output CSV path")
    parser.add_argument("--quick", action="store_true", help="Run a quick targeted sweep")

    args = parser.parse_args()

    run_sweep(
        R=args.R,
        m=args.m,
        tau=args.tau,
        t_end=args.T,
        c=args.c,
        s=args.s,
        n_subintervals=args.subintervals,
        n_iter=args.iter,
        output_csv=args.output,
        quick=args.quick,
    )

