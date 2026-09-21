"""
Main execution script for MQQI Burgers' Equation experiments.
Runs the benchmark problems, verifies values against Tables 1-3,
and generates the figures as images in an 'output_figures/' directory.
"""

import os
import numpy as np
from mqqi import (
    MQQIBurgersSolver,
    get_problem_case,
    cole_analytic_solution,
    plot_paper_figure_1_to_3,
    plot_paper_figure_4,
    PROBLEM_CASES,
)
from mqqi.problems import (
    PAPER_TABLE_1_DATA,
    PAPER_TABLE_2_DATA,
    PAPER_TABLE_3_DATA,
)
import matplotlib.pyplot as plt


def run_all():
    output_dir = os.path.join(os.path.dirname(__file__), "output_figures")
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("MQQI Solver for Burgers' Equation (Chen & Wu, 2006)")
    print("=" * 70)

    # 1. Figure 1 / Table 1: R = 10
    print("\n[1/4] Running Case 1 (R = 10, h = 1/100, tau = 1e-3, lam = 0.0072)...")
    case1 = get_problem_case("fig1")
    solver1 = MQQIBurgersSolver(
        m=case1.m, tau=case1.tau, lam=case1.lam, R=case1.R
    )
    x1, sols1 = solver1.solve(case1.u0_func, case1.t_end, case1.record_times)
    fig1_path = os.path.join(output_dir, "figure_1_R10.png")
    plot_paper_figure_1_to_3(x1, sols1, R_value=case1.R, fig_num=1, save_path=fig1_path)
    plt.close()
    print(f"  -> Saved {fig1_path}")

    # Table 1 comparison
    u1_t1 = sols1[1.0]
    print("\n--- Table 1 Comparison (R = 10, t = 1.0) ---")
    print(f"{'x':>6} | {'Analytic':>10} | {'Hon & Mao':>10} | {'MQQI Paper':>12} | {'MQQI Computed':>14} | {'Rel Err (%)':>12}")
    print("-" * 75)
    for x_eval, (ana, hm, pap) in sorted(PAPER_TABLE_1_DATA.items()):
        idx = int(round(x_eval * case1.m))
        comp = u1_t1[idx]
        rel_err = abs(comp - ana) / ana * 100.0
        print(f"{x_eval:6.2f} | {ana:10.4f} | {hm:10.4f} | {pap:12.6f} | {comp:14.6f} | {rel_err:12.4f}")

    # 2. Figure 2 / Table 2: R = 100
    print("\n[2/4] Running Case 2 (R = 100, h = 1/100, tau = 1e-3, lam = 0.0029)...")
    case2 = get_problem_case("fig2")
    solver2 = MQQIBurgersSolver(
        m=case2.m, tau=case2.tau, lam=case2.lam, R=case2.R
    )
    x2, sols2 = solver2.solve(case2.u0_func, case2.t_end, case2.record_times)
    fig2_path = os.path.join(output_dir, "figure_2_R100.png")
    plot_paper_figure_1_to_3(x2, sols2, R_value=case2.R, fig_num=2, save_path=fig2_path)
    plt.close()
    print(f"  -> Saved {fig2_path}")

    # Table 2 comparison
    u2_t1 = sols2[1.0]
    print("\n--- Table 2 Comparison (R = 100, t = 1.0) ---")
    print(f"{'x':>6} | {'Analytic':>10} | {'Hon & Mao':>10} | {'MQQI Paper':>12} | {'MQQI Computed':>14} | {'Rel Err (%)':>12}")
    print("-" * 75)
    for x_eval, (ana, hm, pap) in sorted(PAPER_TABLE_2_DATA.items()):
        idx = int(round(x_eval * case2.m))
        comp = u2_t1[idx]
        rel_err = abs(comp - ana) / ana * 100.0
        print(f"{x_eval:6.2f} | {ana:10.4f} | {hm:10.4f} | {pap:12.6f} | {comp:14.6f} | {rel_err:12.4f}")

    # 3. Figure 3 / Table 3: R = 10000
    print("\n[3/4] Running Case 3 (R = 10000, h = 1/72, tau = 1e-3, lam = 1.43e-4)...")
    case3 = get_problem_case("fig3")
    solver3 = MQQIBurgersSolver(
        m=case3.m, tau=case3.tau, lam=case3.lam, R=case3.R
    )
    x3, sols3 = solver3.solve(case3.u0_func, case3.t_end, case3.record_times)
    fig3_path = os.path.join(output_dir, "figure_3_R10000.png")
    plot_paper_figure_1_to_3(x3, sols3, R_value=case3.R, fig_num=3, save_path=fig3_path)
    plt.close()
    print(f"  -> Saved {fig3_path}")

    # Table 3 comparison
    u3_t1 = sols3[1.0]
    print("\n--- Table 3 Comparison (R = 10000, t = 1.0) ---")
    print(f"{'x':>8} | {'Christie':>10} | {'Hon & Mao':>10} | {'MQQI Paper':>12} | {'MQQI Computed':>14}")
    print("-" * 65)
    for x_eval, (chr_val, hm, pap) in sorted(PAPER_TABLE_3_DATA.items()):
        idx = int(round(x_eval * case3.m))
        comp = u3_t1[idx]
        print(f"{x_eval:8.3f} | {chr_val:10.4f} | {hm:10.4f} | {pap:12.6f} | {comp:14.6f}")

    # 4. Figure 4: Shock wave development, R = 10000
    print("\n[4/4] Running Case 4 (Shock wave: R = 10000, h = 1/80, tau = 1e-3, lam = 1.2e-4)...")
    case4 = get_problem_case("fig4")
    solver4 = MQQIBurgersSolver(
        m=case4.m, tau=case4.tau, lam=case4.lam, R=case4.R
    )
    x4, sols4 = solver4.solve(case4.u0_func, case4.t_end, case4.record_times)
    fig4_path = os.path.join(output_dir, "figure_4_shock.png")
    plot_paper_figure_4(x4, sols4, fig_num=4, save_path=fig4_path)
    plt.close()
    print(f"  -> Saved {fig4_path}")

    print("\nAll experiments successfully completed!")


if __name__ == "__main__":
    run_all()

