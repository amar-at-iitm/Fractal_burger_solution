"""
Plotting utilities matching the visual presentation in Chen & Wu (2006).
Generates reproductions of Figures 1, 2, 3, and 4 from the paper using Matplotlib.
"""

from typing import Dict, Optional
import matplotlib.pyplot as plt
import numpy as np


def plot_paper_figure_1_to_3(
    x: np.ndarray,
    solutions: Dict[float, np.ndarray],
    R_value: float,
    fig_num: int = 1,
    ax: Optional[plt.Axes] = None,
    save_path: Optional[str] = None,
) -> plt.Axes:
    """
    Plots the solution evolution for Figures 1, 2, or 3 (R = 10, 100, or 10000).

    Style matches the paper:
        t = 0.0: solid blue
        t = 0.2: dashed blue
        t = 0.4: dash-dot blue
        t = 0.6: solid red
        t = 0.8: dashed red
        t = 1.0: dash-dot red

    Parameters
    ----------
    x : np.ndarray
        Spatial grid coordinates.
    solutions : dict
        Mapping of recorded time t -> solution array u(x, t).
    R_value : float
        Reynolds number (10, 100, or 10000).
    fig_num : int, default=1
        Paper figure number (1, 2, or 3).
    ax : plt.Axes, optional
        Existing axes to plot on. If None, a new figure is created.
    save_path : str, optional
        Path to save the generated figure.

    Returns
    -------
    ax : plt.Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 6), dpi=150)

    # Style mapping corresponding to paper Figs 1-3
    style_specs = {
        0.0: {"color": "blue", "linestyle": "-", "label": "t=0.0"},
        0.2: {"color": "blue", "linestyle": "--", "label": "t=0.2"},
        0.4: {"color": "blue", "linestyle": "-.", "label": "t=0.4"},
        0.6: {"color": "red", "linestyle": "-", "label": "t=0.6"},
        0.8: {"color": "red", "linestyle": "--", "label": "t=0.8"},
        1.0: {"color": "red", "linestyle": "-.", "label": "t=1.0"},
    }

    # Plot curves
    for t_val, spec in style_specs.items():
        # Match nearest float key if needed
        matching_key = None
        for k in solutions.keys():
            if abs(k - t_val) < 1e-4:
                matching_key = k
                break

        if matching_key is not None:
            u_t = solutions[matching_key]
            ax.plot(
                x,
                u_t,
                color=spec["color"],
                linestyle=spec["linestyle"],
                linewidth=1.8,
                label=spec["label"],
            )

    # Add R indicator text inside top-left of plot (as in paper)
    r_str = f"R={int(R_value)}" if R_value.is_integer() else f"R={R_value}"
    ax.text(
        0.18,
        0.93,
        r_str,
        fontsize=13,
        fontweight="bold",
        transform=ax.transAxes,
    )

    # Text annotations on curves at approximate peak locations
    annotation_positions = {
        10: {
            0.0: (0.38, 0.87),
            0.2: (0.48, 0.65),
            0.4: (0.50, 0.50),
            0.6: (0.57, 0.42),
            0.8: (0.61, 0.31),
            1.0: (0.63, 0.23),
        },
        100: {
            0.0: (0.42, 0.88),
            0.2: (0.50, 0.76),
            0.4: (0.54, 0.62),
            0.6: (0.59, 0.48),
            0.8: (0.64, 0.38),
            1.0: (0.65, 0.30),
        },
        10000: {
            0.0: (0.43, 0.88),
            0.2: (0.52, 0.76),
            0.4: (0.56, 0.62),
            0.6: (0.60, 0.48),
            0.8: (0.61, 0.39),
            1.0: (0.62, 0.30),
        },
    }

    r_key = int(R_value) if int(R_value) in annotation_positions else 10
    pos_map = annotation_positions.get(r_key, {})
    for t_val, pos in pos_map.items():
        ax.text(
            pos[0],
            pos[1],
            f"t={t_val:.1f}",
            fontsize=10.5,
            fontweight="normal",
        )

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.set_xticks(np.arange(0.0, 1.05, 0.1))
    ax.set_yticks(np.arange(0.0, 1.05, 0.1))
    ax.set_xlabel("x", fontsize=11)
    ax.set_ylabel("u(x, t)", fontsize=11)
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.set_title(
        f"Fig. {fig_num}. MQQI solution for Burgers' equation (R = {int(R_value)})",
        fontsize=12,
        pad=10,
    )

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)

    return ax


def plot_paper_figure_4(
    x: np.ndarray,
    solutions: Dict[float, np.ndarray],
    fig_num: int = 4,
    ax: Optional[plt.Axes] = None,
    save_path: Optional[str] = None,
) -> plt.Axes:
    """
    Plots the solution evolution for Figure 4 (Shock wave, R = 10000).

    Style matches the paper:
        t = 0.2: blue with circle 'o'
        t = 0.6: green with diamond 'd'
        t = 1.0: red with plus '+'
        t = 1.4: magenta with triangle '^'
        t = 2.0: cyan with star '*'

    Parameters
    ----------
    x : np.ndarray
        Spatial grid coordinates.
    solutions : dict
        Mapping of recorded time t -> solution array u(x, t).
    fig_num : int, default=4
        Paper figure number.
    ax : plt.Axes, optional
        Existing axes to plot on. If None, a new figure is created.
    save_path : str, optional
        Path to save the generated figure.

    Returns
    -------
    ax : plt.Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(7.5, 6), dpi=150)

    style_specs = {
        0.2: {"color": "blue", "marker": "o", "label": "t=0.2", "markevery": 1},
        0.6: {"color": "limegreen", "marker": "d", "label": "t=0.6", "markevery": 1},
        1.0: {"color": "red", "marker": "+", "label": "t=1.0", "markevery": 1},
        1.4: {"color": "magenta", "marker": "^", "label": "t=1.4", "markevery": 1},
        2.0: {"color": "cyan", "marker": "*", "label": "t=2.0", "markevery": 1},
    }

    for t_val, spec in style_specs.items():
        matching_key = None
        for k in solutions.keys():
            if abs(k - t_val) < 1e-4:
                matching_key = k
                break

        if matching_key is not None:
            u_t = solutions[matching_key]
            ax.plot(
                x,
                u_t,
                color=spec["color"],
                marker=spec["marker"],
                markevery=spec["markevery"],
                markersize=5,
                markerfacecolor="none",
                markeredgewidth=1.2,
                linewidth=1.2,
                label=spec["label"],
            )

    # Curve text annotations near the wave crests
    text_positions = {
        0.2: (0.52, 1.34),
        0.6: (0.61, 0.95),
        1.0: (0.71, 0.70),
        1.4: (0.80, 0.58),
        2.0: (0.89, 0.48),
    }

    for t_val, pos in text_positions.items():
        ax.text(
            pos[0],
            pos[1],
            f"t={t_val:.1f}",
            fontsize=10.5,
            fontweight="normal",
        )

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(-1.0, 1.52)
    ax.set_xticks(np.arange(0.0, 1.05, 0.1))
    ax.set_yticks(np.arange(-1.0, 1.55, 0.5))
    ax.set_xlabel("x", fontsize=11)
    ax.set_ylabel("u(x, t)", fontsize=11)
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.set_title(
        "Fig. 4. Shock wave development for R = 10000 (MQQI scheme)",
        fontsize=12,
        pad=10,
    )

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)

    return ax


def plot_comparison_with_analytic(
    x: np.ndarray,
    u_computed: np.ndarray,
    u_exact: np.ndarray,
    title: str = "Comparison with Analytic Solution",
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Plots computed MQQI solution vs analytical benchmark solution at t = 1.0.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5), dpi=150)

    ax.plot(x, u_exact, "k-", linewidth=2.0, label="Exact (Cole, 1951)")
    ax.plot(
        x,
        u_computed,
        "ro",
        markersize=4,
        markerfacecolor="none",
        label="MQQI Scheme",
    )
    ax.set_xlabel("x", fontsize=11)
    ax.set_ylabel("u(x, 1.0)", fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(frameon=True)
    return ax

