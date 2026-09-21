"""Shared visualization for the two-good consumer problem."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from models import DemandResult, utility_grid


def make_plot(
    model: str,
    px: float,
    py: float,
    income: float,
    parameters: dict,
    result: DemandResult,
):
    x_intercept = income / px
    y_intercept = income / py
    x_max = max(1.15 * x_intercept, 1.0)
    y_max = max(1.15 * y_intercept, 1.0)

    # Avoid zero in models with logs or negative CES powers.
    x_values = np.linspace(max(x_max * 1e-4, 1e-7), x_max, 350)
    y_values = np.linspace(max(y_max * 1e-4, 1e-7), y_max, 350)
    xx, yy = np.meshgrid(x_values, y_values)
    utilities = utility_grid(model, xx, yy, parameters)
    utilities = np.where(np.isfinite(utilities), utilities, np.nan)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    budget_x = np.linspace(0, x_intercept, 300) if income > 0 else np.array([0.0])
    budget_y = (income - px * budget_x) / py
    ax.fill_between(budget_x, 0, budget_y, color="#dbeafe", alpha=0.55, label="Feasible set")
    ax.plot(budget_x, budget_y, color="#2563eb", linewidth=2.2, label="Budget line")

    finite = utilities[np.isfinite(utilities)]
    if finite.size and np.nanmax(finite) > np.nanmin(finite):
        quantiles = np.nanquantile(finite, [0.2, 0.4, 0.6, 0.8])
        levels = np.unique(np.append(quantiles, result.utility))
        levels = levels[np.isfinite(levels)]
        if len(levels) > 1:
            ax.contour(xx, yy, utilities, levels=levels, colors="#7c3aed", alpha=0.55, linewidths=1)

    if result.solution_type == "multiple optima":
        ax.plot(budget_x, budget_y, color="#dc2626", linewidth=5, alpha=0.55, label="Optimal bundles")
    elif result.solution_type == "two corner optima":
        ax.scatter(
            [x_intercept, 0], [0, y_intercept], s=100, color="#dc2626",
            edgecolor="white", linewidth=1.5, zorder=5, label="Two optimal corners"
        )
    else:
        ax.scatter([result.x], [result.y], s=100, color="#dc2626", edgecolor="white",
                   linewidth=1.5, zorder=5, label=f"Optimum ({result.x:.2f}, {result.y:.2f})")

    ax.set(xlim=(0, x_max), ylim=(0, y_max), xlabel="Quantity of good x", ylabel="Quantity of good y")
    ax.set_title(f"{model}: budget and preferences")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.18)
    ax.legend(loc="best")
    fig.tight_layout()
    return fig
