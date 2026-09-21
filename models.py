"""Analytical consumer-demand models for two goods."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class DemandResult:
    x: float
    y: float
    utility: float
    spending: float
    solution_type: str
    explanation: str
    optimal_set: str | None = None


def _validate_budget(px: float, py: float, income: float) -> None:
    if px <= 0 or py <= 0:
        raise ValueError("Both prices must be strictly positive.")
    if income < 0:
        raise ValueError("Income cannot be negative.")


def _result(
    x: float,
    y: float,
    utility: float,
    px: float,
    py: float,
    solution_type: str,
    explanation: str,
    optimal_set: str | None = None,
) -> DemandResult:
    return DemandResult(
        x=float(x),
        y=float(y),
        utility=float(utility),
        spending=float(px * x + py * y),
        solution_type=solution_type,
        explanation=explanation,
        optimal_set=optimal_set,
    )


def solve_cobb_douglas(px: float, py: float, income: float, alpha: float) -> DemandResult:
    _validate_budget(px, py, income)
    if not 0 < alpha < 1:
        raise ValueError("Cobb–Douglas alpha must lie strictly between 0 and 1.")
    x = alpha * income / px
    y = (1 - alpha) * income / py
    utility = x**alpha * y ** (1 - alpha) if income > 0 else 0.0
    return _result(x, y, utility, px, py, "interior" if income > 0 else "boundary",
                   f"The consumer spends {alpha:.0%} of income on x and {1-alpha:.0%} on y.")


def solve_addictive(px: float, py: float, income: float) -> DemandResult:
    """Solve u(x1, x2) = x1^2 + x2^2 on the budget set.

    Utility is convex, so maximizing it over the budget triangle gives a corner
    solution.  The MRS equality identifies a minimum along the budget line.
    """
    _validate_budget(px, py, income)
    x_corner = income / px
    y_corner = income / py
    utility_x = x_corner**2
    utility_y = y_corner**2

    if np.isclose(utility_x, utility_y, rtol=1e-9, atol=1e-12):
        optimal_set = f"Either ({x_corner:g}, 0) or (0, {y_corner:g})"
        return _result(
            x_corner,
            0.0,
            utility_x,
            px,
            py,
            "two corner optima",
            "The two corners give equal utility. Interior mixtures give lower utility.",
            optimal_set,
        )
    if utility_x > utility_y:
        return _result(
            x_corner,
            0.0,
            utility_x,
            px,
            py,
            "corner",
            "Good 1 gives the higher corner utility, so all income is spent on good 1.",
        )
    return _result(
        0.0,
        y_corner,
        utility_y,
        px,
        py,
        "corner",
        "Good 2 gives the higher corner utility, so all income is spent on good 2.",
    )


def solve_neutral(px: float, py: float, income: float, neutral_good: str) -> DemandResult:
    _validate_budget(px, py, income)
    if neutral_good == "y":
        x, y, utility = income / px, 0.0, income / px
        explanation = "Only x raises utility, so all income is spent on x."
    elif neutral_good == "x":
        x, y, utility = 0.0, income / py, income / py
        explanation = "Only y raises utility, so all income is spent on y."
    else:
        raise ValueError("neutral_good must be either 'x' or 'y'.")
    return _result(x, y, utility, px, py, "corner", explanation)


def solve_substitutes(px: float, py: float, income: float, a: float, b: float) -> DemandResult:
    _validate_budget(px, py, income)
    if a <= 0 or b <= 0:
        raise ValueError("Utility weights a and b must be positive.")
    value_x, value_y = a / px, b / py
    if np.isclose(value_x, value_y, rtol=1e-9, atol=1e-12):
        x, y = income / px, 0.0
        optimal_set = f"Every bundle on {px:g}x + {py:g}y = {income:g}, with x,y >= 0"
        return _result(x, y, a * x + b * y, px, py, "multiple optima",
                       "Both goods provide the same marginal utility per unit of money.", optimal_set)
    if value_x > value_y:
        x, y = income / px, 0.0
        explanation = "x provides more utility per unit of money, so the optimum is the x-axis corner."
    else:
        x, y = 0.0, income / py
        explanation = "y provides more utility per unit of money, so the optimum is the y-axis corner."
    return _result(x, y, a * x + b * y, px, py, "corner", explanation)


def solve_complements(px: float, py: float, income: float, a: float, b: float) -> DemandResult:
    _validate_budget(px, py, income)
    if a <= 0 or b <= 0:
        raise ValueError("Fixed-proportion parameters a and b must be positive.")
    scale = income / (px * a + py * b)
    x, y = a * scale, b * scale
    return _result(x, y, min(x / a, y / b), px, py, "kink",
                   f"The goods are consumed in the fixed proportion x:y = {a:g}:{b:g}.")


def solve_quasilinear(px: float, py: float, income: float, alpha: float) -> DemandResult:
    _validate_budget(px, py, income)
    if alpha <= 0:
        raise ValueError("Quasi-linear alpha must be positive.")
    if income <= 0:
        raise ValueError("Income must be positive because ln(x) is undefined at x = 0.")
    interior_x = alpha * py / px
    if px * interior_x <= income:
        x = interior_x
        y = (income - px * x) / py
        solution_type = "interior" if y > 1e-12 else "boundary"
        explanation = "Income is sufficient to reach the preferred x quantity; remaining income is spent on y."
    else:
        x, y = income / px, 0.0
        solution_type = "corner"
        explanation = "Income is too low to reach the interior candidate, so all income is spent on x."
    utility = alpha * np.log(x) + y
    return _result(x, y, utility, px, py, solution_type, explanation)


SOLVERS: dict[str, Callable[..., DemandResult]] = {
    "Cobb–Douglas": solve_cobb_douglas,
    "Addictive (non-convex preferences)": solve_addictive,
    "Neutral good": solve_neutral,
    "Perfect substitutes": solve_substitutes,
    "Perfect complements": solve_complements,
    "Quasi-linear": solve_quasilinear,
}


def utility_grid(model: str, x: np.ndarray, y: np.ndarray, parameters: dict) -> np.ndarray:
    """Evaluate utility over arrays for contour plotting."""
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        if model == "Cobb–Douglas":
            alpha = parameters["alpha"]
            return x**alpha * y ** (1 - alpha)
        if model == "Addictive (non-convex preferences)":
            return x**2 + y**2
        if model == "Neutral good":
            return x if parameters["neutral_good"] == "y" else y
        if model == "Perfect substitutes":
            return parameters["a"] * x + parameters["b"] * y
        if model == "Perfect complements":
            return np.minimum(x / parameters["a"], y / parameters["b"])
        if model == "Quasi-linear":
            return parameters["alpha"] * np.log(x) + y
    raise ValueError(f"Unknown model: {model}")
