import math

import pytest

from models import (
    solve_addictive,
    solve_cobb_douglas,
    solve_complements,
    solve_neutral,
    solve_quasilinear,
    solve_substitutes,
)


def test_cobb_douglas():
    result = solve_cobb_douglas(2, 4, 100, 0.6)
    assert result.x == pytest.approx(30)
    assert result.y == pytest.approx(10)
    assert result.spending == pytest.approx(100)


def test_addictive_selects_best_corner():
    result = solve_addictive(1, 2, 20)
    assert result.x == pytest.approx(20)
    assert result.y == 0
    assert result.utility == pytest.approx(400)


def test_addictive_equal_prices_has_two_corner_optima():
    result = solve_addictive(2, 2, 20)
    assert result.solution_type == "two corner optima"
    assert result.optimal_set == "Either (10, 0) or (0, 10)"


def test_neutral_good():
    result = solve_neutral(2, 5, 20, "y")
    assert (result.x, result.y) == (10, 0)


def test_substitutes_corner_and_tie():
    assert solve_substitutes(2, 4, 20, 2, 1).x == pytest.approx(10)
    assert solve_substitutes(2, 4, 20, 1, 2).solution_type == "multiple optima"


def test_complements():
    result = solve_complements(2, 3, 40, 1, 2)
    assert result.x == pytest.approx(5)
    assert result.y == pytest.approx(10)


def test_quasilinear_interior_and_corner():
    interior = solve_quasilinear(2, 1, 20, 4)
    assert interior.x == pytest.approx(2)
    assert interior.y == pytest.approx(16)
    corner = solve_quasilinear(2, 1, 2, 4)
    assert corner.x == pytest.approx(1)
    assert corner.y == 0


def test_invalid_price():
    with pytest.raises(ValueError):
        solve_cobb_douglas(0, 2, 10, 0.5)
