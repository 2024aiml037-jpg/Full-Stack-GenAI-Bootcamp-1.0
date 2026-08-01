import pytest


@pytest.fixture()
def calc(calculator_module):
    return calculator_module.Calculator()


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [(5, 3, 8), (-2, 2, 0), (-4, -6, -10), (2.5, 0.5, 3.0)],
)
def test_add_numbers(calc, a, b, expected):
    assert calc.add_numbers(a, b) == expected


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [(5, 3, 2), (3, 5, -2), (-4, -6, 2), (2.5, 0.5, 2.0)],
)
def test_subtract_numbers(calc, a, b, expected):
    assert calc.subtract_numbers(a, b) == expected


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [(5, 3, 15), (-4, 3, -12), (0, 99, 0), (1.5, 2, 3.0)],
)
def test_multiply_numbers(calc, a, b, expected):
    assert calc.multiply_numbers(a, b) == expected


@pytest.mark.parametrize(("a", "b", "expected"), [(6, 3, 2.0), (5, 2, 2.5), (-9, 3, -3.0)])
def test_divide_numbers(calc, a, b, expected):
    assert calc.divide_numbers(a, b) == pytest.approx(expected)


def test_divide_by_zero_returns_error_message(calc):
    assert calc.divide_numbers(1, 0) == "Error: Division by zero"


def test_operations_are_not_type_restricted(calc):
    assert calc.add_numbers("a", "b") == "ab"
    assert calc.multiply_numbers("ab", 2) == "abab"


def test_module_level_demo_values(calculator_module):
    assert calculator_module.result_add == 8
    assert calculator_module.result_sub == 2
    assert calculator_module.result_mul == 15
    assert calculator_module.result_div == pytest.approx(5 / 3)
