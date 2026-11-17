from typing import Final

import pytest

from litellm.proxy.db.connection_budget import ConnectionBudget


@pytest.mark.parametrize("total", range(1, 41))
def test_default_split_leaves_prisma_the_whole_total(total: int) -> None:
    budget: Final = ConnectionBudget.split(total)

    assert (budget.python, budget.rust) == (total, 0)


@pytest.mark.parametrize("total", range(3, 41))
def test_a_reserve_comes_out_of_the_total(total: int) -> None:
    budget: Final = ConnectionBudget.split(total, rust_reserved=2)

    assert (budget.python + budget.rust, budget.rust) == (total, 2)


@pytest.mark.parametrize(("total", "python"), [(2, 1), (1, 1)], ids=["reserve_equals_total", "single_connection"])
def test_prisma_keeps_one_connection_when_the_reserve_does_not_fit(total: int, python: int) -> None:
    budget: Final = ConnectionBudget.split(total, rust_reserved=2)

    assert (budget.python, budget.rust) == (python, 2)


@pytest.mark.parametrize("connection_limit", ["7", 7])
def test_a_positive_integer_limit_is_split(connection_limit: str | int) -> None:
    assert ConnectionBudget.of(connection_limit, rust_reserved=2) == ConnectionBudget.split(7, rust_reserved=2)


@pytest.mark.parametrize("connection_limit", ["0", 0, "-1", -1, "abc", "10.0", 10.5, "", "\u00b2"])
def test_a_limit_that_is_not_a_positive_integer_is_left_to_prisma(connection_limit: str | float) -> None:
    assert ConnectionBudget.of(connection_limit, rust_reserved=2) is None


def test_prisma_params_change_only_beside_a_rust_pool() -> None:
    params: Final = {"connection_limit": "010", "pool_timeout": 30}

    assert ConnectionBudget.split(10).prisma_params(params) == params
    assert ConnectionBudget.split(10, rust_reserved=2).prisma_params(params) == {
        "connection_limit": 8,
        "pool_timeout": 30,
    }
