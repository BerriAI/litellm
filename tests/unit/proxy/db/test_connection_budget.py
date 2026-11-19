from typing import Final

import pytest

from litellm.proxy.db.connection_budget import (
    MIN_CONNECTION_LIMIT,
    RUST_READER_CONNECTIONS_ENV_VAR,
    RUST_RESERVED_CONNECTIONS,
    RUST_WRITER_CONNECTIONS_ENV_VAR,
    ConnectionBudget,
    RustConnections,
    UnsplittableConnectionLimit,
    rust_connections,
)


@pytest.mark.parametrize("total", range(MIN_CONNECTION_LIMIT, 41))
def test_prisma_and_rust_never_open_more_than_the_total(total: int) -> None:
    budget: Final = ConnectionBudget.of("DATABASE_URL", total)

    assert budget.python + budget.rust == total
    assert budget.python >= 1
    assert budget.rust >= 1


@pytest.mark.parametrize(
    ("total", "python", "rust"),
    [(2, 1, 1), (3, 1, 2), (4, 2, 2), (100, 98, 2)],
    ids=["two_connections", "three_connections", "four_connections", "large"],
)
def test_rust_takes_what_is_left_of_its_reserve_after_prisma_keeps_one(total: int, python: int, rust: int) -> None:
    budget: Final = ConnectionBudget.split(total, rust_reserved=2)

    assert (budget.python, budget.rust) == (python, rust)


@pytest.mark.parametrize("connection_limit", ["7", 7, 7.0])
def test_a_whole_number_limit_is_split(connection_limit: str | float) -> None:
    assert ConnectionBudget.of("DATABASE_URL", connection_limit, rust_reserved=2) == ConnectionBudget.split(
        7, rust_reserved=2
    )


@pytest.mark.parametrize("connection_limit", ["1", 1, "0", 0, "-1", -1, "abc", "10.0", 10.5, "", "²", True])
def test_a_limit_that_leaves_no_room_for_both_pools_is_refused(connection_limit: str | float) -> None:
    with pytest.raises(UnsplittableConnectionLimit) as refused:
        ConnectionBudget.of("DATABASE_URL_READ_REPLICA", connection_limit)

    assert str(refused.value) == (
        f"DATABASE_URL_READ_REPLICA connection_limit {connection_limit!r} must be a whole number of at least 2, "
        "one connection for Prisma and at least one for the Rust pool that reads keys for auth"
    )


def test_prisma_params_carry_prisma_share() -> None:
    params: Final = {"connection_limit": "010", "pool_timeout": 30}

    assert ConnectionBudget.split(10).prisma_params(params) == {"connection_limit": 8, "pool_timeout": 30}


@pytest.mark.parametrize(
    ("environ", "expected"),
    [
        ({RUST_WRITER_CONNECTIONS_ENV_VAR: "2", RUST_READER_CONNECTIONS_ENV_VAR: "1"}, RustConnections(2, 1)),
        ({RUST_WRITER_CONNECTIONS_ENV_VAR: "1"}, RustConnections(1, RUST_RESERVED_CONNECTIONS)),
        ({}, RustConnections(RUST_RESERVED_CONNECTIONS, RUST_RESERVED_CONNECTIONS)),
        (
            {RUST_WRITER_CONNECTIONS_ENV_VAR: "-1", RUST_READER_CONNECTIONS_ENV_VAR: "two"},
            RustConnections(RUST_RESERVED_CONNECTIONS, RUST_RESERVED_CONNECTIONS),
        ),
    ],
    ids=["cli_split_both", "cli_split_writer_only", "started_without_cli", "not_a_number"],
)
def test_a_worker_sizes_each_rust_pool_from_the_cli_split(environ: dict[str, str], expected: RustConnections) -> None:
    assert rust_connections(environ) == expected
