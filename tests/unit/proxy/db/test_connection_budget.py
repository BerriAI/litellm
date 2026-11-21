import os
from typing import Final

import pytest

from litellm.proxy.db.connection_budget import (
    MIN_CONNECTION_LIMIT,
    RUST_READER_CONNECTIONS_ENV_VAR,
    RUST_RESERVED_CONNECTIONS,
    RUST_WRITER_CONNECTIONS_ENV_VAR,
    SPEND_LOG_WRITER_RESERVED_CONNECTIONS,
    ConnectionBudget,
    RustConnections,
    UnsplittableConnectionLimit,
    rust_connections,
    rust_writer_reserve,
    with_configured_connection_budget,
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


def test_the_ledger_reserves_two_more_writer_connections_than_the_base() -> None:
    assert SPEND_LOG_WRITER_RESERVED_CONNECTIONS == 2
    assert rust_writer_reserve(True) == RUST_RESERVED_CONNECTIONS + 2


def test_without_the_ledger_the_base_writer_reserve_stands() -> None:
    assert rust_writer_reserve(False) == RUST_RESERVED_CONNECTIONS


@pytest.mark.parametrize(
    ("total", "python", "rust"),
    [(2, 1, 1), (4, 1, 3), (5, 1, 4), (10, 6, 4)],
    ids=["two_connections", "four_connections", "five_connections", "ten_connections"],
)
def test_the_ledger_writer_reserve_is_carved_out_of_the_limit(total: int, python: int, rust: int) -> None:
    budget: Final = ConnectionBudget.of("DATABASE_URL", total, rust_writer_reserve(True))

    assert (budget.python, budget.rust) == (python, rust)


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
    assert rust_connections(environ, rust_spend_log_writer=False) == expected


@pytest.mark.parametrize(
    ("environ", "expected"),
    [
        ({RUST_WRITER_CONNECTIONS_ENV_VAR: "3", RUST_READER_CONNECTIONS_ENV_VAR: "1"}, RustConnections(3, 1)),
        ({}, RustConnections(RUST_RESERVED_CONNECTIONS + 2, RUST_RESERVED_CONNECTIONS)),
    ],
    ids=["cli_split_both", "started_without_cli"],
)
def test_with_the_ledger_a_worker_without_the_cli_split_reserves_the_writer_extra(
    environ: dict[str, str], expected: RustConnections
) -> None:
    assert rust_connections(environ, rust_spend_log_writer=True) == expected


@pytest.mark.parametrize(
    ("yaml_limit", "rust_spend_log_writer", "writer_query", "reader_query", "connections"),
    [
        (2, False, "schema=app&connection_limit=1", "connection_limit=1", RustConnections(1, 1)),
        (10, False, "schema=app&connection_limit=8", "connection_limit=3", RustConnections(2, 2)),
        (None, False, "schema=app", "connection_limit=3", RustConnections(4, 2)),
        (10, True, "schema=app&connection_limit=6", "connection_limit=3", RustConnections(4, 2)),
        (2, True, "schema=app&connection_limit=1", "connection_limit=1", RustConnections(1, 1)),
    ],
)
def test_configured_budget_caps_prisma_and_reserves_rust_until_lifespan_exit(
    monkeypatch: pytest.MonkeyPatch,
    yaml_limit: int | None,
    rust_spend_log_writer: bool,
    writer_query: str,
    reader_query: str,
    connections: RustConnections,
) -> None:
    writer: Final = "postgresql://db/app?schema=app"
    reader: Final = "postgresql://reader/app?connection_limit=3"
    monkeypatch.setenv("DATABASE_URL", writer)
    monkeypatch.setenv("DATABASE_URL_READ_REPLICA", reader)
    monkeypatch.setenv(RUST_WRITER_CONNECTIONS_ENV_VAR, "4")
    monkeypatch.delenv(RUST_READER_CONNECTIONS_ENV_VAR, raising=False)

    with with_configured_connection_budget(yaml_limit, rust_spend_log_writer=rust_spend_log_writer):
        assert os.environ["DATABASE_URL"] == f"postgresql://db/app?{writer_query}"
        assert os.environ["DATABASE_URL_READ_REPLICA"] == f"postgresql://reader/app?{reader_query}"
        assert rust_connections(os.environ, rust_spend_log_writer=rust_spend_log_writer) == connections

    assert os.environ["DATABASE_URL"] == writer
    assert os.environ["DATABASE_URL_READ_REPLICA"] == reader
    assert rust_connections(os.environ, rust_spend_log_writer=rust_spend_log_writer) == RustConnections(4, 2)
    assert RUST_READER_CONNECTIONS_ENV_VAR not in os.environ
