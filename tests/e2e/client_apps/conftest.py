from __future__ import annotations

from collections.abc import Generator
from pathlib import Path
from typing import Final

import pytest

from client_apps.compat_collector import (
    ROWS,
    CompatRecorder,
    cell_address,
    clear_shards,
    client_rows,
    known_clients,
    known_gap_marks,
    outcome_rows,
    publish,
    worker_id,
)

CLIENTS_DIR: Final = Path(__file__).resolve().parent


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    for item, mark in known_gap_marks(items, CLIENTS_DIR):
        item.add_marker(mark)


@pytest.fixture
def compat_result(request: pytest.FixtureRequest) -> CompatRecorder:
    item: Final = request.node  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]  # pytest: untyped
    assert isinstance(item, pytest.Item)
    address: Final = cell_address(item.path, CLIENTS_DIR)
    if address is None:
        pytest.fail(f"{item.path} is not a <client>/<feature_id>/test_<provider>.py cell of a client manifest")
    return CompatRecorder(item, address)


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item: pytest.Item) -> None:
    item.stash[ROWS] = ()


@pytest.hookimpl(wrapper=True)
def pytest_runtest_makereport(
    item: pytest.Item, call: pytest.CallInfo[None]
) -> Generator[None, pytest.TestReport, pytest.TestReport]:
    report: Final = yield
    address: Final = cell_address(item.path, CLIENTS_DIR)
    if address is None:
        return report
    extra: Final = outcome_rows(address, item.stash.get(ROWS, ()), call, report)
    if extra:
        item.stash[ROWS] = (*item.stash.get(ROWS, ()), *extra)
    return report


def pytest_sessionstart(session: pytest.Session) -> None:
    if worker_id() is not None:
        return
    for client in known_clients(CLIENTS_DIR):
        clear_shards(client)


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    for client in known_clients(CLIENTS_DIR):
        publish(client, client_rows(session, client, CLIENTS_DIR), worker_id())
