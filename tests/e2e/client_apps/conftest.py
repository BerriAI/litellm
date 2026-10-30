from __future__ import annotations

from collections.abc import Generator
from typing import Final

import pytest

from client_apps.compat_collector import (
    ROWS,
    CompatRecorder,
    cell_address,
    clear_shards,
    client_rows,
    known_clients,
    outcome_rows,
    publish,
    worker_id,
)


@pytest.fixture
def compat_result(request: pytest.FixtureRequest) -> CompatRecorder:
    item: Final = request.node  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]  # pytest: untyped
    assert isinstance(item, pytest.Item)
    address: Final = cell_address(item.path)
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
    address: Final = cell_address(item.path)
    if address is None:
        return report
    extra: Final = outcome_rows(address, item.stash.get(ROWS, ()), report)
    if extra:
        item.stash[ROWS] = (*item.stash.get(ROWS, ()), *extra)
    return report


def pytest_sessionstart(session: pytest.Session) -> None:
    if worker_id() is not None:
        return
    for client in known_clients():
        clear_shards(client)


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    for client in known_clients():
        publish(client, client_rows(session, client), worker_id())
