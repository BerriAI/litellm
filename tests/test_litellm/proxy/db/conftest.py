import os
from collections.abc import Generator
from typing import Optional

import pytest

DB_ENV_KEYS = (
    "IAM_TOKEN_DB_AUTH",
    "DATABASE_URL",
    "DIRECT_URL",
    "DATABASE_URL_READ_REPLICA",
    "DATABASE_HOST",
    "DATABASE_PORT",
    "DATABASE_USER",
    "DATABASE_USERNAME",
    "DATABASE_NAME",
    "DATABASE_SCHEMA",
    "DATABASE_PASSWORD",
    "DATABASE_HOST_READ_REPLICA",
    "DATABASE_PORT_READ_REPLICA",
    "DATABASE_USER_READ_REPLICA",
    "DATABASE_USERNAME_READ_REPLICA",
    "DATABASE_NAME_READ_REPLICA",
    "DATABASE_SCHEMA_READ_REPLICA",
    "DATABASE_PASSWORD_READ_REPLICA",
)

_db_env_snapshot_key = pytest.StashKey[dict[str, Optional[str]]]()


def _db_env_snapshot() -> dict[str, Optional[str]]:
    return {key: os.environ.get(key) for key in DB_ENV_KEYS}


@pytest.hookimpl(wrapper=True)
def pytest_runtest_setup(item: pytest.Item) -> Generator[None, None, None]:
    item.stash[_db_env_snapshot_key] = _db_env_snapshot()
    return (yield)


@pytest.hookimpl(wrapper=True)
def pytest_runtest_teardown(item: pytest.Item, nextitem: Optional[pytest.Item]) -> Generator[None, None, None]:
    result = yield
    before = item.stash[_db_env_snapshot_key]
    leaked = {key: value for key, value in _db_env_snapshot().items() if value != before[key]}
    for key, original in before.items():
        if original is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = original
    assert not leaked, (
        f"{item.nodeid} leaked DB env vars past monkeypatch teardown: {leaked}. "
        "Product code under test writes DATABASE_URL(_READ_REPLICA) into os.environ as a side effect; "
        "monkeypatch only restores keys it has a record for, so a value written to a previously unset "
        "key survives the test and poisons every later test in this pytest-xdist worker process "
        "(DB-backed e2e tests arm themselves on DATABASE_URL and then fail to connect). "
        "Use the unset_database_url fixture (or monkeypatch.setenv) so restoration is registered."
    )
    return result


@pytest.fixture
def unset_database_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABASE_URL", "about-to-be-unset")
    monkeypatch.delenv("DATABASE_URL")
