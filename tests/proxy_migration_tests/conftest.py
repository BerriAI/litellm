from __future__ import annotations

import os
from collections.abc import Iterator
from typing import Final
from urllib.parse import urlsplit

import pytest


@pytest.fixture(scope="session", autouse=True)
def worker_database() -> Iterator[None]:
    if not os.environ.get("PYTEST_XDIST_WORKER") or "DATABASE_URL" not in os.environ:
        yield
        return
    from tests.integration._support.database import read_rows, scratch_database

    admin_url: Final = os.environ["DATABASE_URL"]
    with scratch_database() as database, pytest.MonkeyPatch.context() as environment:
        environment.setenv("DATABASE_URL", database)
        environment.delenv("DIRECT_URL", raising=False)
        environment.delenv("DATABASE_URL_READ_REPLICA", raising=False)
        yield
    assert (
        read_rows(
            "SELECT datname FROM pg_database WHERE datname = %s",
            (urlsplit(database).path.lstrip("/"),),
            database_url=admin_url,
        )
        == []
    )
