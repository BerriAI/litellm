import os
from collections.abc import Iterator
from typing import Final
from urllib.parse import urlsplit

import pytest

from tests.integration._support.database import read_rows, scratch_database


@pytest.fixture(autouse=True)
def unseeded_database() -> Iterator[None]:
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
