from __future__ import annotations

import hashlib
import os
import time
from datetime import datetime
from typing import Final

import psycopg
from pydantic import AwareDatetime, BaseModel, ConfigDict, TypeAdapter

from setup.ui_query_isolation import validate_target
from setup.upgrade_worker import read_state, release_indexes


class UiFixture(BaseModel):
    model_config = ConfigDict(frozen=True)

    anchor: AwareDatetime
    api_key: str


def prepare_fixture() -> UiFixture:
    validate_target(os.environ["DATABASE_URL"])
    _finish_upgrade()
    with psycopg.connect(os.environ["DATABASE_URL"], connect_timeout=10) as connection:
        _ = connection.execute("SET statement_timeout='30s'")
        row: Final = connection.execute("SELECT spec->>'anchor', rows FROM litellm_perf_seed.checkpoint").fetchone()
        assert row is not None and TypeAdapter(int).validate_python(row[1]) > 0, "Missing completed seed rows"
        fixture: Final = UiFixture(
            anchor=TypeAdapter(datetime).validate_python(row[0]),
            api_key=hashlib.md5(b"perf-key:0", usedforsecurity=False).hexdigest()
            + hashlib.md5(b"perf-key-tail:0", usedforsecurity=False).hexdigest(),
        )
        return fixture


def _finish_upgrade() -> None:
    if os.environ.get("E2E_DATABASE_UPGRADE") != "1":
        return
    with psycopg.connect(os.environ["DATABASE_URL"], autocommit=True, connect_timeout=10) as connection:
        release_indexes(connection)
        deadline: Final = time.monotonic() + 6 * 3600
        while time.monotonic() < deadline:
            match read_state(connection).phase:
                case "failed":
                    raise AssertionError("Snapshot upgrade failed")
                case "complete":
                    return
                case _:
                    pass
            time.sleep(5)
    raise AssertionError("Snapshot indexes did not finish before UI measurements")
