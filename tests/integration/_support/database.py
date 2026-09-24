import hashlib
import os
from typing import Final

import psycopg
from psycopg.rows import dict_row
from pydantic import JsonValue, TypeAdapter

ROWS: Final = TypeAdapter(list[dict[str, JsonValue]])


def read_rows(
    query: str, parameters: tuple[str, ...], *, database_url: str | None = None
) -> list[dict[str, JsonValue]]:
    with psycopg.connect(database_url or os.environ["DATABASE_URL"], row_factory=dict_row) as connection:
        connection.execute("SET TRANSACTION READ ONLY")
        return ROWS.validate_python(connection.execute(query, parameters).fetchall())


def advisory_lock_key(*parts: str) -> int:
    return int.from_bytes(hashlib.sha256(":".join(parts).encode()).digest()[:8], "big", signed=True)


def legacy_advisory_lock_key(text: str) -> int:
    return int.from_bytes(hashlib.blake2b(text.encode(), digest_size=8).digest(), "big", signed=True)


def advisory_waiters(lock_key: int) -> list[dict[str, JsonValue]]:
    unsigned: Final = lock_key & 0xFFFFFFFFFFFFFFFF
    return read_rows(
        "SELECT pid FROM pg_locks WHERE locktype = 'advisory' AND objsubid = 1 "
        "AND classid = %s::oid AND objid = %s::oid AND NOT granted",
        (str(unsigned >> 32), str(unsigned & 0xFFFFFFFF)),
    )
