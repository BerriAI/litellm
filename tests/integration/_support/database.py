import os
import uuid
from collections.abc import Generator
from contextlib import contextmanager
from typing import Final
from urllib.parse import urlsplit, urlunsplit

import psycopg
from psycopg import sql
from psycopg.rows import dict_row
from pydantic import JsonValue, TypeAdapter

ROWS: Final = TypeAdapter(list[dict[str, JsonValue]])


def read_rows(query: str, parameters: tuple[str, ...]) -> list[dict[str, JsonValue]]:
    with psycopg.connect(os.environ["DATABASE_URL"], row_factory=dict_row) as connection:
        connection.execute("SET TRANSACTION READ ONLY")
        return ROWS.validate_python(connection.execute(query, parameters).fetchall())


@contextmanager
def scratch_database() -> Generator[str]:
    name: Final = f"integration_{uuid.uuid4().hex}"
    with psycopg.connect(os.environ["DATABASE_URL"], autocommit=True) as admin:
        admin.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))
        try:
            yield urlunsplit(urlsplit(os.environ["DATABASE_URL"])._replace(path=f"/{name}"))
        finally:
            admin.execute(sql.SQL("DROP DATABASE {} WITH (FORCE)").format(sql.Identifier(name)))
