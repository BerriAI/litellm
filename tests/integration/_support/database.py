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


def execute(statement: str, parameters: tuple[str, ...]) -> None:
    with psycopg.connect(os.environ["DATABASE_URL"]) as connection:
        connection.execute(statement, parameters)
