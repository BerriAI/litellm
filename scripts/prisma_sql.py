#!/usr/bin/env python3
"""Print the SQL Prisma sends for one ORM call, as the starting point of a shared query file.

    SCRATCH_DATABASE_URL=postgresql://... python scripts/prisma_sql.py \\
        'db.litellm_verificationtoken.update(where={"token": "t"}, data={"budget_limits": "{}"})'

The expression runs for real against SCRATCH_DATABASE_URL with `db` bound to a connected
Prisma client, so point it at a scratch database migrated by prisma migrate deploy.
"""

import asyncio
import inspect
import json
import os
import re
import sys
import tempfile
from collections.abc import Iterator
from contextlib import redirect_stdout
from pathlib import Path
from typing import Final

from prisma import Prisma
from prisma.types import DatasourceOverride
from pydantic import BaseModel

TRACEPARENT: Final = re.compile(r"\s*/\* traceparent=[^*]*\*/")
PUBLIC_SCHEMA: Final = '"public".'


class _Fields(BaseModel):
    query: str | None = None
    params: str = "[]"


class _EngineLine(BaseModel):
    fields: _Fields


def _statements(log: str) -> Iterator[tuple[str, str]]:
    lines: Final = (line for line in log.splitlines() if line.startswith("{"))
    fields: Final = (_EngineLine.model_validate_json(line).fields for line in lines)
    return (
        (TRACEPARENT.sub("", field.query).replace(PUBLIC_SCHEMA, ""), field.params)
        for field in fields
        if field.query is not None
    )


async def _run(url: str, expression: str) -> None:
    db: Final = Prisma(datasource=DatasourceOverride(url=url), log_queries=True)
    await db.connect()
    try:
        scope: Final = {"db": db}  # mutable-ok: eval takes its globals as a plain dict
        call: Final[object] = eval(expression, scope)  # pyright: ignore[reportAny]  # eval of the caller's ORM call is untyped
        result: Final[object] = await call if inspect.isawaitable(call) else call
    finally:
        await db.disconnect()
    sys.stderr.write(f"-- result: {result!r}\n")


def main() -> int:
    if len(sys.argv) != 2:
        sys.stderr.write(__doc__ or "")
        return 2
    url: Final = os.environ.get("SCRATCH_DATABASE_URL")
    if not url:
        sys.stderr.write("set SCRATCH_DATABASE_URL to a scratch Postgres migrated by prisma migrate deploy\n")
        return 2
    with tempfile.TemporaryDirectory() as directory:
        log_path: Final = Path(directory) / "engine.log"
        with log_path.open("w") as log, redirect_stdout(log):
            asyncio.run(_run(url, sys.argv[1]))
        statements: Final = tuple(_statements(log_path.read_text()))
    sys.stdout.write(
        "".join(
            f"-- statement {index}, params {json.loads(params)}\n{sql}\n\n"
            for index, (sql, params) in enumerate(statements, 1)
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
