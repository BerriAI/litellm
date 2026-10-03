import json
import os
import uuid
from datetime import date
from typing import Final
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import psycopg
import pytest
from prisma import Prisma
from psycopg import sql

from litellm.proxy.roi_calculator.branch_spend import read_branch_spend


@pytest.mark.asyncio
async def test_branch_spend_uses_request_tags_once_and_respects_utc_window() -> None:
    schema: Final = f"integration_roi_{uuid.uuid4().hex}"
    url: Final = os.environ["DATABASE_URL"]
    parsed: Final = urlsplit(url)
    scoped: Final = urlunsplit(parsed._replace(query=urlencode({**dict(parse_qsl(parsed.query)), "schema": schema})))
    repo: Final = "gitlab.com/group/project"
    tags: Final = (f"repo:{repo}", "branch:feature/one")
    rows: Final = (
        ("2026-09-01 00:00:00", 2, tags),
        ("2026-09-30 23:59:59.999", 3, tags + tags),
        ("2026-10-01 00:00:00", 100, tags),
        ("2026-08-31 23:59:59.999", 100, tags),
        ("2026-09-15 00:00:00", 100, tags + ("branch:conflict",)),
        ("2026-09-15 00:00:00", 100, tags + ("repo:gitlab.com/other/project",)),
        ("2026-09-15 00:00:00", 100, ("branch:feature/one",)),
        ("2026-09-15 00:00:00", 100, tags + ("litellm-roi-estimator",)),
        ("2026-09-15 00:00:00", 0, (f"repo:{repo}", "branch:free")),
        ("2026-09-15 00:00:00", 7, (f"repo:{repo}", "branch:Feature/one")),
    )
    with psycopg.connect(url, autocommit=True) as setup:
        setup.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
        try:
            setup.execute(
                sql.SQL(
                    'CREATE TABLE {}."LiteLLM_SpendLogs" ("startTime" timestamp, spend float, request_tags jsonb)'
                ).format(sql.Identifier(schema))
            )
            for timestamp, spend, request_tags in rows:
                setup.execute(
                    sql.SQL('INSERT INTO {}."LiteLLM_SpendLogs" VALUES (%s::timestamp, %s, %s::jsonb)').format(
                        sql.Identifier(schema)
                    ),
                    (timestamp, spend, json.dumps(request_tags)),
                )
            database: Final = Prisma(datasource={"url": scoped})
            await database.connect()
            try:
                result: Final = await read_branch_spend(database, date(2026, 9, 1), date(2026, 9, 30), (repo,))
            finally:
                await database.disconnect()
            costs: Final = {row.branch: (row.spend, row.requests) for row in result}
            assert costs == {"feature/one": (5, 2), "Feature/one": (7, 1), "free": (0, 1)}
        finally:
            setup.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema)))
