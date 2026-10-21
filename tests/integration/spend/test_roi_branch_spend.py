import json
import os
import uuid
from collections.abc import Mapping
from datetime import date
from typing import Final
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import psycopg
import pytest
from prisma import Prisma
from psycopg import sql

from litellm.proxy.roi_calculator.branch_spend import read_branch_spend
from litellm.types.roi_calculator import ROIBranchSpend
from tests.integration._support.client import Gateway, JsonValue, object_value


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
        ("2026-09-15 00:00:00", 11, tags + ("litellm-roi-estimator",)),
        ("2026-09-15 00:00:00", 0, (f"repo:{repo}", "branch:free")),
        ("2026-09-15 00:00:00", 7, (f"repo:{repo}", "branch:Feature/one")),
    )
    with psycopg.connect(url, autocommit=True) as setup:
        setup.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
        try:
            setup.execute(
                sql.SQL(
                    'CREATE TABLE {}."LiteLLM_SpendLogs" '
                    '("startTime" timestamp, spend float, request_tags jsonb, metadata jsonb)'
                ).format(sql.Identifier(schema))
            )
            for timestamp, spend, request_tags in rows:
                setup.execute(
                    sql.SQL(
                        'INSERT INTO {}."LiteLLM_SpendLogs" ("startTime", spend, request_tags) '
                        "VALUES (%s::timestamp, %s, %s::jsonb)"
                    ).format(sql.Identifier(schema)),
                    (timestamp, spend, json.dumps(request_tags)),
                )
            for marker, spend, extra_tags in (
                (True, 100, ()),
                (True, 100, ("litellm-roi-estimator",)),
                (False, 13, ("litellm-roi-estimator",)),
                (None, 100, ("litellm-roi-estimator",)),
            ):
                setup.execute(
                    sql.SQL(
                        'INSERT INTO {}."LiteLLM_SpendLogs" VALUES (%s::timestamp, %s, %s::jsonb, %s::jsonb)'
                    ).format(sql.Identifier(schema)),
                    (
                        "2026-09-15 00:00:00",
                        spend,
                        json.dumps(tags + extra_tags),
                        json.dumps({"litellm_roi_estimator": marker}),
                    ),
                )
            database: Final = Prisma(datasource={"url": scoped})
            await database.connect()
            try:
                result: Final = await read_branch_spend(database, date(2026, 9, 1), date(2026, 9, 30), (repo,))
            finally:
                await database.disconnect()
            costs: Final = {row.branch: (row.spend, row.requests) for row in result}
            assert costs == {"feature/one": (18, 3), "Feature/one": (7, 1), "free": (0, 1)}
        finally:
            setup.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema)))


def test_documented_header_and_body_tags_reach_recorded_branch_and_pr_cost(gateway: Gateway) -> None:
    import asyncio
    from datetime import datetime, timezone

    from litellm.proxy.roi_calculator.branch_spend import attribute_branch_keys
    from tests.integration._support.client import eventually
    from tests.integration._support.database import read_rows
    from tests.integration._support.wire import Reply, Request, wire_server

    marker: Final = uuid.uuid4().hex
    repo: Final = f"github.com/integration/{marker}"
    branch: Final = "feature/tag-attribution"
    tags: Final = [f"repo:{repo}", f"branch:{branch}"]

    def respond(request: Request) -> Reply:
        body: Final = object_value(json.loads(request.body))
        assert "tags" not in body and "x-litellm-tags" not in request.headers
        return Reply(
            body=json.dumps(
                {
                    "id": f"chatcmpl-{uuid.uuid4().hex}",
                    "object": "chat.completion",
                    "created": 1,
                    "model": "owned-model",
                    "choices": [
                        {"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}
                    ],
                    "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
                }
            ).encode()
        )

    with wire_server(respond) as upstream, gateway.scenario() as scenario:
        model: Final = scenario.model(
            api_base=upstream.url + "/v1", input_cost_per_token=0.001, output_cost_per_token=0.002
        )
        examples: Final[tuple[tuple[Mapping[str, JsonValue], Mapping[str, str]], ...]] = (
            ({"metadata": {"tags": tags}}, {}),
            ({"tags": tags}, {}),
            ({}, {"x-litellm-tags": ", ".join(tags + tags)}),
        )
        for index, (payload, headers) in enumerate(examples):
            response: Final = gateway.request(
                "POST",
                "/v1/chat/completions",
                {
                    "model": model,
                    "messages": [{"role": "user", "content": f"tag attribution {index}"}],
                    **payload,
                },
                headers=headers,
            )
            assert response.status_code == 200, response.text
        assert len(upstream.drain()) == 3
        rows: Final = eventually(
            lambda: read_rows(
                'SELECT spend FROM "LiteLLM_SpendLogs" WHERE request_tags @> %s::jsonb', (json.dumps(tags),)
            ),
            lambda values: len(values) == 3,
            seconds=70,
        )
        expected: Final = 3 * (5 * 0.001 + 3 * 0.002)
        assert sum(float(row["spend"]) for row in rows) == pytest.approx(expected)

        async def recorded() -> tuple[ROIBranchSpend, ...]:
            database: Final = Prisma()
            await database.connect()
            try:
                today: Final = datetime.now(timezone.utc).date()
                return await read_branch_spend(database, today, today, (repo,), casefold_repo=True)
            finally:
                await database.disconnect()

        spending: Final = asyncio.run(recorded())
        costs: Final = attribute_branch_keys(((repo, 1, repo, branch),), spending)
        assert costs[(repo, 1)].spend == pytest.approx(expected)
        assert costs[(repo, 1)].requests == 3
        assert costs[(repo, 1)].status == "matched"
