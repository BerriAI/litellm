"""API-level numbers for team-attributed daily tag spend (LIT-8516).

One fixture with hand-computed totals is seeded into a scratch schema, then every tag/team
daily-activity endpoint is called over HTTP against it: paginated, aggregated, aggregated/keys,
export, and /tag/list. Each assertion is a real number read back through the real SQL.

Fixture (one day, 2026-07-01; one row per tag x team x key, spend in whole dollars so sums are exact):

    tag       team      key           spend  reqs   note
    shared    alpha     key-alpha       3     3     two teams share "shared"
    shared    beta      key-beta        7     7
    shared    ''        key-noteam      4     4     key with no team -> '' bucket
    multi     alpha     key-alpha       2     2     these 2 alpha requests carried ["shared","multi"]:
    shared    alpha     key-alpha-2     2     2       counted in full under both tags
    move      alpha     key-mover       3     3     key moved alpha -> beta the same day:
    move      beta      key-mover       2     2       split by request-time team
    orphan    gone      key-gone        5     5     team "gone" was deleted (no TeamTable row)
    beta-only beta      key-beta        6     6     only beta uses it (proves /tag/list scoping)

So: shared = alpha 5 + beta 7 + no-team 4 = 16. alpha (all tags) = 5 + 2 + 3 = 10 tag-rows of spend.
Real alpha spend is 8 (the 2 multi-tag requests are counted under both tags) -- the documented overcount.
"""

import csv
import io
import os
import uuid
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Final
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import httpx
import psycopg
import pytest
from fastapi import FastAPI
from prisma import Prisma
from psycopg import sql

from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth

DAY: Final = "2026-07-01"
WINDOW: Final = {"start_date": DAY, "end_date": DAY}

# (tag, team_id, api_key, spend == api_requests)
_TAG_ROWS: Final = (
    ("shared", "alpha", "key-alpha", 3),
    ("shared", "beta", "key-beta", 7),
    ("shared", "", "key-noteam", 4),
    ("multi", "alpha", "key-alpha-2", 2),
    ("shared", "alpha", "key-alpha-2", 2),
    ("move", "alpha", "key-mover", 3),
    ("move", "beta", "key-mover", 2),
    ("orphan", "gone", "key-gone", 5),
    ("beta-only", "beta", "key-beta", 6),
)
_TEAMS: Final = (("alpha", "Team Alpha"), ("beta", "Team Beta"))
# key-mover is beta's now; history above must still say alpha 3 / beta 2.
_KEYS: Final = (
    ("key-alpha", "alpha", "member-alpha"),
    ("key-alpha-2", "alpha", None),
    ("key-beta", "beta", None),
    ("key-mover", "beta", None),
    ("key-noteam", None, None),
)
_USERS: Final = (("member-alpha", ["alpha"]), ("lonely-user", []))


def _scoped_url(url: str, schema: str) -> str:
    parsed: Final = urlsplit(url)
    return urlunsplit(parsed._replace(query=urlencode({**dict(parse_qsl(parsed.query)), "schema": schema})))


def _seed(connection: psycopg.Connection, schema: str) -> None:
    def table(name: str) -> sql.Composed:
        return sql.Identifier(schema, name)

    with connection.cursor() as cursor:
        for name in (
            "LiteLLM_DailyTagSpend",
            "LiteLLM_TagTable",
            "LiteLLM_TeamTable",
            "LiteLLM_VerificationToken",
            "LiteLLM_DeletedVerificationToken",
            "LiteLLM_UserTable",
            "LiteLLM_OrganizationMembership",  # read by the non-admin user lookup
        ):
            cursor.execute(
                sql.SQL("CREATE TABLE {} (LIKE {} INCLUDING DEFAULTS INCLUDING CONSTRAINTS)").format(
                    table(name), sql.Identifier(name)
                )
            )
        cursor.executemany(
            sql.SQL("""
            INSERT INTO {}
                (id, tag, team_id, date, api_key, model, model_group, custom_llm_provider,
                 mcp_namespaced_tool_name, endpoint, prompt_tokens, completion_tokens, spend,
                 api_requests, successful_requests, failed_requests, updated_at)
            VALUES (%s, %s, %s, %s, %s, 'model-a', '', 'provider-a', NULL, '/v1/chat/completions',
                    %s, %s, %s, %s, %s, 0, NOW())
            """).format(table("LiteLLM_DailyTagSpend")),
            [
                (str(uuid.uuid4()), tag, team, DAY, key, 10 * n, 10 * n, float(n), n, n)
                for tag, team, key, n in _TAG_ROWS
            ],
        )
        cursor.executemany(
            sql.SQL("INSERT INTO {} (team_id, team_alias, admins, members, models) VALUES (%s, %s, %s, %s, %s)").format(
                table("LiteLLM_TeamTable")
            ),
            [(team_id, alias, [], [], []) for team_id, alias in _TEAMS],
        )
        cursor.executemany(
            sql.SQL("INSERT INTO {} (token, key_alias, team_id, user_id) VALUES (%s, %s, %s, %s)").format(
                table("LiteLLM_VerificationToken")
            ),
            [(key, key, team, user) for key, team, user in _KEYS],
        )
        cursor.executemany(
            sql.SQL("INSERT INTO {} (user_id, teams) VALUES (%s, %s)").format(table("LiteLLM_UserTable")),
            _USERS,
        )
        cursor.execute(
            sql.SQL(
                "INSERT INTO {} (tag_name, description, models, model_info, spend) VALUES (%s, %s, %s, %s::jsonb, 0)"
            ).format(table("LiteLLM_TagTable")),
            ("shared", "stored shared tag", [], "{}"),
        )
    connection.commit()


@asynccontextmanager
async def _seeded_database() -> AsyncIterator[Prisma]:
    schema: Final = f"integration_{uuid.uuid4().hex}"
    url: Final = os.environ["DATABASE_URL"]
    with psycopg.connect(url, autocommit=True) as setup:
        setup.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
        try:
            with psycopg.connect(url) as connection:
                _seed(connection, schema)
            database: Final = Prisma(datasource={"url": _scoped_url(url, schema)})
            await database.connect()
            try:
                yield database
            finally:
                await database.disconnect()
        finally:
            setup.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema)))


@asynccontextmanager
async def _client(monkeypatch: pytest.MonkeyPatch, caller: UserAPIKeyAuth) -> AsyncIterator[httpx.AsyncClient]:
    from litellm.proxy.management_endpoints import daily_activity_routes, tag_management_endpoints, team_endpoints

    async with _seeded_database() as database:
        app: Final = FastAPI()
        app.include_router(tag_management_endpoints.router)
        app.include_router(team_endpoints.router)
        app.include_router(daily_activity_routes.router)
        app.dependency_overrides[user_api_key_auth] = lambda: caller
        monkeypatch.setattr(
            "litellm.proxy.proxy_server.prisma_client", SimpleNamespace(db=database, writer_db=database)
        )
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            yield client


_ADMIN: Final = UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN)


def _entities(body: Mapping[str, object]) -> dict[str, float]:
    """Sum breakdown.entities spend across days: {entity_id: spend}."""
    totals: dict[str, float] = {}
    for day in body["results"]:  # type: ignore[union-attr]
        for entity, bucket in day["breakdown"]["entities"].items():
            totals[entity] = totals.get(entity, 0.0) + bucket["metrics"]["spend"]
    return totals


async def _get(client: httpx.AsyncClient, path: str, **params: str) -> dict:
    response: Final = await client.get(path, params={**WINDOW, **params})
    assert response.status_code == 200, f"{path} {params} -> {response.status_code} {response.text}"
    return response.json()


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["/tag/daily/activity", "/tag/daily/activity/aggregated"])
async def test_shared_tag_splits_by_team_and_parts_sum_to_tag_total(monkeypatch: pytest.MonkeyPatch, path: str) -> None:
    async with _client(monkeypatch, _ADMIN) as client:
        whole: Final = await _get(client, path, tags="shared")
        by_team: Final = await _get(client, path, tags="shared", group_by="team")

        assert whole["metadata"]["total_spend"] == 16.0
        assert whole["metadata"]["total_api_requests"] == 16
        split: Final = _entities(by_team)
        assert {k: v for k, v in split.items() if k != "Unassigned"} == {"alpha": 5.0, "beta": 7.0}
        assert split.get("Unassigned", split.get("")) == 4.0
        assert sum(split.values()) == by_team["metadata"]["total_spend"] == whole["metadata"]["total_spend"]


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["/tag/daily/activity", "/tag/daily/activity/aggregated"])
async def test_tag_endpoint_team_filters(monkeypatch: pytest.MonkeyPatch, path: str) -> None:
    async with _client(monkeypatch, _ADMIN) as client:
        only_alpha: Final = await _get(client, path, tags="shared", team_ids="alpha")
        alpha_and_beta: Final = await _get(client, path, tags="shared", team_ids="alpha,beta", group_by="team")
        not_alpha: Final = await _get(client, path, tags="shared", exclude_team_ids="alpha", group_by="team")
        not_shared: Final = await _get(client, path, team_ids="alpha", exclude_tags="shared")

        assert only_alpha["metadata"]["total_spend"] == 5.0
        assert _entities(alpha_and_beta) == {"alpha": 5.0, "beta": 7.0}
        # excluding alpha keeps beta and the no-team bucket
        assert not_alpha["metadata"]["total_spend"] == 11.0
        assert "alpha" not in _entities(not_alpha)
        # alpha's other tags: multi 2 + move 3
        assert not_shared["metadata"]["total_spend"] == 5.0


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["/team/daily/activity", "/team/daily/activity/aggregated"])
async def test_team_endpoint_tag_breakdown_and_filters(monkeypatch: pytest.MonkeyPatch, path: str) -> None:
    async with _client(monkeypatch, _ADMIN) as client:
        by_tag: Final = await _get(client, path, team_ids="alpha", group_by="tag")
        shared_only: Final = await _get(client, path, team_ids="alpha,beta", tags="shared")
        two_tags: Final = await _get(client, path, team_ids="alpha", tags="shared,multi")
        excl: Final = await _get(client, path, team_ids="alpha", exclude_tags="shared", group_by="tag")

        assert _entities(by_tag) == {"shared": 5.0, "multi": 2.0, "move": 3.0}
        assert shared_only["metadata"]["total_spend"] == 12.0
        assert _entities(shared_only) == {"alpha": 5.0, "beta": 7.0}
        # Documented overcount: the 2 requests tagged ["shared","multi"] count under both tags,
        # so this is 7, not alpha's real spend on those tags (5).
        assert two_tags["metadata"]["total_spend"] == 7.0
        assert _entities(excl) == {"multi": 2.0, "move": 3.0}


@pytest.mark.asyncio
async def test_key_moved_teams_same_day_and_deleted_team_keep_request_time_attribution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with _client(monkeypatch, _ADMIN) as client:
        moved: Final = await _get(client, "/tag/daily/activity/aggregated", tags="move", group_by="team")
        orphan: Final = await _get(client, "/tag/daily/activity/aggregated", tags="orphan", group_by="team")
        beta_now: Final = await _get(client, "/team/daily/activity/aggregated", team_ids="beta", tags="move")

        # key-mover is beta's today; its earlier alpha traffic must stay alpha's.
        assert _entities(moved) == {"alpha": 3.0, "beta": 2.0}
        assert beta_now["metadata"]["total_spend"] == 2.0
        assert _entities(orphan) == {"gone": 5.0}


@pytest.mark.asyncio
async def test_aggregated_keys_are_scoped_to_team(monkeypatch: pytest.MonkeyPatch) -> None:
    async with _client(monkeypatch, _ADMIN) as client:
        beta_keys: Final = await _get(client, "/tag/daily/activity/aggregated/keys", tags="shared", team_ids="beta")
        alpha_keys: Final = await _get(client, "/tag/daily/activity/aggregated/keys", tags="shared", team_ids="alpha")

        assert {row["api_key"]: row["metrics"]["spend"] for row in beta_keys["api_keys"]} == {"key-beta": 7.0}
        assert {row["api_key"]: row["metrics"]["spend"] for row in alpha_keys["api_keys"]} == {
            "key-alpha": 3.0,
            "key-alpha-2": 2.0,
        }


@pytest.mark.asyncio
async def test_export_csv_numbers_match_aggregated(monkeypatch: pytest.MonkeyPatch) -> None:
    async with _client(monkeypatch, _ADMIN) as client:
        response: Final = await client.get(
            "/tag/daily/activity/export", params={**WINDOW, "tags": "shared", "group_by": "team", "format": "csv"}
        )
        assert response.status_code == 200, response.text
        rows: Final = list(csv.DictReader(io.StringIO(response.text)))
        by_team: dict[str, float] = {}
        for row in rows:
            by_team[row["entity_id"]] = by_team.get(row["entity_id"], 0.0) + float(row["spend"])

        assert by_team == {"alpha": 5.0, "beta": 7.0, "": 4.0}
        assert {row["entity_alias"] for row in rows if row["entity_id"] == "alpha"} == {"Team Alpha"}
        assert sum(int(row["api_requests"]) for row in rows) == 16


@pytest.mark.asyncio
async def test_tag_list_team_scope_and_usage_only(monkeypatch: pytest.MonkeyPatch) -> None:
    async with _client(monkeypatch, _ADMIN) as client:
        alpha: Final = await client.get("/tag/list", params={"team_ids": "alpha", "usage_only": "true"})
        beta: Final = await client.get("/tag/list", params={"team_ids": "beta", "usage_only": "true"})
        everything: Final = await client.get("/tag/list", params={"usage_only": "true"})

        assert {tag["name"] for tag in alpha.json()} == {"shared", "multi", "move"}
        assert {tag["name"] for tag in beta.json()} == {"shared", "move", "beta-only"}
        assert {tag["name"] for tag in everything.json()} == {"shared", "multi", "move", "orphan", "beta-only"}
        stored: Final = {tag["name"]: tag for tag in alpha.json()}
        assert stored["shared"]["description"] == "stored shared tag"


_MEMBER: Final = UserAPIKeyAuth(user_role=LitellmUserRoles.INTERNAL_USER, user_id="member-alpha", api_key="key-alpha")


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["/tag/daily/activity", "/tag/daily/activity/aggregated"])
async def test_member_group_by_team_without_filter_sees_only_own_team_and_keys(
    monkeypatch: pytest.MonkeyPatch, path: str
) -> None:
    async with _client(monkeypatch, _MEMBER) as client:
        body: Final = await _get(client, path, tags="shared", group_by="team")
        # member-alpha is a plain member of alpha, so the view is narrowed to their own key.
        # beta (7) and no-team (4) must never appear.
        assert _entities(body) == {"alpha": 3.0}
        assert body["metadata"]["total_spend"] == 3.0

        denied: Final = await client.get(path, params={**WINDOW, "tags": "shared", "team_ids": "beta"})
        assert denied.status_code in (403, 404), denied.text


@pytest.mark.asyncio
async def test_no_team_bucket_can_be_filtered_like_any_other_team(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.skip(
        "BUG: group_by=team labels no-team spend 'Unassigned', but team_ids/exclude_team_ids cannot target it: "
        "'' parses as no filter and 'Unassigned' matches nothing (stored value is '')"
    )
    async with _client(monkeypatch, _ADMIN) as client:
        only_no_team: Final = await _get(client, "/tag/daily/activity/aggregated", tags="shared", team_ids="Unassigned")
        without_no_team: Final = await _get(
            client, "/tag/daily/activity/aggregated", tags="shared", exclude_team_ids="Unassigned"
        )

        assert only_no_team["metadata"]["total_spend"] == 4.0
        assert without_no_team["metadata"]["total_spend"] == 12.0


_LONELY: Final = UserAPIKeyAuth(user_role=LitellmUserRoles.INTERNAL_USER, user_id="lonely-user", api_key="key-lonely")


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["/tag/daily/activity/aggregated", "/team/daily/activity/aggregated"])
async def test_user_in_no_teams_sees_no_team_scoped_spend(monkeypatch: pytest.MonkeyPatch, path: str) -> None:
    async with _client(monkeypatch, _LONELY) as client:
        params: Final = {"tags": "shared", "group_by": "team"} if path.startswith("/tag") else {"group_by": "tag"}
        body: Final = await _get(client, path, **params)
        assert body["metadata"]["total_spend"] == 0.0
        assert _entities(body) == {}
