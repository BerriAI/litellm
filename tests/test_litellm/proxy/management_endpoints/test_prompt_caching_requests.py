import json
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Final

import httpx
import psycopg
import pytest
import pytest_asyncio
from fastapi import FastAPI
from prisma import Prisma
from pydantic import TypeAdapter
from pytest_postgresql import factories

from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.management_endpoints.prompt_caching_requests import router
from litellm.proxy.spend_tracking.savings import (
    extract_cache_creation_tokens,
    extract_cache_read_tokens,
    marks_gateway_injection,
)
from litellm.types.management_endpoints.prompt_caching_requests import (
    PromptCachingRequestFilter,
    PromptCachingRequestsResponse,
)

pytestmark = pytest.mark.usefixtures("local_model_cost_map")

_cache_postgresql_proc: Final = factories.postgresql_proc()  # pyright: ignore[reportUnknownMemberType]  # third-party fixture factory has incomplete callable types
_cache_postgresql: Final = factories.postgresql("_cache_postgresql_proc")
_JSON_OBJECT: Final = TypeAdapter(Mapping[str, object])
_JSON_ROWS: Final = TypeAdapter(tuple[Mapping[str, object], ...])
_START: Final = "2026-09-01T00:00:00Z"
_END: Final = "2026-09-02T00:00:00Z"
_URL: Final = "/cost_optimization/prompt_caching/requests"
_MODEL: Final = "claude-sonnet-5"
_MARKER: Final = "litellm_gateway_injected_cache"
_DDL: Final = """
    CREATE TABLE "LiteLLM_SpendLogs" (
        request_id TEXT PRIMARY KEY, "startTime" TIMESTAMP, "endTime" TIMESTAMP,
        model TEXT, model_id TEXT, custom_llm_provider TEXT, spend DOUBLE PRECISION,
        metadata JSONB, cache_hit TEXT
    )
"""


@dataclass(frozen=True)
class _Case:
    request_id: str
    metadata: Mapping[str, object]
    cache_hit: str | None = None
    start_time: datetime = datetime(2026, 9, 1, 12, 0, 0, 123456)

    def matches(self, filter: PromptCachingRequestFilter) -> bool:
        if self.cache_hit is not None and self.cache_hit.lower() == "true":
            return False
        if not datetime(2026, 9, 1) <= self.start_time <= datetime(2026, 9, 2):
            return False
        usage: Final = self.metadata.get("usage_object")
        normalized: Final = _JSON_OBJECT.validate_python(usage) if isinstance(usage, Mapping) else None
        injected: Final = marks_gateway_injection(self.metadata, "dep-a")
        reads: Final = extract_cache_read_tokens(normalized)
        writes: Final = extract_cache_creation_tokens(normalized)
        match filter:
            case "injected":
                return injected
            case "hits":
                return reads > 0
            case "all":
                return injected or reads > 0 or writes > 0


_CASES: Final = (
    _Case("injected-empty", {_MARKER: ""}),
    _Case("injected-deployment", {_MARKER: "dep-a"}),
    _Case("wrong-deployment", {_MARKER: "dep-b"}),
    _Case("legacy-read", {"usage_object": {"cache_read_input_tokens": 100}}),
    _Case("nested-read", {"usage_object": {"prompt_tokens_details": {"cached_tokens": 100}}}),
    _Case("write", {"usage_object": {"cache_creation_input_tokens": 100}}),
    _Case("nested-write", {"usage_object": {"prompt_tokens_details": {"cache_write_tokens": 100}}}),
    _Case("nested-creation", {"usage_object": {"prompt_tokens_details": {"cache_creation_tokens": 100}}}),
    _Case(
        "top-precedence",
        {"usage_object": {"cache_read_input_tokens": -2, "prompt_tokens_details": {"cached_tokens": 100}}},
    ),
    _Case(
        "zero-fallback",
        {"usage_object": {"cache_read_input_tokens": 0, "prompt_tokens_details": {"cached_tokens": 100}}},
    ),
    _Case(
        "fractional-precedence",
        {"usage_object": {"cache_read_input_tokens": 0.5, "prompt_tokens_details": {"cached_tokens": 100}}},
    ),
    _Case("malformed-number", {"usage_object": {"cache_read_input_tokens": "100"}}),
    _Case("malformed-container", {"usage_object": [100]}),
    _Case("boolean-number", {"usage_object": {"cache_read_input_tokens": True}}),
    _Case("boolean-marker", {_MARKER: True}),
    _Case("response-cache", {_MARKER: "", "usage_object": {"cache_read_input_tokens": 100}}, "True"),
    _Case("outside-before", {_MARKER: ""}, start_time=datetime(2026, 8, 31, 23, 59, 59)),
    _Case(
        "outside-after", {"usage_object": {"cache_read_input_tokens": 100}}, start_time=datetime(2026, 9, 2, 0, 0, 1)
    ),
)


@pytest_asyncio.fixture(loop_scope="function")
async def _cache_prisma(
    _cache_postgresql: psycopg.Connection[tuple[object, ...]],
) -> AsyncIterator[Prisma]:
    info: Final = _cache_postgresql.info
    database: Final = Prisma(datasource={
        "url": f"postgresql://{info.user}@{info.host}:{info.port}/{info.dbname}?connection_limit=1",
    })
    await database.connect()
    try:
        yield database
    finally:
        await database.disconnect()


def _seed(connection: psycopg.Connection[tuple[object, ...]], cases: tuple[_Case, ...] = _CASES) -> None:
    with connection.cursor() as cursor:
        cursor.execute(_DDL)
        cursor.executemany(
            """INSERT INTO "LiteLLM_SpendLogs"
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s)""",
            tuple(
                (
                    case.request_id,
                    case.start_time,
                    datetime(2026, 9, 1, 12, 0, 1),
                    _MODEL,
                    "dep-a",
                    "anthropic",
                    0.01,
                    json.dumps(dict(case.metadata)),
                    case.cache_hit,
                )
                for case in cases
            ),
        )
    connection.commit()


def _app(role: LitellmUserRoles | None) -> FastAPI:
    application: Final = FastAPI()
    application.include_router(router)

    def caller() -> UserAPIKeyAuth:
        return UserAPIKeyAuth(user_role=role)

    application.dependency_overrides[user_api_key_auth] = caller
    return application


@pytest.mark.asyncio
@pytest.mark.parametrize("filter", ["all", "injected", "hits"])
@pytest.mark.parametrize("role", [LitellmUserRoles.PROXY_ADMIN, LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY])
async def test_request_filters_match_accounting_and_paginate_before_projection(
    _cache_postgresql: psycopg.Connection[tuple[object, ...]],
    _cache_prisma: Prisma,
    monkeypatch: pytest.MonkeyPatch,
    filter: PromptCachingRequestFilter,
    role: LitellmUserRoles,
) -> None:
    from litellm.proxy import proxy_server

    _seed(_cache_postgresql)
    monkeypatch.setattr(proxy_server, "prisma_client", SimpleNamespace(db=_cache_prisma))
    monkeypatch.setattr(proxy_server, "llm_router", None)
    expected: Final = tuple(sorted((case.request_id for case in _CASES if case.matches(filter)), reverse=True))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=_app(role)), base_url="http://test") as client:
        first: Final = await client.get(
            _URL, params={"start_date": _START, "end_date": _END, "filter": filter, "page_size": 2}
        )
        assert first.status_code == 200
        first_page: Final = PromptCachingRequestsResponse.model_validate_json(first.content)
        assert tuple(row.request_id for row in first_page.requests) == expected[:2]
        assert first_page.has_more is (len(expected) > 2)
        assert (first_page.next_cursor is not None) is first_page.has_more
        if first_page.next_cursor is not None:
            assert first_page.next_cursor.request_id == expected[1]
            assert first_page.next_cursor.start_time == first_page.requests[-1].start_time
            next_response: Final = await client.get(
                _URL, params={
                    "start_date": _START, "end_date": _END, "filter": filter, "page_size": 2,
                    "cursor_start_time": first_page.next_cursor.start_time.astimezone(
                        timezone(timedelta(hours=-7))
                    ).isoformat(),
                    "cursor_request_id": first_page.next_cursor.request_id,
                }
            )
            assert next_response.status_code == 200
            next_page: Final = PromptCachingRequestsResponse.model_validate_json(next_response.content)
            assert tuple(row.request_id for row in next_page.requests) == expected[2:4]
            assert next_page.has_more is (len(expected) > 4)
            assert (next_page.next_cursor is not None) is next_page.has_more
        second: Final = await client.get(
            _URL, params={"start_date": _START, "end_date": _END, "filter": filter, "page_size": 100}
        )
        assert second.status_code == 200
        complete: Final = PromptCachingRequestsResponse.model_validate_json(second.content)
        assert tuple(row.request_id for row in complete.requests) == expected
        assert complete.has_more is False
        assert complete.next_cursor is None
        assert all(row.start_time.tzinfo == timezone.utc for row in complete.requests)
        payload: Final = _JSON_OBJECT.validate_json(second.content)
        assert set(payload) == {"requests", "page_size", "has_more", "next_cursor"}
        serialized_rows: Final = _JSON_ROWS.validate_python(payload["requests"])
        assert set(serialized_rows[0]) == {
            "request_id",
            "start_time",
            "model",
            "gateway_injected",
            "cache_read_tokens",
            "cache_creation_tokens",
            "spend",
            "net_savings",
        }
        by_id: Final = {row.request_id: row for row in complete.requests}
        if filter == "all":
            assert by_id["injected-empty"].gateway_injected is True
            assert by_id["injected-empty"].net_savings is None
            assert by_id["legacy-read"].gateway_injected is False
            assert by_id["legacy-read"].net_savings is not None and by_id["legacy-read"].net_savings > 0
            assert by_id["write"].net_savings is not None and by_id["write"].net_savings < 0


@pytest.mark.asyncio
@pytest.mark.parametrize("role", [None, LitellmUserRoles.INTERNAL_USER, LitellmUserRoles.INTERNAL_USER_VIEW_ONLY])
async def test_non_admin_is_denied_before_database_access(
    role: LitellmUserRoles | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    from litellm.proxy import proxy_server

    monkeypatch.setattr(proxy_server, "prisma_client", None)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=_app(role)), base_url="http://test") as client:
        response: Final = await client.get(_URL, params={"start_date": _START, "end_date": _END})
    assert response.status_code == 403


@pytest.mark.asyncio
@pytest.mark.parametrize("params", [
    {"filter": "savings"}, {"page_size": 0}, {"page_size": 101}, {"start_date": "invalid"},
    {"cursor_start_time": "invalid", "cursor_request_id": "request"},
    {"cursor_start_time": _START, "cursor_request_id": ""},
])
async def test_invalid_request_is_rejected(params: Mapping[str, str | int]) -> None:
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=_app(LitellmUserRoles.PROXY_ADMIN)), base_url="http://test"
    ) as client:
        response: Final = await client.get(_URL, params={"start_date": _START, "end_date": _END, **params})
    assert response.status_code == 422


@pytest.mark.asyncio
@pytest.mark.parametrize("params", [{"cursor_start_time": _START}, {"cursor_request_id": "request"}])
async def test_incomplete_cursor_is_rejected(
    params: Mapping[str, str], monkeypatch: pytest.MonkeyPatch,
) -> None:
    from litellm.proxy import proxy_server

    monkeypatch.setattr(proxy_server, "prisma_client", None)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=_app(LitellmUserRoles.PROXY_ADMIN)), base_url="http://test"
    ) as client:
        response: Final = await client.get(_URL, params={"start_date": _START, "end_date": _END, **params})
    assert response.status_code == 400


@pytest.mark.asyncio
@pytest.mark.parametrize("delete_before_cursor", [False, True])
async def test_cursor_keeps_remaining_requests_once_during_insertions_and_deletions(
    _cache_postgresql: psycopg.Connection[tuple[object, ...]],
    _cache_prisma: Prisma,
    monkeypatch: pytest.MonkeyPatch,
    delete_before_cursor: bool,
) -> None:
    from litellm.proxy import proxy_server

    cases: Final = (*_CASES, _Case(
        "older-cache-read", {"usage_object": {"cache_read_input_tokens": 100}}, start_time=datetime(2026, 9, 1, 11),
    ))
    _seed(_cache_postgresql, cases)
    monkeypatch.setattr(proxy_server, "prisma_client", SimpleNamespace(db=_cache_prisma))
    monkeypatch.setattr(proxy_server, "llm_router", None)
    expected: Final = (*sorted((case.request_id for case in _CASES if case.matches("all")), reverse=True), "older-cache-read")
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=_app(LitellmUserRoles.PROXY_ADMIN)), base_url="http://test"
    ) as client:
        first: Final = await client.get(_URL, params={"start_date": _START, "end_date": _END, "page_size": 2})
        assert first.status_code == 200
        first_page: Final = PromptCachingRequestsResponse.model_validate_json(first.content)
        assert tuple(row.request_id for row in first_page.requests) == expected[:2]
        assert first_page.next_cursor is not None
        with _cache_postgresql.cursor() as cursor:
            cursor.executemany(
                """INSERT INTO "LiteLLM_SpendLogs"
                   SELECT %s, %s, "endTime", model, model_id, custom_llm_provider, spend, metadata, cache_hit
                   FROM "LiteLLM_SpendLogs" WHERE request_id = %s""",
                (
                    ("newer-request", datetime(2026, 9, 1, 13), expected[0]),
                    ("zz-higher-id", cases[0].start_time, expected[0]),
                ),
            )
            if delete_before_cursor:
                cursor.execute('DELETE FROM "LiteLLM_SpendLogs" WHERE request_id = %s', (expected[0],))
        _cache_postgresql.commit()
        following: Final = await client.get(_URL, params={
            "start_date": _START, "end_date": _END, "page_size": 100,
            "cursor_start_time": first_page.next_cursor.start_time.isoformat(),
            "cursor_request_id": first_page.next_cursor.request_id,
        })
        assert following.status_code == 200
        following_page: Final = PromptCachingRequestsResponse.model_validate_json(following.content)
        assert tuple(row.request_id for row in following_page.requests) == expected[2:]
        assert following_page.has_more is False
        assert following_page.next_cursor is None
