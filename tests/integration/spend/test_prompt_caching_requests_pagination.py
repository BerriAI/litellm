import json
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Final

import pytest
from pydantic import TypeAdapter

from litellm.types.management_endpoints.prompt_caching_requests import (
    PromptCachingRequestFilter,
    PromptCachingRequestsResponse,
)
from tests.integration._support.client import Gateway
from tests.integration._support.database import write_rows

_JSON_OBJECT: Final = TypeAdapter(dict[str, object])
_JSON_ROWS: Final = TypeAdapter(list[Mapping[str, object]])
_URL: Final = "/cost_optimization/prompt_caching/requests"
_MARKER: Final = "litellm_gateway_injected_cache"

_EXPECTED: Final = {
    "injected": ("injected-empty", "injected-deployment"),
    "hits": ("zero-fallback", "nested-read", "legacy-read", "boolean-number"),
    "all": (
        "zero-fallback",
        "write",
        "nested-write",
        "nested-read",
        "nested-creation",
        "legacy-read",
        "injected-empty",
        "injected-deployment",
        "boolean-number",
    ),
}


@dataclass(frozen=True)
class _Case:
    request_id: str
    metadata: Mapping[str, object]
    cache_hit: str | None = None
    start_time: datetime = datetime(2011, 9, 1, 12, 0, 0, 123456)


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
    _Case("outside-before", {_MARKER: ""}, start_time=datetime(2011, 8, 31, 23, 59, 59)),
    _Case(
        "outside-after", {"usage_object": {"cache_read_input_tokens": 100}}, start_time=datetime(2011, 9, 2, 0, 0, 1)
    ),
)


def _window(prefix: str) -> tuple[datetime, datetime]:
    day: Final = datetime(1900, 1, 1) + timedelta(days=int(prefix[2:14], 16) % 200000)
    return day, day + timedelta(days=1)


def _seed(prefix: str, cases: tuple[_Case, ...] = _CASES) -> None:
    shift: Final = _window(prefix)[0] - datetime(2011, 9, 1)
    for case in cases:
        write_rows(
            'INSERT INTO "LiteLLM_SpendLogs" (request_id, call_type, api_key, "startTime", "endTime", model,'
            " model_id, custom_llm_provider, spend, metadata, cache_hit)"
            " VALUES (%s, 'acompletion', %s, %s::timestamp, %s::timestamp, %s, %s, %s, %s, %s::jsonb, %s)",
            (
                f"{prefix}{case.request_id}",
                "test-key",
                (case.start_time + shift).isoformat(),
                (datetime(2011, 9, 1, 12, 0, 1) + shift).isoformat(),
                "claude-sonnet-5",
                "dep-a",
                "anthropic",
                "0.01",
                json.dumps(dict(case.metadata)),
                case.cache_hit,
            ),
        )


def _clean(prefix: str) -> None:
    write_rows('DELETE FROM "LiteLLM_SpendLogs" WHERE request_id LIKE %s', (f"{prefix}%",))


def _strip(prefix: str, request_id: str) -> str:
    assert request_id.startswith(prefix), request_id
    return request_id[len(prefix) :]


def _run_filter_checks(
    gateway: Gateway,
    filter: PromptCachingRequestFilter,
    prefix: str,
    key: str | None,
    window: tuple[datetime, datetime],
) -> None:
    expected: Final = _EXPECTED[filter]
    first: Final = gateway.request(
        "GET",
        _URL,
        params={
            "start_date": window[0].isoformat(),
            "end_date": window[1].isoformat(),
            "filter": filter,
            "page_size": "2",
        },
        key=key,
    )
    assert first.status_code == 200, first.text
    first_page: Final = PromptCachingRequestsResponse.model_validate_json(first.content)
    assert tuple(_strip(prefix, row.request_id) for row in first_page.requests) == expected[:2]
    assert first_page.has_more is (len(expected) > 2)
    assert (first_page.next_cursor is not None) is first_page.has_more
    if first_page.next_cursor is not None:
        assert _strip(prefix, first_page.next_cursor.request_id) == expected[1]
        assert first_page.next_cursor.start_time == first_page.requests[-1].start_time
        next_response: Final = gateway.request(
            "GET",
            _URL,
            params={
                "start_date": window[0].isoformat(),
                "end_date": window[1].isoformat(),
                "filter": filter,
                "page_size": "2",
                "cursor_start_time": first_page.next_cursor.start_time.astimezone(
                    timezone(timedelta(hours=-7))
                ).isoformat(),
                "cursor_request_id": first_page.next_cursor.request_id,
            },
            key=key,
        )
        assert next_response.status_code == 200, next_response.text
        next_page: Final = PromptCachingRequestsResponse.model_validate_json(next_response.content)
        assert tuple(_strip(prefix, row.request_id) for row in next_page.requests) == expected[2:4]
        assert next_page.has_more is (len(expected) > 4)
        assert (next_page.next_cursor is not None) is next_page.has_more
    second: Final = gateway.request(
        "GET",
        _URL,
        params={
            "start_date": window[0].isoformat(),
            "end_date": window[1].isoformat(),
            "filter": filter,
            "page_size": "100",
        },
        key=key,
    )
    assert second.status_code == 200, second.text
    complete: Final = PromptCachingRequestsResponse.model_validate_json(second.content)
    assert tuple(_strip(prefix, row.request_id) for row in complete.requests) == expected
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
    by_id: Final = {_strip(prefix, row.request_id): row for row in complete.requests}
    if filter == "all":
        assert by_id["injected-empty"].gateway_injected is True
        assert by_id["injected-empty"].net_savings is None
        assert by_id["legacy-read"].gateway_injected is False
        assert by_id["legacy-read"].net_savings is not None and by_id["legacy-read"].net_savings > 0
        assert by_id["write"].net_savings is not None and by_id["write"].net_savings < 0


@pytest.mark.asyncio
@pytest.mark.parametrize("filter", ["all", "injected", "hits"])
@pytest.mark.parametrize("role", ["admin", "view-only"])
async def test_request_filters_match_accounting_and_paginate_before_projection(
    gateway: Gateway, filter: PromptCachingRequestFilter, role: str
) -> None:
    prefix: Final = f"pc{uuid.uuid4().hex[:12]}:"
    _seed(prefix)
    try:
        if role == "admin":
            _run_filter_checks(gateway, filter, prefix, None, _window(prefix))
        else:
            with gateway.scenario() as scenario:
                viewer: Final = scenario.user(user_role="proxy_admin_viewer")
                _run_filter_checks(gateway, filter, prefix, scenario.key(user_id=viewer), _window(prefix))
    finally:
        _clean(prefix)


@pytest.mark.asyncio
@pytest.mark.parametrize("delete_before_cursor", [False, True])
async def test_cursor_keeps_remaining_requests_once_during_insertions_and_deletions(
    gateway: Gateway, delete_before_cursor: bool
) -> None:
    prefix: Final = f"pc{uuid.uuid4().hex[:12]}:"
    cases: Final = (
        *_CASES,
        _Case(
            "older-cache-read",
            {"usage_object": {"cache_read_input_tokens": 100}},
            start_time=datetime(2011, 9, 1, 11),
        ),
    )
    _seed(prefix, cases)
    try:
        window: Final = _window(prefix)
        expected: Final = (*_EXPECTED["all"], "older-cache-read")
        first: Final = gateway.request(
            "GET",
            _URL,
            params={"start_date": window[0].isoformat(), "end_date": window[1].isoformat(), "page_size": "2"},
        )
        assert first.status_code == 200, first.text
        first_page: Final = PromptCachingRequestsResponse.model_validate_json(first.content)
        assert tuple(_strip(prefix, row.request_id) for row in first_page.requests) == expected[:2]
        assert first_page.next_cursor is not None
        write_rows(
            'INSERT INTO "LiteLLM_SpendLogs" (request_id, call_type, api_key, "startTime", "endTime", model,'
            " model_id, custom_llm_provider, spend, metadata, cache_hit)"
            ' SELECT %s, call_type, api_key, %s, "endTime", model, model_id, custom_llm_provider, spend,'
            ' metadata, cache_hit FROM "LiteLLM_SpendLogs" WHERE request_id = %s',
            (f"{prefix}newer-request", (window[0] + timedelta(hours=13)).isoformat(), f"{prefix}{expected[0]}"),
        )
        write_rows(
            'INSERT INTO "LiteLLM_SpendLogs" (request_id, call_type, api_key, "startTime", "endTime", model,'
            " model_id, custom_llm_provider, spend, metadata, cache_hit)"
            ' SELECT %s, call_type, api_key, %s, "endTime", model, model_id, custom_llm_provider, spend,'
            ' metadata, cache_hit FROM "LiteLLM_SpendLogs" WHERE request_id = %s',
            (
                f"{prefix}zz-higher-id",
                (cases[0].start_time + (window[0] - datetime(2011, 9, 1))).isoformat(),
                f"{prefix}{expected[0]}",
            ),
        )
        if delete_before_cursor:
            write_rows('DELETE FROM "LiteLLM_SpendLogs" WHERE request_id = %s', (f"{prefix}{expected[0]}",))
        following: Final = gateway.request(
            "GET",
            _URL,
            params={
                "start_date": window[0].isoformat(),
                "end_date": window[1].isoformat(),
                "page_size": "100",
                "cursor_start_time": first_page.next_cursor.start_time.isoformat(),
                "cursor_request_id": first_page.next_cursor.request_id,
            },
        )
        assert following.status_code == 200, following.text
        following_page: Final = PromptCachingRequestsResponse.model_validate_json(following.content)
        assert tuple(_strip(prefix, row.request_id) for row in following_page.requests) == expected[2:]
        assert following_page.has_more is False
        assert following_page.next_cursor is None
    finally:
        _clean(prefix)
