"""LIT-3193 — passthrough endpoints. Drives proxy_logging.post_call_failure_hook
(the integration point pass_through_endpoint reaches on upstream >=300)."""

import asyncio

import pytest
from fastapi import HTTPException

from litellm.caching.dual_cache import DualCache
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
from litellm.proxy.utils import ProxyLogging

from ._helpers import (
    assert_server_span_attrs,
    make_fastapi_http_exception,
    make_httpx_status_error,
)


def _real_user_api_key_dict(parent_span):
    return UserAPIKeyAuth(
        api_key="sk-test",
        team_id="team-lit-3193",
        team_alias="lit-3193-team",
        parent_otel_span=parent_span,
    )


def _proxy_logging():
    return ProxyLogging(user_api_key_cache=UserApiKeyCache(DualCache()))


def _drive_passthrough_failure(*, exception, user_api_key_dict):
    asyncio.run(
        _proxy_logging().post_call_failure_hook(
            user_api_key_dict=user_api_key_dict,
            original_exception=exception,
            request_data={},
        )
    )


VERTEX_PATH = "/vertex_ai/v1/projects/p/locations/us-central1/publishers/google/models/gemini-1.5-pro:generateContent"


@pytest.mark.parametrize(
    "exception, expected_status",
    [
        (make_fastapi_http_exception(401, "no proxy key"), 401),
        (make_fastapi_http_exception(400, "bad request"), 400),
        (make_fastapi_http_exception(403, "upstream forbidden"), 403),
        (make_fastapi_http_exception(404, "upstream not found"), 404),
        (make_fastapi_http_exception(429, "upstream rate limit"), 429),
        (make_httpx_status_error(500, "upstream blew up"), 500),
        (make_httpx_status_error(502, "bad gateway"), 502),
        (make_httpx_status_error(503, "service unavailable"), 503),
        (make_fastapi_http_exception(502, "wrapped 502"), 502),
    ],
    ids=[
        "401-litellm-auth",
        "400-upstream",
        "403-upstream",
        "404-upstream",
        "429-upstream",
        "500-upstream-httpx",
        "502-upstream-httpx",
        "503-upstream-httpx",
        "502-wrapped",
    ],
)
def test_vertex_passthrough_failure_stamps_server_span(
    exception,
    expected_status,
    server_span_factory,
    otel_with_exporter,
    register_otel_callback,
):
    _otel, exporter = otel_with_exporter
    server_span = server_span_factory(
        VERTEX_PATH, http_route="/vertex_ai/{endpoint:path}"
    )
    uakd = _real_user_api_key_dict(server_span)

    _drive_passthrough_failure(exception=exception, user_api_key_dict=uakd)

    assert_server_span_attrs(
        exporter,
        expected_status=expected_status,
        expected_url_path=VERTEX_PATH,
        expected_http_route="/vertex_ai/{endpoint:path}",
        where=f"vertex passthrough {expected_status}",
    )


SMOKE_PASSTHROUGHS = [
    ("/bedrock/model/anthropic.claude-v2/invoke", "/bedrock/{endpoint:path}"),
    ("/anthropic/v1/messages", "/anthropic/{endpoint:path}"),
    ("/openai/v1/chat/completions", "/openai/{endpoint:path}"),
    ("/gemini/v1beta/models/gemini-pro:generateContent", "/gemini/{endpoint:path}"),
    ("/cohere/v1/chat", "/cohere/{endpoint:path}"),
    ("/azure/openai/deployments/gpt4/chat/completions", "/azure/{endpoint:path}"),
]


@pytest.mark.parametrize("path,http_route", SMOKE_PASSTHROUGHS)
@pytest.mark.parametrize(
    "exception, expected_status",
    [
        (make_fastapi_http_exception(400, "upstream bad request"), 400),
        (make_httpx_status_error(502, "upstream"), 502),
    ],
    ids=["400", "502"],
)
def test_passthrough_failure_stamps_server_span(
    path,
    http_route,
    exception,
    expected_status,
    server_span_factory,
    otel_with_exporter,
    register_otel_callback,
):
    _otel, exporter = otel_with_exporter
    server_span = server_span_factory(path, http_route=http_route)
    uakd = _real_user_api_key_dict(server_span)

    _drive_passthrough_failure(exception=exception, user_api_key_dict=uakd)

    assert_server_span_attrs(
        exporter,
        expected_status=expected_status,
        expected_url_path=path,
        expected_http_route=http_route,
        where=f"{path} {expected_status}",
    )
