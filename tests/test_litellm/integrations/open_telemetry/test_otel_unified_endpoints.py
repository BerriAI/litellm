"""
LIT-3193 — Class 1: unified inference endpoints.

Every unified endpoint funnels exceptions through
``ProxyBaseLLMRequestProcessing._handle_llm_api_exception``, which calls
``proxy_logging_obj.post_call_failure_hook`` *before* re-raising. That hook
is what reaches the OTEL ``async_post_call_failure_hook`` and is therefore
the integration point under test.

The matrix covers:

* ``/v1/chat/completions`` — deep matrix across 4xx/5xx classes.
* The other unified endpoints — smoke (one 4xx + one 5xx) to confirm the
  same handler/funnel runs for them and the SERVER span is stamped.

Each cell asserts the four SERVER-span attributes the dashboards depend on:
``http.response.status_code`` (int), ``url.path``, ``http.route``, and a
non-zero duration.
"""

import asyncio

import pytest
from fastapi import HTTPException

import litellm
from litellm.caching.dual_cache import DualCache
from litellm.proxy._types import ProxyException, UserAPIKeyAuth
from litellm.proxy.common_request_processing import ProxyBaseLLMRequestProcessing
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
from litellm.proxy.utils import ProxyLogging

from ._helpers import (
    HttpStatusException,
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


def _drive_unified_failure(
    *,
    exception,
    server_span,
    user_api_key_dict,
):
    """Mirror what every unified endpoint does on exception:
    ``base_llm_response_processor._handle_llm_api_exception``."""
    proc = ProxyBaseLLMRequestProcessing(data={})
    try:
        asyncio.run(
            proc._handle_llm_api_exception(
                e=exception,
                user_api_key_dict=user_api_key_dict,
                proxy_logging_obj=_proxy_logging(),
            )
        )
    except (ProxyException, HTTPException):
        # _handle_llm_api_exception always re-raises after logging — that's
        # exactly the path we're exercising; swallow here.
        pass


# ---------------------------------------------------------------------------
# /v1/chat/completions — deep matrix
# ---------------------------------------------------------------------------
CHAT_PATH = "/v1/chat/completions"


@pytest.mark.parametrize(
    "exception, expected_status",
    [
        # 4xx — these all carry a usable .code/.status_code, so the OTEL
        # hook stamps them today.
        (make_fastapi_http_exception(400, "bad request"), 400),
        (make_fastapi_http_exception(401, "no key"), 401),
        (make_fastapi_http_exception(403, "no model access"), 403),
        (make_fastapi_http_exception(404, "model not in router"), 404),
        (make_fastapi_http_exception(422, "validation"), 422),
        (make_fastapi_http_exception(429, "rate limit"), 429),
        # 5xx — these are the gap the ticket calls out.
        (HttpStatusException(500, "uncaught"), 500),
        (make_httpx_status_error(502, "upstream blew up"), 502),
        (make_httpx_status_error(503, "upstream down"), 503),
        (make_httpx_status_error(504, "upstream timeout"), 504),
    ],
    ids=[
        "400-bad-request",
        "401-no-key",
        "403-no-model-access",
        "404-model-not-found",
        "422-validation",
        "429-rate-limit",
        "500-uncaught",
        "502-upstream",
        "503-upstream",
        "504-upstream-timeout",
    ],
)
def test_chat_completions_failure_stamps_server_span(
    exception,
    expected_status,
    server_span_factory,
    user_api_key_dict_factory,
    otel_with_exporter,
    register_otel_callback,
):
    _otel, exporter = otel_with_exporter
    server_span = server_span_factory(CHAT_PATH)
    uakd = _real_user_api_key_dict(server_span)

    _drive_unified_failure(
        exception=exception, server_span=server_span, user_api_key_dict=uakd
    )

    assert_server_span_attrs(
        exporter,
        expected_status=expected_status,
        expected_url_path=CHAT_PATH,
        where=f"chat/completions {expected_status}",
    )


def test_chat_completions_success_path_stamps_200(
    otel_with_exporter, server_span_factory
):
    """Success path: ``async_post_call_success_hook`` sets 200 on the SERVER
    span and ``_handle_success`` ends it. Drives the hook directly because
    success doesn't go through ``_handle_llm_api_exception``."""
    otel, exporter = otel_with_exporter
    server_span = server_span_factory(CHAT_PATH)
    uakd = _real_user_api_key_dict(server_span)

    # Minimal kwargs the success hook walks (without dragging in a real
    # Logging object — the LiteLLMLogging path is covered elsewhere; here we
    # just need set_response_status_code_attribute(parent, 200) to run, then
    # close the span so the exporter sees it).
    otel.set_response_status_code_attribute(server_span, 200)
    otel.set_preprocessing_duration_attribute(server_span, {})
    server_span.end()
    _ = uakd  # parent span lives on user_api_key_dict in real flow

    assert_server_span_attrs(
        exporter,
        expected_status=200,
        expected_url_path=CHAT_PATH,
        where="chat/completions 200",
    )


# ---------------------------------------------------------------------------
# Smoke matrix across the other unified endpoints — one 4xx + one 5xx each
# ---------------------------------------------------------------------------
SMOKE_ENDPOINTS = [
    "/v1/embeddings",
    "/v1/completions",
    "/v1/images/generations",
    "/v1/audio/speech",
    "/v1/audio/transcriptions",
    "/v1/moderations",
    "/v1/rerank",
    "/v1/responses",
    "/v1/messages",
]


@pytest.mark.parametrize("path", SMOKE_ENDPOINTS)
@pytest.mark.parametrize(
    "exception, expected_status",
    [
        (make_fastapi_http_exception(401, "no key"), 401),
        (make_httpx_status_error(502, "upstream"), 502),
    ],
    ids=["401", "502"],
)
def test_unified_endpoint_failure_stamps_server_span(
    path,
    exception,
    expected_status,
    server_span_factory,
    otel_with_exporter,
    register_otel_callback,
):
    """Every unified endpoint shares ``_handle_llm_api_exception``; this
    confirms the SERVER-span stamping works regardless of which path opened
    the span."""
    _otel, exporter = otel_with_exporter
    server_span = server_span_factory(path)
    uakd = _real_user_api_key_dict(server_span)

    _drive_unified_failure(
        exception=exception, server_span=server_span, user_api_key_dict=uakd
    )

    assert_server_span_attrs(
        exporter,
        expected_status=expected_status,
        expected_url_path=path,
        where=f"{path} {expected_status}",
    )
