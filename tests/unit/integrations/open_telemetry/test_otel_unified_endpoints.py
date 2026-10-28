"""LIT-3193 — unified inference endpoints. Drives _handle_llm_api_exception
to assert SERVER-span attrs (status, url.path, http.route, duration)."""

import asyncio

import pytest
from fastapi import HTTPException

from opentelemetry.trace import Status, StatusCode

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
        pass


CHAT_PATH = "/v1/chat/completions"


@pytest.mark.parametrize(
    "exception, expected_status",
    [
        (make_fastapi_http_exception(400, "bad request"), 400),
        (make_fastapi_http_exception(401, "no key"), 401),
        (make_fastapi_http_exception(403, "no model access"), 403),
        (make_fastapi_http_exception(404, "model not in router"), 404),
        (make_fastapi_http_exception(422, "validation"), 422),
        (make_fastapi_http_exception(429, "rate limit"), 429),
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
    otel, exporter = otel_with_exporter
    server_span = server_span_factory(CHAT_PATH)
    _real_user_api_key_dict(server_span)

    otel.set_response_status_code_attribute(server_span, 200)
    otel.set_preprocessing_duration_attribute(server_span, {})
    server_span.set_status(Status(StatusCode.OK))
    server_span.end()

    assert_server_span_attrs(
        exporter,
        expected_status=200,
        expected_url_path=CHAT_PATH,
        where="chat/completions 200",
    )


# /v1/responses ends the proxy span before async_post_call_success_hook fires,
# so the 200 stamp must happen at span close (here), not in the hook.
@pytest.mark.parametrize(
    "path", ["/v1/chat/completions", "/v1/messages", "/v1/responses"]
)
def test_end_proxy_span_from_kwargs_stamps_200(
    path, otel_with_exporter, server_span_factory
):
    from datetime import datetime

    otel, exporter = otel_with_exporter
    server_span = server_span_factory(path)
    kwargs = {"litellm_params": {"metadata": {"litellm_parent_otel_span": server_span}}}
    otel._end_proxy_span_from_kwargs(kwargs, datetime.now())

    assert_server_span_attrs(
        exporter,
        expected_status=200,
        expected_url_path=path,
        where=f"{path} _end_proxy_span_from_kwargs",
    )


# Bare TypeError has no .code/.status_code, so error_information.error_code is
# empty and _record_exception_on_span skips the stamp — must default to 500.
def test_async_post_call_failure_hook_defaults_to_500(
    otel_with_exporter, server_span_factory
):
    otel, exporter = otel_with_exporter
    server_span = server_span_factory("/v1/responses")
    uakd = _real_user_api_key_dict(server_span)

    asyncio.run(
        otel.async_post_call_failure_hook(
            request_data={},
            original_exception=TypeError("missing required argument"),
            user_api_key_dict=uakd,
        )
    )

    assert_server_span_attrs(
        exporter,
        expected_status=500,
        expected_url_path="/v1/responses",
        where="async_post_call_failure_hook (TypeError) defaults to 500",
    )


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
