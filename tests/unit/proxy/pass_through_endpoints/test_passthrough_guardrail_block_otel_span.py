"""Regression: a guardrail block on a passthrough endpoint must still emit the
otel guardrail span.

The span is emitted from the guardrail-recording path the moment a guardrail
finishes (``add_standard_logging_guardrail_information_to_request_data`` ->
``emit_guardrail_span``), routed through the proxy's registered otel V2 logger,
rather than from a post-call hook that does not fire on every path. A block
raises out of the post-call hook before any later hook runs, so the recording
path is the only place the span is reliably produced. These tests drive the real
``pass_through_request`` with a real ``ProxyLogging`` + a real otel V2 logger
registered as the proxy's ``open_telemetry_logger`` and assert the span is
emitted on both allow and block.
"""

import json
from contextlib import ExitStack
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi import HTTPException

pytest.importorskip("opentelemetry")

from opentelemetry.sdk.trace.export.in_memory_span_exporter import (  # noqa: E402
    InMemorySpanExporter,
)

import litellm  # noqa: E402
from litellm.caching.dual_cache import DualCache  # noqa: E402
from litellm.integrations.custom_guardrail import (  # noqa: E402
    CustomGuardrail,
    log_guardrail_information,
)
from litellm.integrations.otel.logger import OpenTelemetryV2  # noqa: E402
from litellm.integrations.otel.model.config import OpenTelemetryV2Config  # noqa: E402
from litellm.integrations.otel.plumbing import providers  # noqa: E402
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache  # noqa: E402
from litellm.proxy.utils import ProxyLogging  # noqa: E402
from litellm.types.guardrails import GuardrailEventHooks  # noqa: E402

_PT_MOD = "litellm.proxy.pass_through_endpoints.pass_through_endpoints"
_COLLECT = (
    "litellm.proxy.pass_through_endpoints.passthrough_guardrails."
    "PassthroughGuardrailHandler.collect_guardrails"
)
_GUARDRAIL_SPAN = "execute_guardrail block-demo"
_TRIGGER = "BLOCKME"

# pass_through_endpoints imports proxy_server lazily (inside the request
# function), so importing this at module scope does not require the real
# proxy_server and does not mutate sys.modules.
from litellm.proxy.pass_through_endpoints.pass_through_endpoints import (  # noqa: E402
    pass_through_request,
)


class _BlockOnTextGuardrail(CustomGuardrail):
    """Denies (HTTP 400) when the response carries the trigger word; records its
    standard guardrail logging info on both allow and block via the decorator."""

    @log_guardrail_information
    async def async_post_call_success_hook(self, data, user_api_key_dict, response):
        if _TRIGGER in json.dumps(response):
            raise HTTPException(
                status_code=400, detail={"error": "blocked by block-demo guardrail"}
            )
        return response


def _user_api_key_dict():
    d = MagicMock()
    d.api_key = "sk-test"
    d.user_id = "user-1"
    d.team_id = "team-1"
    d.org_id = None
    d.metadata = {}
    d.team_metadata = {}
    d.parent_otel_span = None
    d.request_route = "/mock/echo"
    return d


def _mock_request():
    r = MagicMock()
    r.method = "POST"
    r.query_params = {}
    r.url = "http://testserver/mock/echo"
    headers = MagicMock()
    headers.copy.return_value = {}
    r.headers = headers
    return r


def _httpx_response(text: str) -> httpx.Response:
    body = {"candidates": [{"content": {"role": "model", "parts": [{"text": text}]}}]}
    return httpx.Response(
        status_code=200,
        headers={"content-type": "application/json"},
        content=json.dumps(body).encode("utf-8"),
        request=httpx.Request("POST", "https://upstream.example/echo"),
    )


def _otel_logger_with_exporter():
    cfg = OpenTelemetryV2Config(exporter="in_memory")
    exporter = InMemorySpanExporter()
    tracer_provider = providers.build_tracer_provider(cfg, exporter=exporter)
    return OpenTelemetryV2(config=cfg, tracer_provider=tracer_provider), exporter


def _guardrail_span_names(exporter):
    return [
        s.name
        for s in exporter.get_finished_spans()
        if s.name.startswith("execute_guardrail")
    ]


async def _drive(response_text: str):
    """Run the real pass_through_request with the block-demo guardrail + otel V2
    logger registered, returning (status_code, guardrail_span_names)."""
    otel, exporter = _otel_logger_with_exporter()
    guardrail = _BlockOnTextGuardrail(
        guardrail_name="block-demo", event_hook=[GuardrailEventHooks.post_call]
    )
    proxy_logging = ProxyLogging(user_api_key_cache=UserApiKeyCache(DualCache()))

    saved_callbacks = list(litellm.callbacks)
    litellm.callbacks = [guardrail, otel]

    mock_async_client_obj = MagicMock()
    mock_async_client_obj.client = AsyncMock()
    mock_pt_logging = MagicMock()
    mock_pt_logging.pass_through_async_success_handler = AsyncMock()

    patches = [
        patch(
            f"{_PT_MOD}.HttpPassThroughEndpointHelpers.non_streaming_http_request_handler",
            new_callable=AsyncMock,
            return_value=_httpx_response(response_text),
        ),
        patch(f"{_PT_MOD}._is_streaming_response", return_value=False),
        patch("litellm.proxy.proxy_server.proxy_logging_obj", proxy_logging),
        patch("litellm.proxy.proxy_server.open_telemetry_logger", otel),
        patch("litellm.proxy.proxy_server.llm_router", None),
        patch(f"{_PT_MOD}.pass_through_endpoint_logging", mock_pt_logging),
        patch(f"{_PT_MOD}.get_async_httpx_client", return_value=mock_async_client_obj),
        patch(f"{_PT_MOD}._read_request_body", new_callable=AsyncMock, return_value={}),
        patch(f"{_PT_MOD}._safe_get_request_headers", return_value={}),
        patch(_COLLECT, return_value=["block-demo"]),
    ]
    try:
        with ExitStack() as stack:
            for p in patches:
                stack.enter_context(p)
            try:
                result = await pass_through_request(
                    request=_mock_request(),
                    target="https://upstream.example/echo",
                    custom_headers={"Content-Type": "application/json"},
                    user_api_key_dict=_user_api_key_dict(),
                    stream=False,
                )
                # A deny (HTTP 4xx) re-raises as ProxyException; an allow returns
                # the upstream Response.
                status_code = result.status_code
            except Exception as e:
                status_code = getattr(e, "code", None) or getattr(
                    e, "status_code", None
                )
        return int(status_code), _guardrail_span_names(exporter)
    finally:
        litellm.callbacks = saved_callbacks


@pytest.mark.asyncio
async def test_guardrail_block_emits_otel_guardrail_span():
    status_code, span_names = await _drive(f"{_TRIGGER} please")
    assert status_code == 400
    assert span_names == [_GUARDRAIL_SPAN], (
        "guardrail span must be emitted when a passthrough guardrail blocks, "
        f"got spans: {span_names}"
    )


@pytest.mark.asyncio
async def test_guardrail_allow_emits_otel_guardrail_span():
    status_code, span_names = await _drive("hello world")
    assert status_code == 200
    assert span_names == [_GUARDRAIL_SPAN]
