"""LIT-3443 — passthrough success spans must hang off the SERVER root span.

_init_kwargs_for_pass_through_endpoint is the single place both passthrough
paths get their logging metadata, and update_environment_variables copies that
metadata onto the logging object's model_call_details — which is exactly what
the OTEL success handler reads. So wiring the parent span in there once fixes
both the non-streaming and streaming paths; the streaming handler rebuilds its
kwargs from raw SSE bytes and never sees that metadata, but it doesn't need to.

These tests drive the real passthrough logging code into the real OpenTelemetry
success handler, capturing every span in an InMemorySpanExporter:

  * non-streaming: _init_kwargs_for_pass_through_endpoint -> async_success_handler
  * streaming:     _route_streaming_logging_to_handler over real Anthropic SSE

Before the fix the parent span is never wired in, so the litellm_request span
orphans into its own trace and the SERVER root span is never ended. Each test
asserts the SERVER root is exported (ended) and that nothing escapes into a
foreign trace; the USE_OTEL_LITELLM_REQUEST_SPAN variants additionally assert
the litellm_request child is parented to the SERVER root.
"""

import asyncio
from datetime import datetime
from typing import Optional, Tuple

import pytest
from starlette.requests import Request

import litellm
from litellm.integrations.opentelemetry import LITELLM_PROXY_REQUEST_SPAN_NAME
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.pass_through_endpoints.pass_through_endpoints import (
    HttpPassThroughEndpointHelpers,
)
from litellm.proxy.pass_through_endpoints.streaming_handler import (
    PassThroughStreamingHandler,
)
from litellm.proxy.pass_through_endpoints.success_handler import (
    PassThroughEndpointLogging,
)
from litellm.types.passthrough_endpoints.pass_through_endpoints import (
    EndpointType,
    PassthroughStandardLoggingPayload,
)
from litellm.types.utils import Choices, Message, ModelResponse, Usage

URL_ROUTE = "https://api.anthropic.com/v1/messages"
MODEL = "claude-sonnet-4-5-20250929"


@pytest.fixture
def otel_success_callback(otel_with_exporter, monkeypatch):
    """Register our in-memory OTEL instance where async_success_handler looks
    for success callbacks (litellm._async_success_callback), so the real
    logging path drives it."""
    otel, exporter = otel_with_exporter
    monkeypatch.setattr(litellm, "callbacks", [otel])
    monkeypatch.setattr(litellm, "_async_success_callback", [otel])
    return otel, exporter


def _make_request() -> Request:
    return Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/anthropic/v1/messages",
            "raw_path": b"/anthropic/v1/messages",
            "query_string": b"",
            "headers": [(b"content-type", b"application/json")],
            "scheme": "http",
            "server": ("testserver", 80),
            "client": ("testclient", 50000),
        }
    )


def _build_logging_obj_wired_to_root(
    root_span, *, stream: bool, extra_body: Optional[dict] = None
) -> Tuple[LiteLLMLoggingObj, dict, datetime]:
    """Mirror pass_through_endpoints.py: build the logging object and run the
    real _init_kwargs + update_environment_variables so the parent span lands
    on model_call_details exactly the way production wires it."""
    request = _make_request()
    body = {"model": MODEL, "messages": [{"role": "user", "content": "hi"}]}
    if extra_body:
        body.update(extra_body)
    user_api_key_dict = UserAPIKeyAuth(api_key="sk-test", parent_otel_span=root_span)
    start_time = datetime.now()
    logging_obj = LiteLLMLoggingObj(
        model="unknown",
        messages=[{"role": "user", "content": "hi"}],
        stream=stream,
        call_type="pass_through_endpoint",
        start_time=start_time,
        litellm_call_id="lit-3443-call",
        function_id="1245",
    )
    payload = PassthroughStandardLoggingPayload(
        url=URL_ROUTE, request_body=body, request_method="POST"
    )
    kwargs = HttpPassThroughEndpointHelpers._init_kwargs_for_pass_through_endpoint(
        request=request,
        user_api_key_dict=user_api_key_dict,
        passthrough_logging_payload=payload,
        logging_obj=logging_obj,
        _parsed_body=body,
        litellm_call_id="lit-3443-call",
    )
    logging_obj.update_environment_variables(
        model="unknown",
        user="unknown",
        optional_params={},
        litellm_params=kwargs["litellm_params"],
        call_type="pass_through_endpoint",
    )
    logging_obj.model_call_details["litellm_call_id"] = "lit-3443-call"
    return logging_obj, kwargs, start_time


def _model_response() -> ModelResponse:
    resp = ModelResponse()
    resp.model = MODEL
    resp.choices = [Choices(message=Message(role="assistant", content="hi there"))]
    resp.usage = Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
    return resp


# Real Anthropic SSE stream (single text block) reused for the streaming path.
STREAM_CHUNKS = [
    "event: message_start",
    'data: {"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":"claude-sonnet-4-5-20250929","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":17,"output_tokens":5}}}',
    "event: content_block_start",
    'data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}',
    "event: content_block_delta",
    'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello world"}}',
    "event: content_block_stop",
    'data: {"type":"content_block_stop","index":0}',
    "event: message_delta",
    'data: {"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":2}}',
    "event: message_stop",
    'data: {"type":"message_stop"}',
]


def _assert_root_closed_and_no_orphan(exporter, root_span, where):
    finished = exporter.get_finished_spans()
    root_ctx = root_span.get_span_context()

    server_spans = [s for s in finished if s.name == LITELLM_PROXY_REQUEST_SPAN_NAME]
    assert server_spans, (
        f"{where}: SERVER root span was never ended/exported — exporter saw "
        f"{[s.name for s in finished]}"
    )

    foreign = [s for s in finished if s.context.trace_id != root_ctx.trace_id]
    assert not foreign, (
        f"{where}: span(s) orphaned into a foreign trace: "
        f"{[(s.name, hex(s.context.trace_id)) for s in foreign]} "
        f"(root trace={hex(root_ctx.trace_id)})"
    )


def _assert_child_parented_to_root(exporter, root_span, where):
    finished = exporter.get_finished_spans()
    root_ctx = root_span.get_span_context()
    children = [
        s
        for s in finished
        if s.name != LITELLM_PROXY_REQUEST_SPAN_NAME
        and s.parent is not None
        and s.parent.span_id == root_ctx.span_id
    ]
    assert children, (
        f"{where}: no litellm_request child parented to the SERVER root — "
        f"finished={[(s.name, s.parent and hex(s.parent.span_id)) for s in finished]}"
    )
    for child in children:
        assert child.context.trace_id == root_ctx.trace_id, (
            f"{where}: child {child.name} in trace {hex(child.context.trace_id)}, "
            f"expected root trace {hex(root_ctx.trace_id)}"
        )


@pytest.mark.parametrize("use_request_span", [False, True])
def test_non_streaming_passthrough_links_to_server_root(
    otel_success_callback,
    server_span_factory,
    monkeypatch,
    use_request_span,
):
    if use_request_span:
        monkeypatch.setenv("USE_OTEL_LITELLM_REQUEST_SPAN", "true")
    _otel, exporter = otel_success_callback
    root = server_span_factory("/anthropic/v1/messages")

    logging_obj, kwargs, start_time = _build_logging_obj_wired_to_root(
        root, stream=False
    )
    end_time = datetime.now()
    asyncio.run(
        logging_obj.async_success_handler(
            result=_model_response(),
            start_time=start_time,
            end_time=end_time,
            cache_hit=False,
            **kwargs,
        )
    )

    where = f"non-streaming (use_request_span={use_request_span})"
    _assert_root_closed_and_no_orphan(exporter, root, where)
    if use_request_span:
        _assert_child_parented_to_root(exporter, root, where)


@pytest.mark.parametrize("use_request_span", [False, True])
def test_streaming_passthrough_links_to_server_root(
    otel_success_callback,
    server_span_factory,
    monkeypatch,
    use_request_span,
):
    if use_request_span:
        monkeypatch.setenv("USE_OTEL_LITELLM_REQUEST_SPAN", "true")
    _otel, exporter = otel_success_callback
    root = server_span_factory("/anthropic/v1/messages")

    logging_obj, _kwargs, start_time = _build_logging_obj_wired_to_root(
        root, stream=True
    )
    raw_bytes = ["\n".join(STREAM_CHUNKS).encode("utf-8")]
    end_time = datetime.now()
    asyncio.run(
        PassThroughStreamingHandler._route_streaming_logging_to_handler(
            litellm_logging_obj=logging_obj,
            passthrough_success_handler_obj=PassThroughEndpointLogging(),
            url_route=URL_ROUTE,
            request_body={"model": MODEL, "stream": True},
            endpoint_type=EndpointType.ANTHROPIC,
            start_time=start_time,
            raw_bytes=raw_bytes,
            end_time=end_time,
        )
    )

    where = f"streaming (use_request_span={use_request_span})"
    _assert_root_closed_and_no_orphan(exporter, root, where)
    if use_request_span:
        _assert_child_parented_to_root(exporter, root, where)


def test_client_body_metadata_cannot_clobber_parent_span(
    otel_success_callback,
    server_span_factory,
    monkeypatch,
):
    """A passthrough request body whose metadata mirrors the internal
    litellm_parent_otel_span key must not override the real parent span. The
    internal span is wired after the client-metadata merge, so the SERVER root
    still links and closes. With the old ordering the JSON scalar would win and
    the litellm_request span would orphan."""
    monkeypatch.setenv("USE_OTEL_LITELLM_REQUEST_SPAN", "true")
    _otel, exporter = otel_success_callback
    root = server_span_factory("/anthropic/v1/messages")

    logging_obj, kwargs, start_time = _build_logging_obj_wired_to_root(
        root,
        stream=False,
        extra_body={"metadata": {"litellm_parent_otel_span": "not-a-real-span"}},
    )
    end_time = datetime.now()
    asyncio.run(
        logging_obj.async_success_handler(
            result=_model_response(),
            start_time=start_time,
            end_time=end_time,
            cache_hit=False,
            **kwargs,
        )
    )

    where = "client-metadata-clobber"
    _assert_root_closed_and_no_orphan(exporter, root, where)
    _assert_child_parented_to_root(exporter, root, where)


def test_init_kwargs_internal_keys_resist_client_metadata(server_span_factory):
    """Deterministic contract test on _init_kwargs_for_pass_through_endpoint:
    a request body whose metadata mirrors the internal user_api_key and
    litellm_parent_otel_span keys must not override the authenticated values.
    Pure dict assertion, no async or OTEL execution. Fails on the old ordering
    where the client values were merged in last."""
    real_span = server_span_factory("/anthropic/v1/messages")
    user_api_key_dict = UserAPIKeyAuth(
        api_key="sk-real-key", parent_otel_span=real_span
    )
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": "hi"}],
        "metadata": {
            "user_api_key": "sk-SPOOFED",
            "litellm_parent_otel_span": "not-a-real-span",
        },
    }
    logging_obj = LiteLLMLoggingObj(
        model="unknown",
        messages=[{"role": "user", "content": "hi"}],
        stream=False,
        call_type="pass_through_endpoint",
        start_time=datetime.now(),
        litellm_call_id="lit-3443-clobber",
        function_id="1245",
    )
    payload = PassthroughStandardLoggingPayload(
        url=URL_ROUTE, request_body=body, request_method="POST"
    )
    kwargs = HttpPassThroughEndpointHelpers._init_kwargs_for_pass_through_endpoint(
        request=_make_request(),
        user_api_key_dict=user_api_key_dict,
        passthrough_logging_payload=payload,
        logging_obj=logging_obj,
        _parsed_body=body,
        litellm_call_id="lit-3443-clobber",
    )
    md = kwargs["litellm_params"]["metadata"]
    # api_key is stored hashed on the auth object; the authenticated value must
    # win over the client-supplied spoof.
    assert md["user_api_key"] == user_api_key_dict.api_key
    assert md["user_api_key"] != "sk-SPOOFED"
    assert md["litellm_parent_otel_span"] is real_span
