"""OTel v2 Arize AX and Phoenix: the request root span carries the request's input and output, which both show as the
trace's, when the OpenInference logger's provider started that root."""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import httpx
import pytest
from _openinference_support import (
    RESPONSES_TOOLS,
    _assert_chat_request,
    _attribute_text,
    _chat_caller_response,
    _chat_caller_stream,
    _chat_plain_response,
    _chat_stream_response,
    _chat_tool_call,
    _json_messages,
    _json_object,
    _normalize_chat_caller_stream,
    _responses_response,
    _rig,
    _sse_json_values,
)
from integration._support.client import Gateway
from integration._support.wire import Reply, Request, Wire, wire_server
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest
from pydantic import JsonValue
from test_presidio_streaming_output import MASK, PERSON, analyzer, anonymizer, openai_frame

_ROOT_IO_KEYS: Final = frozenset(
    {"openinference.span.kind", "input.value", "input.mime_type", "output.value", "output.mime_type"}
)
_PROMPT: Final = [{"role": "user", "content": "weather in Paris?"}]
_COLLECTOR: Final = {"otel": {"exporter": "http/protobuf", "endpoint": "unused"}}


@dataclass(frozen=True, slots=True)
class _Span:
    trace_id: bytes
    parent_span_id: bytes
    attributes: Mapping[str, str]


def _decoded_spans(requests: tuple[Request, ...]) -> Iterator[_Span]:
    for request in requests:
        if request.headers.get("content-type") != "application/x-protobuf":
            continue
        batch: ExportTraceServiceRequest = ExportTraceServiceRequest.FromString(request.body)
        for resource_spans in batch.resource_spans:
            for scope_spans in resource_spans.scope_spans:
                for span in scope_spans.spans:
                    yield _Span(
                        trace_id=span.trace_id,
                        parent_span_id=span.parent_span_id,
                        attributes={attribute.key: _attribute_text(attribute.value) for attribute in span.attributes},
                    )


def _request_trace(destination: Wire, marker: str) -> tuple[_Span, _Span]:
    """The trace root and the OpenInference LLM span of the request tagged ``marker``, once both have arrived."""

    def found(spans: tuple[_Span, ...]) -> tuple[_Span, _Span] | None:
        llm: Final = next(
            (
                span
                for span in spans
                if span.attributes.get("openinference.span.kind") == "LLM"
                and span.attributes.get("litellm.metadata.trace_marker") == marker
            ),
            None,
        )
        if llm is None:
            return None
        roots: Final = tuple(span for span in spans if span.trace_id == llm.trace_id and not span.parent_span_id)
        assert len(roots) <= 1, roots
        return (roots[0], llm) if roots else None

    def collect(previous: tuple[_Span, ...], deadline: float) -> tuple[_Span, _Span]:
        current: Final = (*previous, *_decoded_spans(destination.drain()))
        trace: Final = found(current)
        if trace is not None:
            return trace
        assert time.monotonic() < deadline, "\n".join(
            f"{span.trace_id.hex()[:8]} {span.parent_span_id.hex()[:8] or 'root'} {dict(span.attributes)}"
            for span in current
            if "db.system" not in span.attributes
        )
        time.sleep(0.25)
        return collect(current, deadline)

    return collect((), time.monotonic() + 30)


def _chat_root(destination: Wire) -> _Span:
    """The root span of the rig's one chat completions request, once it has arrived."""

    def collect(previous: tuple[_Span, ...], deadline: float) -> _Span:
        current: Final = (*previous, *_decoded_spans(destination.drain()))
        roots: Final = tuple(
            span
            for span in current
            if not span.parent_span_id and span.attributes.get("http.route") == "/v1/chat/completions"
        )
        assert len(roots) <= 1, roots
        if roots:
            return roots[0]
        assert time.monotonic() < deadline, "\n".join(
            f"{span.trace_id.hex()[:8]} {span.parent_span_id.hex()[:8] or 'root'} {dict(span.attributes)}"
            for span in current
            if "db.system" not in span.attributes
        )
        time.sleep(0.25)
        return collect(current, deadline)

    return collect((), time.monotonic() + 30)


def _chat_request(
    proxy: Gateway,
    model: str,
    marker: str,
    *,
    stream: bool = False,
    no_log: bool = False,
    headers: Mapping[str, str] | None = None,
) -> httpx.Response:
    return proxy.request(
        "POST",
        "/v1/chat/completions",
        {
            "model": model,
            "messages": _PROMPT,
            "metadata": {"trace_marker": marker},
            "cache": {"no-cache": True},
            **({"stream": True, "stream_options": {"include_usage": True}} if stream else {}),
            **({"no-log": True} if no_log else {}),
        },
        headers=headers,
    )


def _plain_upstream(marker: str) -> tuple[Reply, Callable[[Request], Reply]]:
    reply: Final = _chat_plain_response(marker, "sunny in Paris " + marker)

    def upstream(request: Request) -> Reply:
        _assert_chat_request(request, messages=_PROMPT, include_tools=False)
        return reply

    return reply, upstream


def _assert_root_carries_chat_io(root: _Span, marker: str) -> None:
    attributes: Final = root.attributes
    assert attributes["openinference.span.kind"] == "CHAIN", attributes
    assert _json_messages(attributes["input.value"]) == _PROMPT, attributes
    assert attributes["input.mime_type"] == "application/json", attributes
    assert [(turn["role"], turn["content"]) for turn in _json_messages(attributes["output.value"])] == [
        ("assistant", "sunny in Paris " + marker)
    ], attributes
    assert attributes["output.mime_type"] == "application/json", attributes


def _assert_root_left_alone(root: _Span) -> None:
    assert _ROOT_IO_KEYS.isdisjoint(root.attributes), root.attributes


@pytest.mark.parametrize("callback", ("arize_phoenix", "arize"))
def test_arize_and_phoenix_root_carries_the_chat_input_and_output(
    callback: str, gateway: Gateway, tmp_path: Path
) -> None:
    marker: Final = f"root-io-{callback}-" + uuid.uuid4().hex
    reply, upstream = _plain_upstream(marker)

    with _rig(gateway, tmp_path, upstream, callbacks=(callback,), callback_settings=_COLLECTOR) as rig:
        response: Final = _chat_request(rig.proxy, rig.model, marker)
        assert response.status_code == 200, response.text
        assert _json_object(response.content) == _chat_caller_response(reply, rig.model), response.text
        root, llm = _request_trace(rig.destination, marker)

    _assert_root_carries_chat_io(root, marker)
    assert llm.parent_span_id, llm


def test_phoenix_root_carries_the_assembled_stream_output(gateway: Gateway, tmp_path: Path) -> None:
    marker: Final = "root-io-stream-" + uuid.uuid4().hex
    reply: Final = _chat_stream_response(marker, (_chat_tool_call(marker),))

    def upstream(request: Request) -> Reply:
        _assert_chat_request(
            request,
            messages=_PROMPT,
            include_tools=False,
            stream=True,
            stream_options={"include_usage": True},
        )
        return reply

    with _rig(gateway, tmp_path, upstream, callbacks=("arize_phoenix",), callback_settings=_COLLECTOR) as rig:
        response: Final = _chat_request(rig.proxy, rig.model, marker, stream=True)
        assert response.status_code == 200, response.text
        assert _normalize_chat_caller_stream(_sse_json_values(response.content)) == _chat_caller_stream(
            reply, rig.model
        ), response.text
        root, _ = _request_trace(rig.destination, marker)

    assert root.attributes["openinference.span.kind"] == "CHAIN", root.attributes
    assert _json_messages(root.attributes["input.value"]) == _PROMPT, root.attributes
    output: Final = _json_messages(root.attributes["output.value"])
    assert [(turn["role"], turn["tool_calls"]) for turn in output] == [
        (
            "assistant",
            [
                {
                    "id": "call_" + marker,
                    "type": "function",
                    "function": {"name": "lookup_weather", "arguments": '{"city": "Paris"}'},
                }
            ],
        )
    ], root.attributes


def test_phoenix_root_carries_the_responses_input_with_instructions(gateway: Gateway, tmp_path: Path) -> None:
    marker: Final = "root-io-responses-" + uuid.uuid4().hex

    def upstream(request: Request) -> Reply:
        assert _json_object(request.body)["instructions"] == "be terse", request
        return _responses_response(marker)

    with _rig(gateway, tmp_path, upstream, callbacks=("arize_phoenix",), callback_settings=_COLLECTOR) as rig:
        response: Final = rig.proxy.request(
            "POST",
            "/v1/responses",
            {
                "model": rig.model,
                "instructions": "be terse",
                "input": "weather in Paris?",
                "tools": RESPONSES_TOOLS,
                "tool_choice": {"type": "function", "name": "lookup_weather"},
                "metadata": {"trace_marker": marker},
                "cache": {"no-cache": True},
            },
        )
        assert response.status_code == 200, response.text
        root, _ = _request_trace(rig.destination, marker)

    assert root.attributes["openinference.span.kind"] == "CHAIN", root.attributes
    assert _json_messages(root.attributes["input.value"]) == [
        {"role": "system", "content": "be terse"},
        {"role": "user", "content": "weather in Paris?"},
    ], root.attributes
    output: Final = _json_messages(root.attributes["output.value"])
    assert [(item["type"], item["call_id"], item["arguments"]) for item in output] == [
        ("function_call", "call_" + marker, '{"city": "Paris"}')
    ], root.attributes


def test_phoenix_listed_first_stamps_the_root_the_generic_collector_also_receives(
    gateway: Gateway, tmp_path: Path
) -> None:
    marker: Final = "root-io-folded-" + uuid.uuid4().hex
    _, upstream = _plain_upstream(marker)

    with _rig(gateway, tmp_path, upstream, callbacks=("arize_phoenix", "otel"), callback_settings=_COLLECTOR) as rig:
        response: Final = _chat_request(rig.proxy, rig.model, marker)
        assert response.status_code == 200, response.text
        root, _ = _request_trace(rig.destination, marker)

    _assert_root_carries_chat_io(root, marker)


@pytest.mark.parametrize(
    ("callbacks", "environment"),
    [
        (("otel", "arize_phoenix"), {}),
        (("arize_phoenix",), {"OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT": "no_content"}),
        (("weave_otel",), {"WANDB_BASE_URL": "http://127.0.0.1"}),
    ],
    ids=["root-owned-by-generic-otel", "content-capture-off", "weave"],
)
def test_root_is_left_alone(
    callbacks: tuple[str, ...], environment: dict[str, str], gateway: Gateway, tmp_path: Path
) -> None:
    marker: Final = "root-io-alone-" + uuid.uuid4().hex
    _, upstream = _plain_upstream(marker)

    with _rig(
        gateway, tmp_path, upstream, callbacks=callbacks, callback_settings=_COLLECTOR, environment=environment
    ) as rig:
        response: Final = _chat_request(rig.proxy, rig.model, marker)
        assert response.status_code == 200, response.text
        root, llm = _request_trace(rig.destination, marker)

    _assert_root_left_alone(root)
    assert llm.parent_span_id, llm


@pytest.mark.parametrize(
    ("litellm_settings", "headers", "stream"),
    [
        ({"turn_off_message_logging": True}, None, False),
        ({"turn_off_message_logging": True}, None, True),
        (None, {"x-litellm-enable-message-redaction": "true"}, False),
    ],
    ids=["turn-off-message-logging", "turn-off-message-logging-stream", "redaction-header"],
)
def test_root_is_left_alone_when_the_request_messages_are_redacted(
    litellm_settings: Mapping[str, JsonValue] | None,
    headers: Mapping[str, str] | None,
    stream: bool,
    gateway: Gateway,
    tmp_path: Path,
) -> None:
    marker: Final = "root-io-redacted-" + uuid.uuid4().hex
    reply: Final = (
        _chat_stream_response(marker, (_chat_tool_call(marker),)) if stream else _chat_plain_response(marker, marker)
    )

    with _rig(
        gateway,
        tmp_path,
        lambda _request: reply,
        callbacks=("arize_phoenix",),
        callback_settings=_COLLECTOR,
        litellm_settings=litellm_settings,
    ) as rig:
        response: Final = _chat_request(rig.proxy, rig.model, marker, stream=stream, headers=headers)
        assert response.status_code == 200, response.text
        root, llm = _request_trace(rig.destination, marker)

    _assert_root_left_alone(root)
    assert _json_messages(llm.attributes["input.value"]) == [{"role": "user", "content": "redacted-by-litellm"}], llm


@pytest.mark.parametrize(
    ("callbacks", "callback_settings", "no_log"),
    [
        (("arize_phoenix",), _COLLECTOR, True),
        (
            ("otel",),
            {
                "otel": {
                    **_COLLECTOR["otel"],
                    "mapper_names": ["genai", "openinference"],
                    "turn_off_message_logging": True,
                }
            },
            False,
        ),
    ],
    ids=["no-log-request", "callback-turns-off-message-logging"],
)
def test_root_is_left_alone_when_the_messages_stay_off_the_llm_span(
    callbacks: tuple[str, ...],
    callback_settings: Mapping[str, JsonValue],
    no_log: bool,
    gateway: Gateway,
    tmp_path: Path,
) -> None:
    marker: Final = "root-io-withheld-" + uuid.uuid4().hex
    _, upstream = _plain_upstream(marker)

    with _rig(gateway, tmp_path, upstream, callbacks=callbacks, callback_settings=callback_settings) as rig:
        response: Final = _chat_request(rig.proxy, rig.model, marker, no_log=no_log)
        assert response.status_code == 200, response.text
        root: Final = _chat_root(rig.destination)

    _assert_root_left_alone(root)


def test_phoenix_stream_root_carries_the_reply_a_later_masking_guardrail_served(
    gateway: Gateway, tmp_path: Path
) -> None:
    marker: Final = "root-io-masked-stream-" + uuid.uuid4().hex
    reply: Final = Reply(
        content_type="text/event-stream",
        chunks=(
            openai_frame(marker, {"role": "assistant", "content": f"{PERSON} says hi"}),
            openai_frame(marker, {}, "stop"),
            b"data: [DONE]\n\n",
        ),
    )

    with wire_server(analyzer) as analyze_sink, wire_server(anonymizer) as anonymize_sink:
        guardrail: Final[Mapping[str, JsonValue]] = {
            "guardrail_name": "presidio-" + marker,
            "litellm_params": {
                "guardrail": "presidio",
                "mode": "post_call",
                "default_on": True,
                "presidio_analyzer_api_base": analyze_sink.url,
                "presidio_anonymizer_api_base": anonymize_sink.url,
                "presidio_filter_scope": "output",
            },
        }
        with _rig(
            gateway,
            tmp_path,
            lambda _request: reply,
            callbacks=("arize_phoenix",),
            callback_settings=_COLLECTOR,
            guardrails=(guardrail,),
        ) as rig:
            response: Final = _chat_request(rig.proxy, rig.model, marker, stream=True)
            assert response.status_code == 200, response.text
            assert PERSON not in response.text and MASK in response.text, response.text
            root, _ = _request_trace(rig.destination, marker)

    output: Final = _json_messages(root.attributes["output.value"])
    assert [(turn["role"], turn["content"]) for turn in output] == [("assistant", f"{MASK} says hi")], root.attributes
