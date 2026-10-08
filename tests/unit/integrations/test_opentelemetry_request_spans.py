import asyncio
import json
from collections.abc import Sequence
from typing import Final

import httpx
import pytest
import respx
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExportResult
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

import litellm
from litellm.integrations.opentelemetry import OpenTelemetry, OpenTelemetryConfig

_OPENAI_URL: Final = "https://api.openai.com/v1/chat/completions"
_EXPECTED_SPAN_NAMES: Final = ("litellm_request", "raw_gen_ai_request")
_USER: Final = "OTEL_USER"
_USAGE: Final = {"prompt_tokens": 8, "completion_tokens": 2, "total_tokens": 10}
_COMPLETION: Final = {
    "id": "chatcmpl-otel",
    "object": "chat.completion",
    "created": 1700000000,
    "model": "gpt-4.1-mini-2025-04-14",
    "service_tier": "default",
    "system_fingerprint": "fp_otel",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "hello"}, "finish_reason": "stop"}],
    "usage": _USAGE,
}
_STREAM: Final = (
    "".join(
        f"data: {json.dumps(chunk)}\n\n"
        for chunk in (
            {
                "id": "chatcmpl-otel",
                "object": "chat.completion.chunk",
                "created": 1700000000,
                "model": "gpt-4.1-mini-2025-04-14",
                "choices": [{"index": 0, "delta": {"role": "assistant", "content": "hello"}, "finish_reason": None}],
            },
            {
                "id": "chatcmpl-otel",
                "object": "chat.completion.chunk",
                "created": 1700000000,
                "model": "gpt-4.1-mini-2025-04-14",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "usage": _USAGE,
            },
        )
    )
    + "data: [DONE]\n\n"
)
_LITELLM_REQUEST_ATTRIBUTES: Final = (
    "gen_ai.request.model",
    "gen_ai.system",
    "gen_ai.request.temperature",
    "llm.is_streaming",
    "llm.user",
    "gen_ai.response.id",
    "gen_ai.response.model",
    "gen_ai.usage.total_tokens",
    "gen_ai.usage.output_tokens",
    "gen_ai.usage.input_tokens",
)
_RAW_STREAMING_ATTRIBUTES: Final = (
    "llm.openai.messages",
    "llm.openai.temperature",
    "llm.openai.user",
    "llm.openai.extra_body",
    "llm.openai.model",
)
_RAW_NON_STREAMING_ATTRIBUTES: Final = (
    *_RAW_STREAMING_ATTRIBUTES,
    "llm.openai.id",
    "llm.openai.choices",
    "llm.openai.created",
    "llm.openai.object",
    "llm.openai.service_tier",
    "llm.openai.system_fingerprint",
    "llm.openai.usage",
)


def _is_our_request(span: ReadableSpan) -> bool:
    return span.name == "litellm_request" and (span.attributes or {}).get("llm.user") == _USER


def _trace_id(span: ReadableSpan) -> int:
    assert span.context is not None
    return span.context.trace_id


class _SignallingExporter(InMemorySpanExporter):
    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        super().__init__()
        self.loop: Final = loop
        self.request_span_exported: Final = asyncio.Event()

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        result: Final = super().export(spans)
        if any(_is_our_request(span) for span in spans):
            self.loop.call_soon_threadsafe(self.request_span_exported.set)
        return result


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [True, False])
async def test_otel_callback_emits_the_request_and_raw_provider_spans(
    streaming: bool, monkeypatch: pytest.MonkeyPatch, respx_mock: respx.MockRouter
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.delenv("OTEL_SEMCONV_STABILITY_OPT_IN", raising=False)
    exporter: Final = _SignallingExporter(asyncio.get_running_loop())
    tracer_provider: Final = TracerProvider()
    tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(
        litellm,
        "callbacks",
        [OpenTelemetry(config=OpenTelemetryConfig(exporter=exporter), tracer_provider=tracer_provider)],
    )
    respx_mock.post(_OPENAI_URL).mock(
        return_value=httpx.Response(200, text=_STREAM, headers={"content-type": "text/event-stream"})
        if streaming
        else httpx.Response(200, json=_COMPLETION)
    )

    response: Final = await litellm.acompletion(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": "hi"}],
        temperature=0.1,
        user=_USER,
        stream=streaming,
        api_key="sk-unit-test",
    )
    if streaming:
        assert [chunk async for chunk in response]
    await asyncio.wait_for(exporter.request_span_exported.wait(), timeout=10)

    finished: Final = exporter.get_finished_spans()
    request_span: Final = next(span for span in finished if _is_our_request(span))
    ours: Final = tuple(span for span in finished if _trace_id(span) == _trace_id(request_span))
    assert tuple(sorted(span.name for span in ours)) == _EXPECTED_SPAN_NAMES
    spans: Final = {span.name: span for span in ours}
    request_attributes: Final = spans["litellm_request"].attributes or {}
    assert all(request_attributes.get(name) is not None for name in _LITELLM_REQUEST_ATTRIBUTES)
    assert request_attributes["gen_ai.request.model"] == "gpt-4.1-mini"
    assert request_attributes["gen_ai.system"] == "openai"
    assert request_attributes["gen_ai.request.temperature"] == 0.1
    assert request_attributes["llm.is_streaming"] == str(streaming)
    assert request_attributes["llm.user"] == _USER
    assert request_attributes["gen_ai.response.id"] == "chatcmpl-otel"
    assert request_attributes["gen_ai.usage.input_tokens"] == _USAGE["prompt_tokens"]
    assert request_attributes["gen_ai.usage.output_tokens"] == _USAGE["completion_tokens"]
    assert request_attributes["gen_ai.usage.total_tokens"] == _USAGE["total_tokens"]
    raw_attributes: Final = spans["raw_gen_ai_request"].attributes or {}
    expected_raw: Final = _RAW_STREAMING_ATTRIBUTES if streaming else _RAW_NON_STREAMING_ATTRIBUTES
    assert all(raw_attributes.get(name) is not None for name in expected_raw)
