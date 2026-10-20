import asyncio
import json
import time
from collections.abc import Mapping
from typing import Final

import pytest
from integration._support.wire import Reply, Request, Wire, wire_server
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

import litellm
from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER

_IDENTITY: Final = {
    "user_api_key_alias": "caller-span-alias",
    "user_api_key_user_id": "caller-span-user",
    "user_api_key_team_id": "caller-span-team",
}
_CHAT_REPLY: Final = Reply(
    body=json.dumps(
        {
            "id": "chatcmpl-caller-span",
            "object": "chat.completion",
            "created": 1,
            "model": "gpt-4o-mini",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
    ).encode()
)


def _is_identity(key: str) -> bool:
    return key == "langfuse.observation.metadata" or key.startswith("langfuse.trace.metadata.user_api_key_")


def _exported(langfuse: Wire, seen: list[tuple[str, dict[str, object]]]) -> list[tuple[str, dict[str, object]]]:
    for batch in langfuse.drain():
        request: ExportTraceServiceRequest = ExportTraceServiceRequest.FromString(batch.body)
        seen.extend(
            (
                span.name,
                {
                    attribute.key: getattr(attribute.value, attribute.value.WhichOneof("value"))
                    for attribute in span.attributes
                },
            )
            for resource in request.resource_spans
            for scope in resource.scope_spans
            for span in scope.spans
        )
    return seen


async def _exported_until(
    langfuse: Wire, seen: list[tuple[str, dict[str, object]]], name: str, prompt: str
) -> list[tuple[str, dict[str, object]]]:
    """The spans the Langfuse sink got, once a ``name`` span for ``prompt`` is among them; polls without blocking the loop."""
    deadline: Final = time.monotonic() + 25
    while not any(span == name and prompt in json.dumps(attributes) for span, attributes in _exported(langfuse, seen)):
        assert time.monotonic() < deadline, [span for span, _ in seen]
        await asyncio.sleep(0.2)
    return seen


async def _complete(upstream: Wire, prompt: str) -> None:
    await litellm.acompletion(
        model="openai/gpt-4o-mini",
        api_base=upstream.url + "/v1",
        api_key="synthetic-caller-span-key",
        messages=[{"role": "user", "content": prompt}],
        metadata=dict(_IDENTITY),
    )
    await GLOBAL_LOGGING_WORKER.flush()


@pytest.mark.asyncio
async def test_langfuse_otel_keeps_the_caller_identity_off_the_callers_active_span(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app_exporter: Final = InMemorySpanExporter()
    app_provider: Final = TracerProvider()
    app_provider.add_span_processor(SimpleSpanProcessor(app_exporter))

    def chat(_request: Request) -> Reply:
        return _CHAT_REPLY

    with wire_server(chat) as upstream, wire_server(lambda _request: Reply()) as langfuse:
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-caller-span")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-caller-span")
        monkeypatch.setenv("LANGFUSE_OTEL_HOST", langfuse.url)
        monkeypatch.setenv("OTEL_BSP_SCHEDULE_DELAY", "100")
        monkeypatch.setattr(litellm, "callbacks", ["langfuse_otel"])

        seen: Final[list[tuple[str, dict[str, object]]]] = []  # mutable-ok: drain() consumes the sink queue
        with app_provider.get_tracer("app").start_as_current_span("app.operation"):
            await _complete(upstream, "inside the caller span")
            await _exported_until(langfuse, seen, "raw_gen_ai_request", "inside the caller span")
        await _complete(upstream, "no caller span")
        exported: Final = await _exported_until(langfuse, seen, "litellm_request", "no caller span")

    (app_span,) = app_exporter.get_finished_spans()
    app_attributes: Final[Mapping[str, object]] = app_span.attributes or {}
    assert app_attributes["langfuse.observation.type"] == "generation", sorted(app_attributes)
    assert not [key for key in app_attributes if _is_identity(key)], sorted(app_attributes)
    (own_span,) = [attributes for name, attributes in exported if name == "litellm_request"]
    assert json.loads(str(own_span["langfuse.observation.metadata"])) == _IDENTITY, own_span
    assert {key: value for key, value in own_span.items() if key.startswith("langfuse.trace.metadata.")} == {
        f"langfuse.trace.metadata.{field}": value for field, value in _IDENTITY.items()
    }, sorted(own_span)
