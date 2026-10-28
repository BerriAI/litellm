import asyncio
import json
import uuid
from collections.abc import Iterator
from typing import Final

import pytest
from integration._support.wire import Reply, Request, Wire, wire_server
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest
from opentelemetry.proto.trace.v1.trace_pb2 import Span

import litellm
from litellm.integrations.langfuse.langfuse_sdk import resolve_trace_id

REDACTED: Final = "redacted-by-litellm"


def _accept(request: Request) -> Reply:
    return Reply()


def _spans(bodies: tuple[bytes, ...]) -> Iterator[Span]:
    for body in bodies:
        for resource_spans in ExportTraceServiceRequest.FromString(body).resource_spans:
            for scope_spans in resource_spans.scope_spans:
                yield from scope_spans.spans


def _generations(bodies: tuple[bytes, ...], trace_id: str) -> tuple[dict[str, str], ...]:
    attributes: Final = (
        {attribute.key: attribute.value.string_value for attribute in span.attributes}
        for span in _spans(bodies)
        if span.trace_id.hex() == trace_id
    )
    return tuple(span for span in attributes if span.get("langfuse.observation.type") == "generation")


async def _trace_exports(wire: Wire, trace_id: str, seen: tuple[bytes, ...], attempts: int) -> tuple[bytes, ...]:
    bodies: Final = seen + tuple(
        request.body for request in wire.drain() if request.method == "POST" and request.target.endswith("/traces")
    )
    if attempts == 0 or _generations(bodies, trace_id):
        return bodies
    await asyncio.sleep(0.25)
    return await _trace_exports(wire, trace_id, bodies, attempts - 1)


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_langfuse_export_carries_no_raw_prompt_or_answer_when_message_logging_is_off(
    stream: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    prompt: Final = f"prompt-{uuid.uuid4()}"
    answer: Final = f"answer-{uuid.uuid4()}"
    trace_name: Final = f"litellm-test-{uuid.uuid4()}"
    monkeypatch.setattr(litellm, "turn_off_message_logging", True)
    monkeypatch.setattr(litellm, "success_callback", ["langfuse"])
    with wire_server(_accept) as wire:
        response: Final = await litellm.acompletion(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            mock_response=answer,
            stream=stream,
            metadata={"trace_id": trace_name},
            langfuse_public_key=f"pk-lf-{trace_name}",
            langfuse_secret_key="sk-lf-local",
            langfuse_host=wire.url,
        )
        if stream:
            _ = [chunk async for chunk in response]
        bodies: Final = await _trace_exports(wire, resolve_trace_id(trace_name), (), 120)

    generations: Final = _generations(bodies, resolve_trace_id(trace_name))
    assert len(generations) == 1, generations
    assert json.loads(generations[0]["langfuse.observation.input"]) == {
        "messages": [{"content": REDACTED, "role": "user"}]
    }
    assert json.loads(generations[0]["langfuse.observation.output"])["content"] == REDACTED
    assert all(prompt.encode() not in body and answer.encode() not in body for body in bodies)
