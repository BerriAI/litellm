import asyncio
import base64
import json
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import product
from typing import Final, Literal

import pytest
from integration._support.client import JSON_OBJECT
from integration._support.upstream import INTERNAL_FIELDS, aws_event_frame
from integration._support.wire import Reply, Request, wire_server
from pydantic import JsonValue

import litellm

Provider = Literal["openai", "azure", "anthropic", "gemini", "bedrock_converse", "bedrock_invoke"]

PROVIDERS: Final[tuple[Provider, ...]] = (
    "openai",
    "azure",
    "anthropic",
    "gemini",
    "bedrock_converse",
    "bedrock_invoke",
)
TOP_LEVEL_EXTRA_BODY_PROVIDERS: Final = frozenset({"openai", "azure"})
SENTINEL_METADATA: Final = {"user_api_key_hash": "h", "_litellm_sentinel": "x"}
DENYLISTED: Final = INTERNAL_FIELDS | frozenset(SENTINEL_METADATA)
BEDROCK_MODEL: Final = "anthropic.claude-3-haiku-20240307-v1:0"

OPENAI_MESSAGE: Final[JsonValue] = {
    "id": "chatcmpl-fence",
    "object": "chat.completion",
    "created": 1,
    "model": "gpt-4o-mini",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "fenced"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
}
OPENAI_CHUNKS: Final[tuple[JsonValue, ...]] = (
    {
        "id": "chatcmpl-fence",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "gpt-4o-mini",
        "choices": [{"index": 0, "delta": {"role": "assistant", "content": "fenced"}, "finish_reason": None}],
    },
    {
        "id": "chatcmpl-fence",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "gpt-4o-mini",
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    },
)
ANTHROPIC_MESSAGE: Final[JsonValue] = {
    "id": "msg_fence",
    "type": "message",
    "role": "assistant",
    "model": "claude-sonnet-4-5-20250929",
    "content": [{"type": "text", "text": "fenced"}],
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": {"input_tokens": 3, "output_tokens": 1},
}
ANTHROPIC_EVENTS: Final[tuple[dict[str, JsonValue], ...]] = (
    {
        "type": "message_start",
        "message": {
            "id": "msg_fence",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-4-5-20250929",
            "content": [],
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": 3, "output_tokens": 0},
        },
    },
    {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
    {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "fenced"}},
    {"type": "content_block_stop", "index": 0},
    {
        "type": "message_delta",
        "delta": {"stop_reason": "end_turn", "stop_sequence": None},
        "usage": {"output_tokens": 1},
    },
    {"type": "message_stop"},
)
GEMINI_RESPONSE: Final[JsonValue] = {
    "candidates": [{"content": {"role": "model", "parts": [{"text": "fenced"}]}, "finishReason": "STOP", "index": 0}],
    "usageMetadata": {"promptTokenCount": 3, "candidatesTokenCount": 1, "totalTokenCount": 4},
}
CONVERSE_RESPONSE: Final[JsonValue] = {
    "output": {"message": {"role": "assistant", "content": [{"text": "fenced"}]}},
    "stopReason": "end_turn",
    "usage": {"inputTokens": 3, "outputTokens": 1, "totalTokens": 4},
    "metrics": {"latencyMs": 1},
}
CONVERSE_EVENTS: Final[tuple[tuple[str, Mapping[str, JsonValue]], ...]] = (
    ("messageStart", {"role": "assistant"}),
    ("contentBlockDelta", {"contentBlockIndex": 0, "delta": {"text": "fenced"}}),
    ("contentBlockStop", {"contentBlockIndex": 0}),
    ("messageStop", {"stopReason": "end_turn"}),
    ("metadata", {"usage": {"inputTokens": 3, "outputTokens": 1, "totalTokens": 4}, "metrics": {"latencyMs": 1}}),
)


@dataclass(frozen=True, slots=True)
class Case:
    provider: Provider
    stream: bool
    asynchronous: bool

    @property
    def id(self) -> str:
        return f"{self.provider}-{'stream' if self.stream else 'nonstream'}-{'async' if self.asynchronous else 'sync'}"


CASES: Final = tuple(
    Case(provider, stream, asynchronous)
    for provider, stream, asynchronous in product(PROVIDERS, (False, True), (False, True))
)


def _sse(events: tuple[JsonValue, ...], done: bool) -> tuple[bytes, ...]:
    frames: Final = tuple(f"data: {json.dumps(event)}\n\n".encode() for event in events)
    return frames + ((b"data: [DONE]\n\n",) if done else ())


def _anthropic_sse() -> tuple[bytes, ...]:
    return tuple(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n".encode() for event in ANTHROPIC_EVENTS)


def _invoke_event_stream() -> bytes:
    return b"".join(
        aws_event_frame("chunk", {"bytes": base64.b64encode(json.dumps(event).encode()).decode()}, "fence", "fence")
        for event in ANTHROPIC_EVENTS
    )


def _converse_event_stream() -> bytes:
    return b"".join(aws_event_frame(event_type, payload, "fence", "fence") for event_type, payload in CONVERSE_EVENTS)


def _reply(request: Request) -> Reply:
    target: Final = request.target
    streaming: Final = JSON_OBJECT.validate_json(request.body).get("stream") is True
    if "/v1/messages" in target:
        return (
            Reply(content_type="text/event-stream", chunks=_anthropic_sse())
            if streaming
            else Reply(body=json.dumps(ANTHROPIC_MESSAGE).encode())
        )
    if ":streamGenerateContent" in target:
        return Reply(content_type="text/event-stream", chunks=_sse((GEMINI_RESPONSE,), done=False))
    if ":generateContent" in target:
        return Reply(body=json.dumps(GEMINI_RESPONSE).encode())
    if target.endswith("/converse-stream"):
        return Reply(content_type="application/vnd.amazon.eventstream", body=_converse_event_stream())
    if target.endswith("/converse"):
        return Reply(body=json.dumps(CONVERSE_RESPONSE).encode())
    if target.endswith("/invoke-with-response-stream"):
        return Reply(content_type="application/vnd.amazon.eventstream", body=_invoke_event_stream())
    if target.endswith("/invoke"):
        return Reply(body=json.dumps(ANTHROPIC_MESSAGE).encode())
    if streaming:
        return Reply(content_type="text/event-stream", chunks=_sse(OPENAI_CHUNKS, done=True))
    return Reply(body=json.dumps(OPENAI_MESSAGE).encode())


def _provider_parameters(provider: Provider, url: str) -> Mapping[str, JsonValue]:
    match provider:
        case "openai":
            return {"model": "openai/gpt-4o-mini", "api_base": url, "api_key": "synthetic-openai-key"}
        case "azure":
            return {
                "model": "azure/fence-deployment",
                "api_base": url,
                "api_key": "synthetic-azure-key",
                "api_version": "2024-10-21",
            }
        case "anthropic":
            return {
                "model": "anthropic/claude-sonnet-4-5-20250929",
                "api_base": url,
                "api_key": "synthetic-anthropic-key",
            }
        case "gemini":
            return {"model": "gemini/gemini-2.5-flash", "api_base": url, "api_key": "synthetic-gemini-key"}
        case "bedrock_converse":
            return {
                "model": f"bedrock/converse/{BEDROCK_MODEL}",
                "aws_bedrock_runtime_endpoint": url,
                "api_key": "synthetic-bedrock-bearer",
                "aws_region_name": "us-east-1",
            }
        case "bedrock_invoke":
            return {
                "model": f"bedrock/invoke/{BEDROCK_MODEL}",
                "aws_bedrock_runtime_endpoint": url,
                "api_key": "synthetic-bedrock-bearer",
                "aws_region_name": "us-east-1",
            }


def _keys_at_any_depth(value: JsonValue) -> frozenset[str]:
    if isinstance(value, dict):
        return frozenset(value).union(*(_keys_at_any_depth(child) for child in value.values()))
    if isinstance(value, list):
        return frozenset[str]().union(*(_keys_at_any_depth(child) for child in value))
    return frozenset()


def _stream_text(chunks: list[litellm.ModelResponseStream]) -> str:
    return "".join(str(chunk.choices[0].delta.content or "") for chunk in chunks)


async def _completion_text(case: Case, parameters: Mapping[str, JsonValue]) -> str:
    if case.asynchronous and case.stream:
        return _stream_text([chunk async for chunk in await litellm.acompletion(**parameters, stream=True)])
    if case.asynchronous:
        return str((await litellm.acompletion(**parameters)).choices[0].message.content)
    if case.stream:
        return _stream_text(await asyncio.to_thread(lambda: list(litellm.completion(**parameters, stream=True))))
    return str((await asyncio.to_thread(litellm.completion, **parameters)).choices[0].message.content)


@pytest.fixture
def httpx_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setenv("LITELLM_RUST", "false")


@pytest.mark.usefixtures("httpx_only")
@pytest.mark.parametrize("case", CASES, ids=[case.id for case in CASES])
@pytest.mark.covers("other.provider_wire.sdk_internal_parameters_fenced_across_providers")
async def test_internal_params_never_reach_provider_body_and_extra_body_does(case: Case) -> None:
    with wire_server(_reply) as wire:
        parameters: Final = {
            **_provider_parameters(case.provider, wire.url),
            "messages": [{"role": "user", "content": "fence control"}],
            "max_tokens": 16,
            "timeout": 5,
            "num_retries": 0,
            "stream_chunk_size": 64,
            "metadata": SENTINEL_METADATA,
            "extra_body": {"custom_provider_key": 1},
        }
        text: Final = await _completion_text(case, parameters)
        received: Final = wire.drain()
    assert text == "fenced", f"{case.id}: peer reply was not surfaced"
    assert len(received) == 1, (
        f"{case.id}: expected exactly one upstream request, saw {[request.target for request in received]}"
    )
    body: Final = JSON_OBJECT.validate_json(received[0].body)
    leaked: Final = DENYLISTED & _keys_at_any_depth(body)
    assert not leaked, f"{case.id}: internal fields reached the provider body {sorted(leaked)}: {json.dumps(body)}"
    if case.provider in TOP_LEVEL_EXTRA_BODY_PROVIDERS:
        assert body["custom_provider_key"] == 1, (
            f"{case.id}: extra_body was not merged at the top level: {json.dumps(body)}"
        )
        return
    # TODO: anthropic ships extra_body nested as a literal "extra_body" key and converse under
    # additionalModelRequestFields.extra_body; only delivery is asserted until the intended shape is decided
    assert "custom_provider_key" in _keys_at_any_depth(body), (
        f"{case.id}: extra_body never reached the provider: {json.dumps(body)}"
    )
