"""Shared rig for the second wave of `/v1/messages` provider contracts (`test_dec1_msgprov2_*`).

The first wave (`dec1_msgprov_support.py`, imported here without changes) pins the basic request and
answer shapes for Bedrock, Vertex, Azure, Fireworks and Baseten. This wave covers what it left out:
the native Anthropic-compatible endpoint that Fireworks, Baseten and any OpenAI-compatible
deployment reach when `model_info.supported_endpoints` lists `/v1/messages`
(`OpenAILikeAnthropicMessagesConfig`), the Messages to chat completions bridge in more depth (content
blocks both ways, thinking, refusals, cache usage and the exact SSE event sequence), anthropic-beta and
count_tokens per provider, the Python-only features that the Rust-native providers (`anthropic`,
`azure_ai` Claude and Bedrock InvokeModel, `litellm/rust_bridge/catalog.py` RULES) must still honour,
and spend rows and budgets on each native provider.

Harness facts reused from the first wave: every prompt carries a fresh marker because the shared proxy
has `litellm_settings.cache: true`, only `POST` requests count as provider calls, SpendLogs are read
through `SPEND_COLUMNS` by `model_group`, and Vertex tokens come from the shared upstream's
`/_oauth/token` route.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager
from typing import Final

from integration._support.client import Gateway, eventually, object_value
from integration._support.database import read_rows
from integration._support.wire import Reply, Request
from integration.messages_endpoint.dec1_messages_support import JSON_OBJECT, Rig, message_json, sse, stream_chunks
from integration.messages_endpoint.dec1_msgprov_support import body_of, message_id
from pydantic import JsonValue

PASSTHROUGH_INFO: Final[Mapping[str, JsonValue]] = {"supported_endpoints": ["/v1/messages"]}
"""The per-deployment opt-in that forwards `/v1/messages` untranslated to `{api_base}/v1/messages`
(`litellm/llms/anthropic/messages/native/handler.py` `_deployment_passes_through_anthropic_messages`)."""

HOSTED_VLLM_BACKEND: Final = "synthetic-chat-model"
"""A model name with no cost map row, so no capability gate (vision, tool choice) refuses a block."""
HOSTED_VLLM_KEY: Final = "synthetic-vllm-key"
DEFAULT_ANTHROPIC_VERSION: Final = "2023-06-01"
STRUCTURED_OUTPUTS_BETA: Final = "structured-outputs-2025-11-13"
CONTEXT_MANAGEMENT_BETA: Final = "context-management-2025-06-27"

INITIAL_STREAM_USAGE: Final[Mapping[str, JsonValue]] = {
    "input_tokens": 0,
    "output_tokens": 0,
    "cache_creation_input_tokens": 0,
    "cache_read_input_tokens": 0,
}
"""What the chat bridge puts in `message_start.message.usage` before any provider usage is known
(`AnthropicStreamWrapper._create_initial_usage_delta`, with
`litellm.anthropic_messages_estimate_initial_input_tokens` left off on the shared proxy)."""


def anthropic_answer(model: str) -> Callable[[Request], Reply]:
    """Answer an Anthropic-shaped call by its own `stream` flag, stamping `model` into the message."""

    def respond(request: Request) -> Reply:
        identity: Final = message_id()
        if body_of(request).get("stream") is True:
            return Reply(content_type="text/event-stream", chunks=stream_chunks(identity, model=model))
        return Reply(body=json.dumps(message_json(identity, model=model)).encode())

    return respond


def anthropic_message(model: str, **fields: object) -> Reply:
    return Reply(body=json.dumps(message_json(message_id(), model=model, **fields)).encode())  # pyright: ignore[reportArgumentType]  # test-local keyword passthrough


def anthropic_stream_with_usage(
    model: str,
    *,
    input_tokens: int,
    output_tokens: int,
    cache_read: int = 0,
    cache_creation: int = 0,
) -> Reply:
    """The six-event Anthropic stream with cache counts on `message_start`, as Anthropic reports them."""
    identity: Final = message_id()
    return Reply(
        content_type="text/event-stream",
        chunks=(
            sse(
                "message_start",
                {
                    "type": "message_start",
                    "message": {
                        "id": identity,
                        "type": "message",
                        "role": "assistant",
                        "model": model,
                        "content": [],
                        "stop_reason": None,
                        "stop_sequence": None,
                        "usage": {
                            "input_tokens": input_tokens,
                            "output_tokens": 1,
                            "cache_read_input_tokens": cache_read,
                            "cache_creation_input_tokens": cache_creation,
                        },
                    },
                },
            ),
            sse(
                "content_block_start",
                {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
            ),
            sse(
                "content_block_delta",
                {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "cached"}},
            ),
            sse("content_block_stop", {"type": "content_block_stop", "index": 0}),
            sse(
                "message_delta",
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                    "usage": {"output_tokens": output_tokens},
                },
            ),
            sse("message_stop", {"type": "message_stop"}),
        ),
    )


# ---------------------------------------------------------------- OpenAI chat chunks for bridge streams


def chat_chunk(identity: str, model: str, delta: Mapping[str, JsonValue], finish: str | None = None) -> bytes:
    frame: Final = {
        "id": identity,
        "object": "chat.completion.chunk",
        "created": 1,
        "model": model,
        "choices": [{"index": 0, "delta": dict(delta), "finish_reason": finish}],
    }
    return b"data: " + json.dumps(frame).encode() + b"\n\n"


def chat_usage_chunk(identity: str, model: str, usage: Mapping[str, JsonValue]) -> bytes:
    frame: Final = {
        "id": identity,
        "object": "chat.completion.chunk",
        "created": 1,
        "model": model,
        "choices": [],
        "usage": dict(usage),
    }
    return b"data: " + json.dumps(frame).encode() + b"\n\n"


def chat_stream(
    model: str,
    deltas: Sequence[tuple[Mapping[str, JsonValue], str | None]],
    *,
    usage: Mapping[str, JsonValue] | None = None,
) -> Reply:
    """A chat completions SSE stream built from `(delta, finish_reason)` pairs, a usage chunk and `[DONE]`."""
    identity: Final = f"chatcmpl-{uuid.uuid4().hex}"
    final_usage: Final = (
        usage if usage is not None else {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18}
    )
    return Reply(
        content_type="text/event-stream",
        chunks=(
            *(chat_chunk(identity, model, delta, finish) for delta, finish in deltas),
            chat_usage_chunk(identity, model, final_usage),
            b"data: [DONE]\n\n",
        ),
    )


def chat_message_reply(
    model: str,
    message: Mapping[str, JsonValue],
    *,
    finish_reason: str = "stop",
    usage: Mapping[str, JsonValue] | None = None,
    identity: str | None = None,
) -> Reply:
    """A non-streamed chat completion whose single choice carries exactly `message`."""
    return Reply(
        body=json.dumps(
            {
                "id": identity if identity is not None else f"chatcmpl-{uuid.uuid4().hex}",
                "object": "chat.completion",
                "created": 1,
                "model": model,
                "choices": [{"index": 0, "message": dict(message), "finish_reason": finish_reason}],
                "usage": dict(usage)
                if usage is not None
                else {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
            }
        ).encode()
    )


# ---------------------------------------------------------------- SSE reading


def event_names(events: Sequence[tuple[str, Mapping[str, JsonValue]]]) -> tuple[str, ...]:
    return tuple(name for name, _data in events)


def collapsed(names: Sequence[str]) -> tuple[str, ...]:
    """Event names with consecutive repeats folded into one, so the number of deltas does not matter."""
    return tuple(name for index, name in enumerate(names) if index == 0 or names[index - 1] != name)


def blocks_by_index(events: Sequence[tuple[str, Mapping[str, JsonValue]]]) -> dict[int, dict[str, JsonValue]]:
    """`content_block_start.content_block` keyed by its `index`."""
    return {
        int(str(data["index"])): object_value(data["content_block"])
        for name, data in events
        if name == "content_block_start"
    }


def deltas_of(events: Sequence[tuple[str, Mapping[str, JsonValue]]], index: int) -> list[dict[str, JsonValue]]:
    return [
        object_value(data["delta"]) for name, data in events if name == "content_block_delta" and data["index"] == index
    ]


def the_event(events: Sequence[tuple[str, Mapping[str, JsonValue]]], wanted: str) -> Mapping[str, JsonValue]:
    matching: Final = [data for name, data in events if name == wanted]
    assert len(matching) == 1, (wanted, events)
    return matching[0]


# ---------------------------------------------------------------- Stored spend


def stored_spend(gateway: Gateway, query: str, identity: str) -> float | None:
    rows: Final = gateway.rows(query, (identity,))
    return None if not rows or rows[0]["spend"] is None else float(str(rows[0]["spend"]))


KEY_SPEND: Final = 'SELECT spend FROM "LiteLLM_VerificationToken" WHERE token = %s'
TEAM_SPEND: Final = 'SELECT spend FROM "LiteLLM_TeamTable" WHERE team_id = %s'
USER_SPEND: Final = 'SELECT spend FROM "LiteLLM_UserTable" WHERE user_id = %s'
END_USER_SPEND: Final = 'SELECT spend FROM "LiteLLM_EndUserTable" WHERE user_id = %s'


def spend_settles(gateway: Gateway, query: str, identity: str, amount: float, *, seconds: float = 90) -> float:
    """Poll until the stored spend of one entity is `amount` (within float noise) and return it."""
    reached: Final = eventually(
        lambda: stored_spend(gateway, query, identity),
        lambda spend: spend is not None and abs(spend - amount) < 1e-9,
        seconds=seconds,
    )
    assert reached is not None
    return reached


def hashed(key: str) -> str:
    from hashlib import sha256

    return sha256(key.encode()).hexdigest()


def rows_for_group(model_group: str, expected: int, *, seconds: float = 90) -> list[dict[str, JsonValue]]:
    from integration.messages_endpoint.dec1_messages_support import SPEND_COLUMNS

    return eventually(
        lambda: read_rows(SPEND_COLUMNS + "WHERE model_group=%s", (model_group,)),
        lambda values: len(values) >= expected,
        seconds=seconds,
    )


def json_body(raw: bytes) -> dict[str, JsonValue]:
    return JSON_OBJECT.validate_json(raw)


# ---------------------------------------------------------------- Native Messages providers


def native_deployment(
    gateway: Gateway,
    provider: str,
    respond: Callable[[Request], Reply] | None = None,
    **params: JsonValue,
) -> AbstractContextManager[Rig]:
    """One deployment of a provider that has a native Messages config, behind an owned peer.

    `anthropic`, `azure_ai` (Claude on Foundry) and `bedrock_invoke` run on the Rust path today
    (`litellm/rust_bridge/catalog.py` RULES: RUST_REQUIRED); `vertex_ai` is native Messages on Python and is
    the next port (draft #51294). `respond` defaults to an echo in that provider's wire format.
    """
    from integration.messages_endpoint.dec1_msgprov_support import (
        AZURE_AI_BACKEND,
        AZURE_AI_KEY,
        BEDROCK_MODEL_ID,
        BEDROCK_TOKEN,
        VERTEX_BACKEND,
        VERTEX_LOCATION,
        VERTEX_PROJECT,
        anthropic_body_echo,
        invoke_echo,
        provider_deployment,
        vertex_credentials,
        vertex_echo,
    )

    match provider:
        case "anthropic":
            return provider_deployment(
                gateway,
                respond or anthropic_answer(ANTHROPIC_BACKEND),
                model=f"anthropic/{ANTHROPIC_BACKEND}",
                api_key=ANTHROPIC_KEY,
                **params,
            )
        case "azure_ai":
            return provider_deployment(
                gateway,
                respond or anthropic_body_echo,
                model=f"azure_ai/{AZURE_AI_BACKEND}",
                api_key=AZURE_AI_KEY,
                **params,
            )
        case "bedrock_invoke":
            return provider_deployment(
                gateway,
                respond or invoke_echo,
                model=f"bedrock/invoke/{BEDROCK_MODEL_ID}",
                api_key=BEDROCK_TOKEN,
                aws_region_name="us-east-1",
                **params,
            )
        case "vertex_ai":
            settings: Final[dict[str, JsonValue]] = {
                "vertex_project": VERTEX_PROJECT,
                "vertex_location": VERTEX_LOCATION,
                "vertex_credentials": vertex_credentials(gateway),
                **params,
            }
            return provider_deployment(
                gateway, respond or vertex_echo, model=f"vertex_ai/{VERTEX_BACKEND}", api_key=None, **settings
            )
        case _:
            raise AssertionError(provider)


ANTHROPIC_BACKEND: Final = "claude-sonnet-4-5-20250929"
ANTHROPIC_KEY: Final = "synthetic-anthropic-key"
RUST_NATIVE_PROVIDERS: Final = ("anthropic", "azure_ai", "bedrock_invoke")
NATIVE_PROVIDERS: Final = (*RUST_NATIVE_PROVIDERS, "vertex_ai")
SPEND_PROVIDER_NAME: Final[Mapping[str, str]] = {
    "anthropic": "anthropic",
    "azure_ai": "azure_ai",
    "bedrock_invoke": "bedrock",
    "vertex_ai": "vertex_ai",
}
"""The `custom_llm_provider` column each native provider writes on its SpendLogs row."""


def native_error(provider: str, status: int, error_type: str, message: str) -> Reply:
    """An error in the provider's own wire format: AWS JSON for Bedrock, Anthropic's envelope otherwise."""
    from integration.messages_endpoint.dec1_msgprov_support import anthropic_error_reply, aws_error

    if provider == "bedrock_invoke":
        aws_types: Final = {
            429: "ThrottlingException",
            500: "InternalServerException",
            503: "ServiceUnavailableException",
        }
        return aws_error(status, aws_types.get(status, "ValidationException"), message)
    return anthropic_error_reply(status, error_type, message)
