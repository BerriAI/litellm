"""`POST /v1/messages` served by a `bedrock/converse/<id>` deployment through the chat bridge.

`BedrockModelInfo.get_bedrock_provider_config_for_messages_api` returns no native Messages config for
the Converse route, so the Anthropic request is adapted to chat completions
(`litellm/llms/anthropic/messages/adapters/transformation.py`) and then to a Converse document by
`AmazonConverseConfig`: `POST {api_base}/model/{id}/converse` (or `/converse-stream`) with the model
id's `:` sent as `%3A`, Anthropic `messages` rebuilt as Converse content blocks, `max_tokens` and
`stop_sequences` moved into `inferenceConfig`, `system` as `[{"text": ...}]`, tools as
`toolConfig.tools[].toolSpec`, and `cache_control` turned into a separate `cachePoint` block. The
answer goes back the same way: Converse `stopReason` to an OpenAI finish reason to an Anthropic
`stop_reason`, which loses `stop_sequence` on the way.

The chat bridge is Python only (`litellm/messages/dispatch.py` marks CHAT_COMPLETIONS targets
PYTHON_ONLY), so every case here runs on Python today and is the contract a Rust bridge must keep.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from contextlib import AbstractContextManager
from typing import Final

import pytest
from integration._support.client import Gateway, object_value
from integration._support.wire import Reply, Request
from integration.messages_endpoint.dec1_messages_support import JSON_OBJECT, Rig, anthropic_error, sse_events
from integration.messages_endpoint.dec1_msgprov_support import (
    BEDROCK_MODEL_ID,
    BEDROCK_TOKEN,
    CACHE_CREATION_COST,
    CACHE_READ_COST,
    INPUT_COST,
    OUTPUT_COST,
    PNG_BASE64,
    PRICES,
    TOOL,
    at,
    aws_error,
    body_of,
    converse_echo,
    converse_reply,
    converse_stream_reply,
    only_spend_row,
    path_of,
    provider_deployment,
    spend_of,
    unique_text,
)
from pydantic import JsonValue

_CONVERSE: Final = f"/model/{BEDROCK_MODEL_ID}/converse"
_CONVERSE_STREAM: Final = f"/model/{BEDROCK_MODEL_ID}/converse-stream"
_CONVERSE_TOP_LEVEL: Final = frozenset(
    {
        "messages",
        "system",
        "inferenceConfig",
        "toolConfig",
        "guardrailConfig",
        "additionalModelRequestFields",
        "additionalModelResponseFieldPaths",
        "promptVariables",
        "requestMetadata",
        "performanceConfig",
        "serviceTier",
        "outputConfig",
    }
)
"""Top-level members of the Converse request, from the AWS API reference (`API_runtime_Converse`, read 2026-11)."""


def _converse(
    gateway: Gateway, respond: Callable[[Request], Reply] = converse_echo, **params: JsonValue
) -> AbstractContextManager[Rig]:
    return provider_deployment(
        gateway,
        respond,
        model=f"bedrock/converse/{BEDROCK_MODEL_ID}",
        api_key=BEDROCK_TOKEN,
        aws_region_name="us-east-1",
        **params,
    )


# ---------------------------------------------------------------- URL, method and auth


def test_a_converse_deployment_posts_to_the_converse_path_with_the_bearer_token(gateway: Gateway) -> None:
    """A `bedrock/converse/<id>` deployment receives one `POST /model/<id>/converse` with
    `Authorization: Bearer <api_key>`. The raw target encodes the version colon as `%3A`, unlike the
    invoke route, which keeps it raw."""
    with _converse(gateway) as rig:
        response: Final = rig.post(rig.body())
        assert response.status_code == 200, response.text
        call: Final = rig.one_call()
        assert (call.method, path_of(call)) == ("POST", _CONVERSE)
        assert "v1%3A0/converse" in call.target, call.target
        assert call.headers["authorization"] == f"Bearer {BEDROCK_TOKEN}"


def test_a_streaming_request_posts_to_converse_stream(gateway: Gateway) -> None:
    """With `stream: true` the call goes to `/model/<id>/converse-stream` and the caller gets an
    HTTP 200 `text/event-stream` answer."""
    with _converse(gateway) as rig:
        response, _text = rig.stream(rig.body(stream=True))
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        assert path_of(rig.one_call()) == _CONVERSE_STREAM


# ---------------------------------------------------------------- body translation


def test_the_converse_body_has_text_blocks_and_inference_config_and_no_anthropic_fields(gateway: Gateway) -> None:
    """A plain request becomes `messages: [{"role": "user", "content": [{"text": <prompt>}]}]` and
    `inferenceConfig: {"maxTokens": 64}`. `model`, `max_tokens`, `stream` and `anthropic_version`
    never appear, and every top-level key is a Converse API member, since Converse rejects unknown
    members with a ValidationException."""
    prompt: Final = unique_text()
    with _converse(gateway) as rig:
        assert rig.post(rig.body(prompt)).status_code == 200
        body: Final = rig.one_body()
        assert body["messages"] == [{"role": "user", "content": [{"text": prompt}]}]
        assert body["inferenceConfig"] == {"maxTokens": 64}
        assert set(body) <= _CONVERSE_TOP_LEVEL, sorted(body)


def test_the_system_prompt_becomes_a_converse_system_text_block(gateway: Gateway) -> None:
    """`system: "You are terse."` is sent as the top-level `system: [{"text": "You are terse."}]` and
    never as a message with role `system`, which Converse does not accept."""
    with _converse(gateway) as rig:
        assert rig.post(rig.body(system="You are terse.")).status_code == 200
        body: Final = rig.one_body()
        assert body["system"] == [{"text": "You are terse."}]
        messages: Final = body["messages"]
        assert isinstance(messages, list)
        assert [object_value(message)["role"] for message in messages] == ["user"]


def test_stop_sequences_and_sampling_land_in_inference_config(gateway: Gateway) -> None:
    """`stop_sequences`, `temperature` and `top_p` map to `inferenceConfig.stopSequences`,
    `temperature` and `topP` next to `maxTokens`."""
    with _converse(gateway) as rig:
        assert rig.post(rig.body(stop_sequences=["###"], temperature=0.25, top_p=0.5)).status_code == 200
        assert rig.one_body()["inferenceConfig"] == {
            "maxTokens": 64,
            "stopSequences": ["###"],
            "temperature": 0.25,
            "topP": 0.5,
        }


def test_tools_become_converse_tool_specs(gateway: Gateway) -> None:
    """An Anthropic tool becomes `toolConfig.tools[0].toolSpec` with the same `name` and
    `description` and the `input_schema` wrapped as `inputSchema.json`."""
    with _converse(gateway) as rig:
        assert rig.post(rig.body(tools=[TOOL])).status_code == 200
        spec: Final = object_value(at(rig.one_body(), "toolConfig", "tools", 0, "toolSpec"))
        assert (spec["name"], spec["description"]) == (TOOL["name"], TOOL["description"])
        schema: Final = object_value(at(spec, "inputSchema", "json"))
        assert (schema["type"], schema["properties"], schema["required"]) == (
            "object",
            {"city": {"type": "string"}},
            ["city"],
        )


@pytest.mark.parametrize(
    ("tool_choice", "converse_choice"),
    [
        ({"type": "any"}, {"any": {}}),
        ({"type": "auto"}, {"auto": {}}),
        ({"type": "tool", "name": "get_weather"}, {"tool": {"name": "get_weather"}}),
    ],
    ids=["any", "auto", "tool"],
)
def test_tool_choice_maps_to_the_converse_tool_choice_union(
    gateway: Gateway, tool_choice: dict[str, JsonValue], converse_choice: dict[str, JsonValue]
) -> None:
    """Anthropic `tool_choice` goes through the OpenAI value (`required`, `auto`, a named function) and
    lands as Converse `toolConfig.toolChoice`: `{"any": {}}`, `{"auto": {}}` or
    `{"tool": {"name": ...}}`."""
    with _converse(gateway) as rig:
        assert rig.post(rig.body(tools=[TOOL], tool_choice=tool_choice)).status_code == 200
        assert at(rig.one_body(), "toolConfig", "toolChoice") == converse_choice


def test_enabled_thinking_goes_to_additional_model_request_fields(gateway: Gateway) -> None:
    """`thinking: {"type": "enabled", "budget_tokens": 2048}` is not a Converse member, so it must be
    sent inside `additionalModelRequestFields.thinking` unchanged. UNVERIFIED: the bridge may raise
    `maxTokens` to fit the budget; only the thinking object is asserted."""
    thinking: Final[dict[str, JsonValue]] = {"type": "enabled", "budget_tokens": 2048}
    with _converse(gateway) as rig:
        assert rig.post(rig.body(max_tokens=4096, thinking=thinking)).status_code == 200
        assert at(rig.one_body(), "additionalModelRequestFields", "thinking") == thinking


def test_cache_control_on_a_text_block_becomes_a_following_cache_point(gateway: Gateway) -> None:
    """A user text block with `cache_control: {"type": "ephemeral"}` becomes two Converse blocks,
    `{"text": ...}` then `{"cachePoint": {"type": "default"}}`, so Bedrock caches the prefix the
    caller marked. Dropping it silently costs the caller full price on every turn."""
    prompt: Final = unique_text()
    content: Final[list[JsonValue]] = [{"type": "text", "text": prompt, "cache_control": {"type": "ephemeral"}}]
    with _converse(gateway) as rig:
        assert rig.post(rig.body(messages=[{"role": "user", "content": content}])).status_code == 200
        assert at(rig.one_body(), "messages", 0, "content") == [{"text": prompt}, {"cachePoint": {"type": "default"}}]


def test_an_image_block_becomes_a_converse_image_with_raw_bytes(gateway: Gateway) -> None:
    """A base64 PNG image block becomes `{"image": {"format": "png", "source": {"bytes": <base64>}}}`.
    UNVERIFIED: the block order next to the text block; only the image block is located and compared."""
    content: Final[list[JsonValue]] = [
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": PNG_BASE64}},
        {"type": "text", "text": unique_text()},
    ]
    with _converse(gateway) as rig:
        assert rig.post(rig.body(messages=[{"role": "user", "content": content}])).status_code == 200
        blocks: Final = at(rig.one_body(), "messages", 0, "content")
        assert isinstance(blocks, list)
        images: Final = [block for block in blocks if isinstance(block, dict) and "image" in block]
        assert images == [{"image": {"format": "png", "source": {"bytes": PNG_BASE64}}}], blocks


# ---------------------------------------------------------------- response translation


def test_a_converse_text_answer_becomes_an_anthropic_message(gateway: Gateway) -> None:
    """Converse `output.message.content: [{"text": "pong"}]` with `stopReason: end_turn` and 11 in, 7
    out tokens becomes an Anthropic message with `content: [{"type": "text", "text": "pong"}]`,
    `stop_reason: end_turn` and `usage` 11 and 7."""
    with _converse(gateway, lambda _r: converse_reply()) as rig:
        response: Final = rig.post(rig.body())
        assert response.status_code == 200, response.text
        payload: Final = JSON_OBJECT.validate_json(response.content)
        assert (payload["type"], payload["role"]) == ("message", "assistant")
        assert payload["content"] == [{"type": "text", "text": "pong"}]
        assert payload["stop_reason"] == "end_turn"
        usage: Final = object_value(payload["usage"])
        assert (usage["input_tokens"], usage["output_tokens"]) == (11, 7)


def test_a_converse_tool_use_becomes_an_anthropic_tool_use_block(gateway: Gateway) -> None:
    """A Converse `toolUse` block (`toolUseId`, `name`, `input`) with `stopReason: tool_use` becomes
    `{"type": "tool_use", "id", "name", "input"}` with the same values and `stop_reason: tool_use`.
    BUG: a tool-only Converse answer comes back with an extra `{"type": "text", "text": ""}` before the
    `tool_use` block. Converse sets the chat `content` to `""` whenever no text block was answered
    (`litellm/llms/bedrock/chat/converse_transformation.py:3078`), and the adapter keeps an empty string as a
    text block unless compaction blocks are present
    (`litellm/llms/anthropic/messages/adapters/transformation.py:1564`). Anthropic never answers an empty text
    block and rejects one on input, which the product itself states when it drops such blocks from tool-call
    turns (`litellm/litellm_core_utils/prompt_templates/factory.py:2503`). Seen in run 43314424795."""
    pytest.skip("BUG: a tool-only Converse answer on /v1/messages gains an empty text block before tool_use")
    tool_use: Final[dict[str, JsonValue]] = {
        "toolUse": {"toolUseId": "tooluse_msgprov01", "name": "get_weather", "input": {"city": "Lisbon"}}
    }
    with _converse(gateway, lambda _r: converse_reply(content=[tool_use], stop_reason="tool_use")) as rig:
        payload: Final = JSON_OBJECT.validate_json(rig.post(rig.body(tools=[TOOL])).content)
        assert payload["stop_reason"] == "tool_use"
        assert payload["content"] == [
            {"type": "tool_use", "id": "tooluse_msgprov01", "name": "get_weather", "input": {"city": "Lisbon"}}
        ]


@pytest.mark.parametrize(
    ("converse_stop", "anthropic_stop"),
    [("end_turn", "end_turn"), ("max_tokens", "max_tokens"), ("tool_use", "tool_use")],
)
def test_the_converse_stop_reason_maps_back_to_the_anthropic_value(
    gateway: Gateway, converse_stop: str, anthropic_stop: str
) -> None:
    """`stopReason` goes through the OpenAI finish reason and back: end_turn, max_tokens (via `length`)
    and tool_use (via `tool_calls`) all survive the round trip."""
    with _converse(gateway, lambda _r: converse_reply(stop_reason=converse_stop)) as rig:
        assert JSON_OBJECT.validate_json(rig.post(rig.body()).content)["stop_reason"] == anthropic_stop


def test_a_converse_stop_sequence_comes_back_as_end_turn(gateway: Gateway) -> None:
    """Pinned current behaviour: Converse `stopReason: stop_sequence` maps to the OpenAI `stop`
    (`core_helpers.py` finish map), which the adapter turns into `end_turn`, so the caller loses
    `stop_reason: stop_sequence` and `stop_sequence` stays null. The invoke route keeps both."""
    with _converse(gateway, lambda _r: converse_reply(stop_reason="stop_sequence")) as rig:
        payload: Final = JSON_OBJECT.validate_json(rig.post(rig.body(stop_sequences=["###"])).content)
        assert (payload["stop_reason"], payload.get("stop_sequence")) == ("end_turn", None)


def test_converse_cache_usage_is_split_back_into_anthropic_cache_fields(gateway: Gateway) -> None:
    """Converse usage `inputTokens: 10`, `cacheReadInputTokens: 5`, `cacheWriteInputTokens: 9` must come
    back as Anthropic `input_tokens: 10`, `cache_read_input_tokens: 5` and
    `cache_creation_input_tokens: 9`. The bridge adds cache tokens into `prompt_tokens` and the adapter
    subtracts them again, so a mismatch on either side double counts."""
    usage: Final[dict[str, JsonValue]] = {
        "inputTokens": 10,
        "outputTokens": 4,
        "cacheReadInputTokens": 5,
        "cacheWriteInputTokens": 9,
        "totalTokens": 28,
    }
    with _converse(gateway, lambda _r: converse_reply(usage=usage)) as rig:
        answer: Final = object_value(JSON_OBJECT.validate_json(rig.post(rig.body()).content)["usage"])
        assert (
            answer["input_tokens"],
            answer["output_tokens"],
            answer["cache_read_input_tokens"],
            answer["cache_creation_input_tokens"],
        ) == (10, 4, 5, 9)


# ---------------------------------------------------------------- streaming


def test_the_converse_event_stream_is_bridged_to_anthropic_sse(gateway: Gateway) -> None:
    """Converse stream frames (`messageStart`, two `contentBlockDelta`, `contentBlockStop`,
    `messageStop`, `metadata`) reach the caller as an Anthropic SSE stream that starts with
    `message_start`, ends with `message_stop`, carries the text `hello stream` across its
    `content_block_delta` events, and reports `stop_reason: end_turn` on `message_delta`."""
    with _converse(gateway, lambda _r: converse_stream_reply()) as rig:
        _response, text = rig.stream(rig.body(stream=True))
        events: Final = sse_events(text)
        names: Final = [name for name, _data in events]
        assert (names[0], names[-1]) == ("message_start", "message_stop"), text
        deltas: Final = [object_value(data["delta"]) for name, data in events if name == "content_block_delta"]
        assert "".join(str(delta.get("text", "")) for delta in deltas) == "hello stream"
        message_delta: Final = next(data for name, data in events if name == "message_delta")
        assert object_value(message_delta["delta"])["stop_reason"] == "end_turn"


def test_stream_usage_from_converse_metadata_reaches_message_delta(gateway: Gateway) -> None:
    """The Converse `metadata` frame (`inputTokens: 13`, `outputTokens: 6`) arrives after
    `messageStop`, yet `message_delta.usage.output_tokens` must report 6. UNVERIFIED: where the
    bridge reports `input_tokens` (on `message_start` or `message_delta`); only output is asserted."""
    reply: Final = converse_stream_reply(usage={"inputTokens": 13, "outputTokens": 6, "totalTokens": 19})
    with _converse(gateway, lambda _r: reply) as rig:
        _response, text = rig.stream(rig.body(stream=True))
        delta: Final = next(data for name, data in sse_events(text) if name == "message_delta")
        assert object_value(delta["usage"])["output_tokens"] == 6, text


# ---------------------------------------------------------------- errors


@pytest.mark.parametrize(
    ("status", "error_class", "anthropic_type"),
    [
        (400, "ValidationException", "invalid_request_error"),
        (429, "ThrottlingException", "rate_limit_error"),
        (403, "AccessDeniedException", "permission_error"),
    ],
)
def test_a_converse_error_is_answered_in_the_anthropic_envelope(
    gateway: Gateway, status: int, error_class: str, anthropic_type: str
) -> None:
    """Converse errors go through the same `_map_bedrock_exception` as InvokeModel, and the bridge
    answers them in the Anthropic error envelope with the same status and the matching type."""
    message: Final = f"synthetic {error_class}"
    with _converse(gateway, lambda _r: aws_error(status, error_class, message)) as rig:
        response: Final = rig.post(rig.body())
        assert response.status_code == status, response.text
        error: Final = anthropic_error(response)
        assert error["type"] == anthropic_type
        assert message in str(error["message"])


# ---------------------------------------------------------------- spend


def test_the_converse_spend_row_prices_cache_tokens(gateway: Gateway) -> None:
    """A Converse answer with 10 fresh input, 5 cache read, 9 cache write and 4 output tokens writes one
    `anthropic_messages` row for provider `bedrock` with
    `spend = 10 * 0.001 + 5 * 0.0001 + 9 * 0.003 + 4 * 0.002 = 0.0455`. UNVERIFIED: that the row's
    `prompt_tokens` counts all 24 input tokens, as the bridge's chat usage does."""
    usage: Final[dict[str, JsonValue]] = {
        "inputTokens": 10,
        "outputTokens": 4,
        "cacheReadInputTokens": 5,
        "cacheWriteInputTokens": 9,
        "totalTokens": 28,
    }
    with _converse(gateway, lambda _r: converse_reply(usage=usage), **PRICES) as rig:
        assert rig.post(rig.body()).status_code == 200
        row: Final = only_spend_row(rig)
        assert (row["call_type"], row["custom_llm_provider"]) == ("anthropic_messages", "bedrock")
        assert (row["prompt_tokens"], row["completion_tokens"]) == (24, 4)
        assert spend_of(row) == pytest.approx(
            10 * INPUT_COST + 5 * CACHE_READ_COST + 9 * CACHE_CREATION_COST + 4 * OUTPUT_COST
        )


def test_a_streamed_converse_call_is_billed_from_the_metadata_frame(gateway: Gateway) -> None:
    """A streamed Converse call with metadata usage of 13 in and 6 out writes a row with
    `spend = 13 * 0.001 + 6 * 0.002 = 0.025`, so the usage that arrives after `messageStop` is
    not lost to billing."""
    reply: Final = converse_stream_reply(usage={"inputTokens": 13, "outputTokens": 6, "totalTokens": 19})
    with _converse(gateway, lambda _r: reply, **PRICES) as rig:
        rig.stream(rig.body(stream=True))
        assert spend_of(only_spend_row(rig)) == pytest.approx(13 * INPUT_COST + 6 * OUTPUT_COST)


def test_caller_metadata_never_reaches_converse_as_a_top_level_field(gateway: Gateway) -> None:
    """Anthropic `metadata: {"user_id": ...}` has no Converse member. The body must stay within the
    Converse top level, so no top-level `metadata` or `user` is sent, which Converse would reject."""
    with _converse(gateway) as rig:
        assert rig.post(rig.body(metadata={"user_id": "synthetic-user-01"})).status_code == 200
        body: Final = body_of(rig.one_call())
        assert set(body) <= _CONVERSE_TOP_LEVEL, json.dumps(body)
