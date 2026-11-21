"""`gemini/` chat: tool call identity, parallel calls, and how a stream ends.

Tool calls. Every Gemini `functionCall` part becomes one OpenAI tool call with a running `index` and an
id `call_<28 hex>` minted by the proxy, unless Gemini 3 sent its own `functionCall.id`, which is kept so
it can be echoed back. When the part carries a `thoughtSignature`, the signature is appended to the id as
`<id>__thought__<signature>` (so plain OpenAI clients replay it without knowing about it) and is also put
on the tool call's `provider_specific_fields.thought_signature`; a deployment can turn the id embedding
off with `thought_signature_in_tool_call_id: false` (`_transform_parts`,
`vertex_and_google_ai_studio_gemini.py:2383`, `encode_tool_call_id_with_signature`,
`factory.py:1243`). Any tool call forces `finish_reason: tool_calls` whatever Google's finish reason was
(`_check_finish_reason`, `:2776`).

Streams. Tool call indexes keep counting across frames (`cumulative_tool_call_index`), a prompt blocked
in a stream frame ends the stream with one `content_filter` chunk (`_check_prompt_level_content_filter`),
a candidate safety stop ends with `content_filter`, and an `{"error": ...}` frame that arrives after
HTTP 200 is raised as a provider error (`_check_streaming_error`, `:4282`).

The Vertex suite already pins a single tool call, `tool_choice` modes and a single streamed call; this
file covers what those leave out. Every prompt carries a fresh marker.
"""

from __future__ import annotations

import json
import re
from typing import Final

from integration._support.client import Gateway, list_value, object_value, string_value
from integration.providers.dec1_gemini2_support import (
    SIGNATURE,
    WEATHER_TOOL,
    deltas,
    finish_reasons,
    frame,
    function_call,
    sse_reply,
    stream_or_json,
    streamed_tool_calls,
)
from integration.providers.dec1_gemini_support import (
    GEMINI_3_FLASH,
    choice,
    gemini,
    gemini_reply,
    marker,
    message_of,
    sse_payloads,
    usage_metadata,
)
from pydantic import JsonValue

_MINTED_ID: Final = re.compile(r"^call_[0-9a-f]{28}$")


def _tool_calls(message: dict[str, JsonValue]) -> list[dict[str, JsonValue]]:
    return [object_value(call) for call in list_value(message.get("tool_calls") or [])]


def _stream_body() -> dict[str, JsonValue]:
    return {"messages": [{"role": "user", "content": f"weather {marker()}"}], "tools": [WEATHER_TOOL]}


# ---------------------------------------------------------------- non-streamed tool calls


def test_two_function_calls_become_two_indexed_tool_calls(gateway: Gateway) -> None:
    """A candidate with two `functionCall` parts (Paris, then Oslo) comes back as two tool calls in that
    order with `index` 0 and 1, both named `get_weather`, with arguments `{"city": "Paris"}` and
    `{"city": "Oslo"}` as JSON strings."""
    parts: Final[list[JsonValue]] = [
        function_call("get_weather", {"city": "Paris"}),
        function_call("get_weather", {"city": "Oslo"}),
    ]
    with gemini(gateway, lambda _r: gemini_reply(parts=parts)) as rig:
        calls: Final = _tool_calls(message_of(rig.chat(tools=[WEATHER_TOOL])))
        assert [call["index"] for call in calls] == [0, 1]
        assert [json.loads(string_value(object_value(call["function"])["arguments"])) for call in calls] == [
            {"city": "Paris"},
            {"city": "Oslo"},
        ]


def test_a_tool_call_forces_the_tool_calls_finish_reason_over_stop(gateway: Gateway) -> None:
    """Google ends a function call turn with `finishReason: STOP`; the caller still gets
    `finish_reason: "tool_calls"`, because a message with tool calls wins over the mapped reason."""
    with gemini(gateway, lambda _r: gemini_reply(parts=[function_call("get_weather", {"city": "Rome"})])) as rig:
        assert choice(rig.chat(tools=[WEATHER_TOOL]))["finish_reason"] == "tool_calls"


def test_a_tool_call_without_a_google_id_gets_a_minted_call_id(gateway: Gateway) -> None:
    """A `functionCall` with no `id` and no signature gets an id of the form `call_` plus 28 lowercase hex
    characters (`f"call_{uuid.uuid4().hex[:28]}"`), and two calls in one answer get different ids."""
    parts: Final[list[JsonValue]] = [
        function_call("get_weather", {"city": "Lima"}),
        function_call("get_weather", {"city": "Quito"}),
    ]
    with gemini(gateway, lambda _r: gemini_reply(parts=parts)) as rig:
        ids: Final = [string_value(call["id"]) for call in _tool_calls(message_of(rig.chat(tools=[WEATHER_TOOL])))]
        assert all(_MINTED_ID.match(item) for item in ids), ids
        assert len(set(ids)) == 2


def test_a_thought_signature_is_appended_to_the_tool_call_id(gateway: Gateway) -> None:
    """A `functionCall` part with `thoughtSignature: <sig>` comes back with id `call_<28 hex>__thought__<sig>`
    and with `provider_specific_fields.thought_signature: <sig>` on the tool call, so a client that only
    echoes ids still replays the signature Gemini requires."""
    part: Final = function_call("get_weather", {"city": "Kyiv"}, signature=SIGNATURE)
    with gemini(gateway, lambda _r: gemini_reply(parts=[part])) as rig:
        call: Final = _tool_calls(message_of(rig.chat(tools=[WEATHER_TOOL])))[0]
        prefix, separator, signature = string_value(call["id"]).partition("__thought__")
        assert (separator, signature) == ("__thought__", SIGNATURE)
        assert _MINTED_ID.match(prefix), prefix
        assert object_value(call["provider_specific_fields"])["thought_signature"] == SIGNATURE


def test_a_deployment_can_keep_the_signature_out_of_the_tool_call_id(gateway: Gateway) -> None:
    """With `thought_signature_in_tool_call_id: false` on the deployment the same signed part comes back
    with a plain `call_<28 hex>` id, while `provider_specific_fields.thought_signature` still carries the
    signature (`embeds_thought_signature_in_tool_call_id`, `model_param_helper.py:49`)."""
    part: Final = function_call("get_weather", {"city": "Kyiv"}, signature=SIGNATURE)
    with gemini(gateway, lambda _r: gemini_reply(parts=[part]), thought_signature_in_tool_call_id=False) as rig:
        call: Final = _tool_calls(message_of(rig.chat(tools=[WEATHER_TOOL])))[0]
        assert _MINTED_ID.match(string_value(call["id"])), call["id"]
        assert object_value(call["provider_specific_fields"])["thought_signature"] == SIGNATURE


def test_a_gemini_3_native_call_id_is_kept_as_the_tool_call_id(gateway: Gateway) -> None:
    """When Gemini 3 returns `functionCall.id: "fc-dec1-1"` (and no signature) the tool call id is exactly
    `fc-dec1-1`, not a minted `call_` id, so the caller can answer with that id."""
    part: Final = function_call("get_weather", {"city": "Bern"}, call_id="fc-dec1-1")
    with gemini(gateway, lambda _r: gemini_reply(parts=[part], model=GEMINI_3_FLASH), backend=GEMINI_3_FLASH) as rig:
        assert _tool_calls(message_of(rig.chat(tools=[WEATHER_TOOL])))[0]["id"] == "fc-dec1-1"


def test_non_ascii_arguments_are_returned_unescaped(gateway: Gateway) -> None:
    """Arguments `{"city": "Zürich"}` come back as the JSON string `{"city": "Zürich"}` with the umlaut
    kept, not `\\u00fc` (`json.dumps(..., ensure_ascii=False)` in `_transform_parts`)."""
    with gemini(gateway, lambda _r: gemini_reply(parts=[function_call("get_weather", {"city": "Zürich"})])) as rig:
        call: Final = _tool_calls(message_of(rig.chat(tools=[WEATHER_TOOL])))[0]
        assert object_value(call["function"])["arguments"] == '{"city": "Zürich"}'


def test_text_and_a_tool_call_in_one_turn_are_both_returned(gateway: Gateway) -> None:
    """A candidate with a text part "Checking." followed by a `functionCall` returns `content: "Checking."`
    and one tool call, with `finish_reason: "tool_calls"`."""
    parts: Final[list[JsonValue]] = [{"text": "Checking."}, function_call("get_weather", {"city": "Riga"})]
    with gemini(gateway, lambda _r: gemini_reply(parts=parts)) as rig:
        response: Final = rig.chat(tools=[WEATHER_TOOL])
        message: Final = message_of(response)
        assert message["content"] == "Checking."
        assert len(_tool_calls(message)) == 1
        assert choice(response)["finish_reason"] == "tool_calls"


# ---------------------------------------------------------------- streamed tool calls and stream ends


def test_streamed_calls_across_frames_keep_counting_indexes(gateway: Gateway) -> None:
    """Two frames, each with one `functionCall` (Paris then Oslo), arrive as tool call deltas with `index`
    0 and then 1, not 0 twice, and the stream finishes with `tool_calls`."""
    frames: Final = [
        frame([function_call("get_weather", {"city": "Paris"})]),
        frame([function_call("get_weather", {"city": "Oslo"})], finish="STOP", usage=usage_metadata()),
    ]
    with gemini(gateway, stream_or_json(sse_reply(frames), gemini_reply())) as rig:
        response, text = rig.stream("/v1/chat/completions", _stream_body())
        assert response.status_code == 200, text
        calls: Final = streamed_tool_calls(text)
        assert [call["index"] for call in calls] == [0, 1]
        assert finish_reasons(text)[-1] == "tool_calls"


def test_a_streamed_signed_call_carries_the_signature_in_its_id(gateway: Gateway) -> None:
    """A streamed `functionCall` part with `thoughtSignature` arrives as a tool call delta whose id ends with
    `__thought__<sig>`, the same id format as the non-streamed answer. UNVERIFIED: that the stream wrapper
    does not rewrite the id of the first tool call delta."""
    frames: Final = [
        frame(
            [function_call("get_weather", {"city": "Oslo"}, signature=SIGNATURE)], finish="STOP", usage=usage_metadata()
        )
    ]
    with gemini(gateway, stream_or_json(sse_reply(frames), gemini_reply())) as rig:
        _response, text = rig.stream("/v1/chat/completions", _stream_body())
        ids: Final = [string_value(call["id"]) for call in streamed_tool_calls(text) if call.get("id")]
        assert ids and ids[0].endswith(f"__thought__{SIGNATURE}"), ids


def test_a_streamed_safety_stop_ends_with_content_filter(gateway: Gateway) -> None:
    """A stream whose last frame says `finishReason: SAFETY` ends with a chunk whose `finish_reason` is
    `content_filter`, after the text that arrived before it."""
    frames: Final = [
        frame([{"text": "partial "}]),
        frame([], finish="SAFETY", usage=usage_metadata()),
    ]
    with gemini(gateway, stream_or_json(sse_reply(frames), gemini_reply())) as rig:
        _response, text = rig.stream("/v1/chat/completions", {"messages": [{"role": "user", "content": marker()}]})
        assert finish_reasons(text)[-1] == "content_filter"
        assert "".join(str(delta.get("content") or "") for delta in deltas(text)) == "partial "


def test_a_prompt_blocked_in_the_stream_ends_with_one_content_filter_chunk(gateway: Gateway) -> None:
    """A single stream frame `{"promptFeedback": {"blockReason": "PROHIBITED_CONTENT"}}` with no candidates
    produces a chunk with `finish_reason: "content_filter"` and no content, matching the non-streamed
    blocked answer (`_check_prompt_level_content_filter`, `vertex_and_google_ai_studio_gemini.py:2800`)."""
    blocked: Final[dict[str, JsonValue]] = {
        "promptFeedback": {"blockReason": "PROHIBITED_CONTENT"},
        "usageMetadata": {"promptTokenCount": 9, "totalTokenCount": 9},
        "modelVersion": "gemini-2.5-flash",
    }
    with gemini(gateway, stream_or_json(sse_reply([blocked]), gemini_reply())) as rig:
        response, text = rig.stream("/v1/chat/completions", {"messages": [{"role": "user", "content": marker()}]})
        assert response.status_code == 200, text
        assert finish_reasons(text) == ("content_filter",)
        assert not any(delta.get("content") for delta in deltas(text))


def test_an_error_frame_after_text_ends_the_stream_with_the_google_error(gateway: Gateway) -> None:
    """Google can answer HTTP 200 and then send `{"error": {"code": 429, "status": "RESOURCE_EXHAUSTED",
    "message": "dec1 quota"}}` as an SSE frame. The caller first receives the text "hello " and then an
    error event whose text names `RESOURCE_EXHAUSTED - dec1 quota` (`_check_streaming_error` builds
    `f"{status} - {message}"`). UNVERIFIED: the exact error event shape the proxy writes mid-stream."""
    error_frame: Final[dict[str, JsonValue]] = {
        "error": {"code": 429, "status": "RESOURCE_EXHAUSTED", "message": "dec1 quota"}
    }
    frames: Final = [frame([{"text": "hello "}]), error_frame]
    with gemini(gateway, stream_or_json(sse_reply(frames), gemini_reply()), num_retries=0) as rig:
        _response, text = rig.stream("/v1/chat/completions", {"messages": [{"role": "user", "content": marker()}]})
        assert "hello " in text
        assert "RESOURCE_EXHAUSTED - dec1 quota" in text


def test_an_error_frame_before_any_text_is_answered_with_its_code(gateway: Gateway) -> None:
    """When the very first frame is `{"error": {"code": 429, ...}}` the proxy has not sent anything yet and
    answers the stream request with status 429 instead of 200. UNVERIFIED: that the proxy reads the first
    chunk before committing the status line for `gemini/` streams."""
    error_frame: Final[dict[str, JsonValue]] = {
        "error": {"code": 429, "status": "RESOURCE_EXHAUSTED", "message": "dec1 first-frame quota"}
    }
    with gemini(gateway, stream_or_json(sse_reply([error_frame]), gemini_reply()), num_retries=0) as rig:
        response, text = rig.stream("/v1/chat/completions", {"messages": [{"role": "user", "content": marker()}]})
        assert response.status_code == 429, text


def test_streamed_cached_tokens_reach_the_final_usage(gateway: Gateway) -> None:
    """With `stream_options.include_usage` and a last frame reporting `promptTokenCount: 4000`,
    `cachedContentTokenCount: 3000`, the usage chunk carries `prompt_tokens: 4000` and
    `prompt_tokens_details.cached_tokens: 3000` (`_calculate_usage`)."""
    frames: Final = [
        frame([{"text": "cached "}]),
        frame([{"text": "answer"}], finish="STOP", usage=usage_metadata(4000, 2, cachedContentTokenCount=3000)),
    ]
    with gemini(gateway, stream_or_json(sse_reply(frames), gemini_reply())) as rig:
        _response, text = rig.stream(
            "/v1/chat/completions",
            {"messages": [{"role": "user", "content": marker()}], "stream_options": {"include_usage": True}},
        )
        usages: Final = [object_value(item["usage"]) for item in sse_payloads(text) if item.get("usage")]
        final: Final = usages[-1]
        assert final["prompt_tokens"] == 4000
        assert object_value(final["prompt_tokens_details"])["cached_tokens"] == 3000
