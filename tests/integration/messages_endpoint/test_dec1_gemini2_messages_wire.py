"""`/v1/messages` on a `gemini/` deployment: tool choice, stop sequences, structured output and replay.

The proxy answers Anthropic's Messages API for Gemini by translating the Anthropic request to an OpenAI
chat request (`LiteLLMAnthropicMessagesAdapter`, `litellm/llms/anthropic/messages/adapters/
transformation.py`) and then to Gemini. Round one pinned the basic shape (text, system, max_tokens,
thinking budget, one tool, streaming text, 429); this file pins the remaining knobs a Claude Code style
client sends: `tool_choice` any/tool/none map to `toolConfig.functionCallingConfig` ANY/ANY with
`allowed_function_names`/NONE (`translate_anthropic_tool_choice_to_openai`, `:920`), `stop_sequences`
become `generationConfig.stop_sequences`, `output_format` json_schema becomes
`response_mime_type: application/json` plus `response_json_schema` (`:1008`), a `tool_result` block becomes
a Gemini `function_response` named after the earlier `tool_use`, a base64 image block becomes `inline_data`,
and a `tool_use` id that Gemini signed comes back with the `__thought__` suffix stripped and the signature in
`provider_specific_fields.signature` (`_translate_openai_content_to_anthropic`, `:1519`), which the client
sends back so the next Gemini call carries `thoughtSignature` on the replayed `functionCall`.
"""

from __future__ import annotations

import re
from typing import Final

from integration._support.client import Gateway, list_value, object_value
from integration.providers.dec1_gemini2_support import (
    SIGNATURE,
    finish_reasons,
    frame,
    function_call,
    generation_config,
    sse_reply,
    stream_or_json,
)
from integration.providers.dec1_gemini_support import (
    at,
    gemini,
    gemini_echo,
    gemini_reply,
    json_of,
    marker,
    sse_payloads,
    usage_metadata,
)
from integration.providers.dec1_vertex_support import PNG_B64
from pydantic import JsonValue

_TOOL: Final[dict[str, JsonValue]] = {
    "name": "get_weather",
    "description": "Weather for a city",
    "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
}


def _body(content: JsonValue = None, **fields: JsonValue) -> dict[str, JsonValue]:
    return {
        "max_tokens": 256,
        "messages": [{"role": "user", "content": content if content is not None else f"hi {marker()}"}],
        **fields,
    }


def _blocks(body: dict[str, JsonValue]) -> list[dict[str, JsonValue]]:
    return [object_value(block) for block in list_value(body["content"])]


def test_tool_choice_any_becomes_function_calling_mode_any(gateway: Gateway) -> None:
    """`tool_choice: {"type": "any"}` reaches Gemini as `toolConfig.functionCallingConfig: {"mode": "ANY"}`
    (Anthropic any -> OpenAI required -> Gemini ANY)."""
    with gemini(gateway, gemini_echo) as rig:
        assert rig.post("/v1/messages", _body(tools=[_TOOL], tool_choice={"type": "any"})).status_code == 200
        assert at(rig.peer.one_body(), "toolConfig", "functionCallingConfig") == {"mode": "ANY"}


def test_a_named_tool_choice_restricts_the_allowed_function_names(gateway: Gateway) -> None:
    """`tool_choice: {"type": "tool", "name": "get_weather"}` reaches Gemini as
    `{"mode": "ANY", "allowed_function_names": ["get_weather"]}`."""
    with gemini(gateway, gemini_echo) as rig:
        choice: Final[dict[str, JsonValue]] = {"type": "tool", "name": "get_weather"}
        assert rig.post("/v1/messages", _body(tools=[_TOOL], tool_choice=choice)).status_code == 200
        assert at(rig.peer.one_body(), "toolConfig", "functionCallingConfig") == {
            "mode": "ANY",
            "allowed_function_names": ["get_weather"],
        }


def test_tool_choice_none_becomes_function_calling_mode_none(gateway: Gateway) -> None:
    """`tool_choice: {"type": "none"}` reaches Gemini as `{"mode": "NONE"}`, so Gemini answers in text even
    though tools are declared."""
    with gemini(gateway, gemini_echo) as rig:
        assert rig.post("/v1/messages", _body(tools=[_TOOL], tool_choice={"type": "none"})).status_code == 200
        assert at(rig.peer.one_body(), "toolConfig", "functionCallingConfig") == {"mode": "NONE"}


def test_stop_sequences_and_sampling_reach_the_generation_config(gateway: Gateway) -> None:
    """`stop_sequences: ["END", "STOP"]`, `temperature: 0.3` and `top_p: 0.8` reach Gemini as
    `generationConfig.stop_sequences: ["END", "STOP"]`, `temperature: 0.3` and `top_p: 0.8`
    (`_translate_stop_sequences_to_openai`, `:1182`)."""
    with gemini(gateway, gemini_echo) as rig:
        body: Final = _body(stop_sequences=["END", "STOP"], temperature=0.3, top_p=0.8)
        assert rig.post("/v1/messages", body).status_code == 200
        config: Final = generation_config(rig.peer.one_body())
        assert (config.get("stop_sequences"), config.get("temperature"), config.get("top_p")) == (
            ["END", "STOP"],
            0.3,
            0.8,
        )


def test_output_format_json_schema_becomes_a_gemini_json_schema(gateway: Gateway) -> None:
    """`output_format: {"type": "json_schema", "schema": {...answer: string...}}` reaches Gemini as
    `response_mime_type: application/json` and a `response_json_schema` whose `properties` still name
    `answer` as a string (`translate_anthropic_output_format_to_openai`, `:1008`)."""
    schema: Final[dict[str, JsonValue]] = {"type": "object", "properties": {"answer": {"type": "string"}}}
    with gemini(gateway, gemini_echo) as rig:
        body: Final = _body(output_format={"type": "json_schema", "schema": schema})
        assert rig.post("/v1/messages", body).status_code == 200
        config: Final = generation_config(rig.peer.one_body())
        assert config.get("response_mime_type") == "application/json"
        assert at(config, "response_json_schema", "properties", "answer", "type") == "string"


def test_a_tool_result_becomes_a_named_function_response(gateway: Gateway) -> None:
    """A conversation of user question, assistant `tool_use` (id `toolu_dec1`, `get_weather`) and user
    `tool_result` with text `sunny` reaches Gemini with a model turn holding the `functionCall` and a final
    turn holding `function_response: {"name": "get_weather", "response": {"content": "sunny"}}`
    (`convert_to_gemini_tool_call_result`, `factory.py:1648`)."""
    messages: Final[list[JsonValue]] = [
        {"role": "user", "content": f"weather in Oslo? {marker()}"},
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "toolu_dec1", "name": "get_weather", "input": {"city": "Oslo"}}],
        },
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_dec1", "content": "sunny"}]},
    ]
    with gemini(gateway, gemini_echo) as rig:
        assert rig.post("/v1/messages", {"max_tokens": 64, "messages": messages, "tools": [_TOOL]}).status_code == 200
        contents: Final = list_value(rig.peer.one_body()["contents"])
        model_parts: Final = list_value(object_value(contents[1])["parts"])
        assert object_value(object_value(model_parts[0])["function_call"])["name"] == "get_weather"
        last_parts: Final = list_value(object_value(contents[-1])["parts"])
        assert object_value(last_parts[0])["function_response"] == {
            "name": "get_weather",
            "response": {"content": "sunny"},
        }


def test_a_base64_image_block_becomes_inline_data(gateway: Gateway) -> None:
    """An Anthropic `image` block with a base64 PNG source reaches Gemini as an `inline_data` part with
    `mime_type: image/png` and the same base64 data. UNVERIFIED: the part key spelling (`inline_data`) is
    the one the chat path writes; the adapter routes images through the same conversion."""
    content: Final[list[JsonValue]] = [
        {"type": "text", "text": f"what is this {marker()}"},
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": PNG_B64}},
    ]
    with gemini(gateway, gemini_echo) as rig:
        assert rig.post("/v1/messages", _body(content)).status_code == 200
        parts: Final = list_value(object_value(list_value(rig.peer.one_body()["contents"])[0])["parts"])
        inline: Final = [
            object_value(object_value(part)["inline_data"]) for part in parts if "inline_data" in object_value(part)
        ]
        assert len(inline) == 1
        assert (inline[0]["mime_type"], inline[0]["data"]) == ("image/png", PNG_B64)


def test_a_signed_tool_use_id_is_cleaned_and_the_signature_kept(gateway: Gateway) -> None:
    """A Gemini `functionCall` with `thoughtSignature` comes back as a `tool_use` block whose `id` has no
    `__thought__` suffix and matches `^[a-zA-Z0-9_-]+$`, with `provider_specific_fields.signature` equal to
    the signature (`normalize_anthropic_tool_use_id`, `anthropic/common_utils.py:2301`)."""
    part: Final = function_call("get_weather", {"city": "Oslo"}, signature=SIGNATURE)
    with gemini(gateway, lambda _r: gemini_reply(parts=[part])) as rig:
        body: Final = json_of(rig.post("/v1/messages", _body(tools=[_TOOL])))
        tool_use: Final = next(block for block in _blocks(body) if block["type"] == "tool_use")
        assert "__thought__" not in str(tool_use["id"])
        assert re.fullmatch(r"[a-zA-Z0-9_-]+", str(tool_use["id"]))
        assert object_value(tool_use["provider_specific_fields"])["signature"] == SIGNATURE


def test_a_replayed_tool_use_signature_reaches_gemini_as_thought_signature(gateway: Gateway) -> None:
    """When the client sends back the assistant `tool_use` with `provider_specific_fields: {"signature": <sig>}`
    and then its `tool_result`, the replayed `functionCall` part in the Gemini model turn carries
    `thoughtSignature: <sig>`, which Gemini 3 requires on every replayed call
    (`_extract_signature_from_tool_use_content`, adapter `:723`). UNVERIFIED: the part key spelling
    `thoughtSignature` on the outgoing part."""
    messages: Final[list[JsonValue]] = [
        {"role": "user", "content": f"weather in Oslo? {marker()}"},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu_dec1_signed",
                    "name": "get_weather",
                    "input": {"city": "Oslo"},
                    "provider_specific_fields": {"signature": SIGNATURE},
                }
            ],
        },
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_dec1_signed", "content": "sunny"}]},
    ]
    with gemini(gateway, gemini_echo) as rig:
        assert rig.post("/v1/messages", {"max_tokens": 64, "messages": messages, "tools": [_TOOL]}).status_code == 200
        model_turn: Final = object_value(list_value(rig.peer.one_body()["contents"])[1])
        call_parts: Final = [
            object_value(part) for part in list_value(model_turn["parts"]) if "function_call" in object_value(part)
        ]
        assert call_parts and call_parts[0].get("thoughtSignature") == SIGNATURE, model_turn


def test_a_streamed_tool_use_ends_with_stop_reason_tool_use(gateway: Gateway) -> None:
    """A streamed Gemini `functionCall` arrives as Anthropic events: a `content_block_start` with a
    `tool_use` block named `get_weather`, `input_json_delta` deltas whose `partial_json` joins to
    `{"city": "Oslo"}`, and a `message_delta` with `stop_reason: "tool_use"`; no OpenAI chat chunk leaks into the stream."""
    frames: Final = [frame([function_call("get_weather", {"city": "Oslo"})], finish="STOP", usage=usage_metadata())]
    with gemini(gateway, stream_or_json(sse_reply(frames), gemini_reply())) as rig:
        response, text = rig.stream("/v1/messages", _body(tools=[_TOOL]))
        assert response.status_code == 200, text
        events: Final = sse_payloads(text)
        starts: Final = [
            object_value(event["content_block"]) for event in events if event.get("type") == "content_block_start"
        ]
        assert any(block.get("type") == "tool_use" and block.get("name") == "get_weather" for block in starts)
        partial: Final = "".join(
            str(object_value(event["delta"]).get("partial_json", ""))
            for event in events
            if event.get("type") == "content_block_delta"
        )
        assert partial.replace(" ", "") == '{"city":"Oslo"}'
        stops: Final = [
            object_value(event["delta"]).get("stop_reason") for event in events if event.get("type") == "message_delta"
        ]
        assert stops[-1] == "tool_use"
        assert finish_reasons(text) == ()
