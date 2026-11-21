"""`/v1/chat/completions` on a Bedrock Converse deployment: the request Bedrock sees and the answer the caller gets.

A Bedrock Claude, Nova or other Converse model id behind the proxy is called with an OpenAI chat body, and
the proxy must translate it into one `POST /model/<url-encoded id>/converse` whose body uses Converse's own
shapes: `messages[].content[]` blocks of `{"text"}`, `{"image"}`, `{"toolUse"}`, `{"toolResult"}`,
`{"guardContent"}` and `{"cachePoint"}`, a top-level `system` list, `inferenceConfig` for sampling,
`toolConfig` for tools, `additionalModelRequestFields` for model specific fields such as Claude
`thinking`, and pass-through members such as `guardrailConfig`, `serviceTier` and `requestMetadata`.
The Converse answer (`output.message.content`, `stopReason`, `usage`) must come back as an OpenAI chat
completion with the same text, tool calls, reasoning, finish reason, token counts, cache counts and cost.

These tests own the Bedrock peer, so they read exactly what crossed the wire. The router core is
already Rust and the Converse transformation is next in line, so every mapping pinned here is a
contract a rewrite must keep. Existing files cover nearby ground (`test_bedrock_converse_*`,
`test_bedrock_native_structured_output_wire.py`, `streaming/test_dec1_stream_provider_translation.py`);
these tests target what they leave open.
"""

from __future__ import annotations

import json
from typing import Final

import pytest
from integration._support.client import Gateway
from integration._support.wire import Reply, Request
from integration.providers.dec1_bedrock_support import (
    CACHE_CREATION_COST,
    CACHE_READ_COST,
    CLAUDE_SONNET_45,
    INPUT_COST,
    NOVA_PRO,
    OUTPUT_COST,
    PRICES,
    TOKEN,
    WEATHER_TOOL,
    bedrock,
    body_of,
    converse_json,
    converse_reply,
    error_message,
    json_reply,
    marker,
    path_of,
    spend_row,
)
from pydantic import JsonValue

MODEL: Final = f"bedrock/{CLAUDE_SONNET_45}"
PNG_BASE64: Final = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGP4z8DwHwAFAAH/iZk9HQAAAABJRU5ErkJggg=="


def _ok(_: Request) -> Reply:
    return converse_reply()


def test_a_bare_claude_id_posts_once_to_the_url_encoded_converse_path_with_the_bearer_key(gateway: Gateway) -> None:
    """A `bedrock/anthropic.claude-sonnet-4-5-20250929-v1:0` deployment with `api_key` set is called once.

    Bedrock must receive exactly one `POST /model/anthropic.claude-sonnet-4-5-20250929-v1%3A0/converse`
    (the `:` percent-encoded by `encode_model_id`, `base_aws_llm.py:662`) with `Authorization: Bearer <key>`
    and `Content-Type: application/json`, because the id is in `BEDROCK_CONVERSE_MODELS`.
    """
    with bedrock(gateway, _ok, model=MODEL) as rig:
        response: Final = rig.chat()
        assert response.status_code == 200, response.text
        sent: Final = rig.one_post()
        assert sent.target == "/model/anthropic.claude-sonnet-4-5-20250929-v1%3A0/converse", sent.target
        assert sent.headers["authorization"] == f"Bearer {TOKEN}"
        assert sent.headers["content-type"].startswith("application/json")


def test_the_converse_text_answer_becomes_an_openai_chat_completion(gateway: Gateway) -> None:
    """Bedrock answers `output.message.content: [{"text": "pong"}]`, `stopReason: end_turn` and 11 in, 7 out tokens.

    The caller gets `object: chat.completion`, one choice with `message.role: assistant`,
    `message.content: "pong"`, `finish_reason: stop`, and `usage` 11, 7 and 18.
    """
    with bedrock(gateway, _ok, model=MODEL) as rig:
        response: Final = rig.chat()
        assert response.status_code == 200, response.text
        body: Final = response.json()
        assert body["object"] == "chat.completion"
        choice: Final = body["choices"][0]
        assert choice["message"]["role"] == "assistant"
        assert choice["message"]["content"] == "pong"
        assert choice["finish_reason"] == "stop"
        assert (body["usage"]["prompt_tokens"], body["usage"]["completion_tokens"], body["usage"]["total_tokens"]) == (
            11,
            7,
            18,
        )


def test_user_text_becomes_a_converse_text_block_and_system_becomes_the_top_level_system_list(gateway: Gateway) -> None:
    """A system message and a user message are sent.

    Converse has no `system` role, so Bedrock must receive `system: [{"text": <system text>}]` and
    `messages: [{"role": "user", "content": [{"text": <prompt>}]}]` with no message whose role is `system`.
    """
    prompt: Final = marker()
    with bedrock(gateway, _ok, model=MODEL) as rig:
        response: Final = rig.chat_body(
            {"messages": [{"role": "system", "content": "Answer tersely."}, {"role": "user", "content": prompt}]}
        )
        assert response.status_code == 200, response.text
        sent: Final = body_of(rig.one_post())
        assert sent["system"] == [{"text": "Answer tersely."}], sent
        assert sent["messages"] == [{"role": "user", "content": [{"text": prompt}]}], sent


def test_max_tokens_temperature_and_stop_land_in_inference_config(gateway: Gateway) -> None:
    """`max_tokens: 77`, `temperature: 0.25` and `stop: ["END"]` are OpenAI sampling params.

    Converse reads them only from `inferenceConfig`, so Bedrock must see `inferenceConfig.maxTokens == 77`,
    `temperature == 0.25` and `stopSequences == ["END"]`, and none of the OpenAI names at the top level.
    """
    with bedrock(gateway, _ok, model=MODEL) as rig:
        response: Final = rig.chat(max_tokens=77, temperature=0.25, stop=["END"])
        assert response.status_code == 200, response.text
        sent: Final = body_of(rig.one_post())
        config: Final = sent["inferenceConfig"]
        assert isinstance(config, dict)
        assert config["maxTokens"] == 77
        assert config["temperature"] == 0.25
        assert config["stopSequences"] == ["END"]
        assert not {"max_tokens", "temperature", "stop"} & set(sent), sent


def test_openai_function_tools_become_converse_tool_specs(gateway: Gateway) -> None:
    """One OpenAI function tool `get_weather` with a JSON schema is sent.

    Bedrock must receive `toolConfig.tools == [{"toolSpec": {"name": "get_weather", "description": ...,
    "inputSchema": {"json": <the same schema>}}}]`.
    """
    with bedrock(gateway, _ok, model=MODEL) as rig:
        response: Final = rig.chat(tools=[WEATHER_TOOL])
        assert response.status_code == 200, response.text
        sent: Final = body_of(rig.one_post())
        tools: Final = sent["toolConfig"]["tools"]  # pyright: ignore[reportIndexIssue, reportArgumentType, reportCallIssue]
        spec: Final = tools[0]["toolSpec"]  # pyright: ignore[reportIndexIssue, reportArgumentType, reportCallIssue, reportOptionalSubscript]
        assert spec["name"] == "get_weather"  # pyright: ignore[reportIndexIssue, reportArgumentType, reportCallIssue, reportOptionalSubscript]
        assert spec["description"] == "Look up the weather for a city"  # pyright: ignore[reportIndexIssue, reportArgumentType, reportCallIssue, reportOptionalSubscript]
        assert spec["inputSchema"]["json"]["properties"] == {"city": {"type": "string"}}  # pyright: ignore[reportIndexIssue, reportArgumentType, reportCallIssue, reportOptionalSubscript]


@pytest.mark.parametrize(
    ("tool_choice", "expected"),
    [
        ("required", {"any": {}}),
        ("auto", {"auto": {}}),
        ({"type": "function", "function": {"name": "get_weather"}}, {"tool": {"name": "get_weather"}}),
    ],
    ids=["required", "auto", "named"],
)
def test_openai_tool_choice_maps_to_the_converse_tool_choice_union(
    gateway: Gateway, tool_choice: JsonValue, expected: dict[str, JsonValue]
) -> None:
    """`tool_choice` `required`, `auto` and a named function become Converse `toolConfig.toolChoice`.

    Bedrock must see `{"any": {}}`, `{"auto": {}}` and `{"tool": {"name": "get_weather"}}` respectively
    (`map_tool_choice_values`, `converse_transformation.py:793`).
    """
    with bedrock(gateway, _ok, model=MODEL) as rig:
        response: Final = rig.chat(tools=[WEATHER_TOOL], tool_choice=tool_choice)
        assert response.status_code == 200, response.text
        sent: Final = body_of(rig.one_post())
        assert sent["toolConfig"]["toolChoice"] == expected, sent  # pyright: ignore[reportIndexIssue, reportArgumentType, reportCallIssue]


def test_tool_choice_none_is_refused_with_400_before_bedrock_is_called(gateway: Gateway) -> None:
    """Converse has no way to say "no tool", so `tool_choice: "none"` without `drop_params` raises.

    The caller must get HTTP 400 naming `tool_choice=none`, and the Bedrock peer must receive no POST.
    """
    with bedrock(gateway, _ok, model=MODEL) as rig:
        response: Final = rig.chat(tools=[WEATHER_TOOL], tool_choice="none")
        assert response.status_code == 400, response.text
        assert "tool_choice=none" in error_message(response), response.text
        assert rig.posts() == ()


def test_tool_choice_none_with_drop_params_is_sent_without_a_tool_choice(gateway: Gateway) -> None:
    """With `drop_params: true` on the request, `tool_choice: "none"` is dropped instead of refused.

    Bedrock must receive the tool specs but no `toolConfig.toolChoice` key.
    """
    with bedrock(gateway, _ok, model=MODEL) as rig:
        response: Final = rig.chat(tools=[WEATHER_TOOL], tool_choice="none", drop_params=True)
        assert response.status_code == 200, response.text
        sent: Final = body_of(rig.one_post())
        assert "toolChoice" not in sent["toolConfig"], sent  # pyright: ignore[reportOperatorIssue]


def test_a_tool_only_converse_answer_becomes_openai_tool_calls(gateway: Gateway) -> None:
    """Bedrock answers only a `toolUse` block (`toolUseId: tooluse_dec1`, `name: get_weather`, `input: {"city": "Oslo"}`).

    The caller must get `finish_reason: tool_calls` and one tool call with `id: tooluse_dec1`,
    `type: function`, `function.name: get_weather` and `function.arguments` that parses to
    `{"city": "Oslo"}`. The text content is empty (UNVERIFIED whether `""` or `null`: `_transform_response`
    sets `content_str = ""`, `converse_transformation.py:3056`, and `Message` may normalize it).
    """

    def respond(_: Request) -> Reply:
        return converse_reply(
            content=[{"toolUse": {"toolUseId": "tooluse_dec1", "name": "get_weather", "input": {"city": "Oslo"}}}],
            stop_reason="tool_use",
        )

    with bedrock(gateway, respond, model=MODEL) as rig:
        response: Final = rig.chat(tools=[WEATHER_TOOL])
        assert response.status_code == 200, response.text
        choice: Final = response.json()["choices"][0]
        assert choice["finish_reason"] == "tool_calls"
        assert not choice["message"]["content"], choice
        call: Final = choice["message"]["tool_calls"][0]
        assert call["id"] == "tooluse_dec1"
        assert call["type"] == "function"
        assert call["function"]["name"] == "get_weather"
        assert json.loads(call["function"]["arguments"]) == {"city": "Oslo"}


def test_text_followed_by_two_tool_uses_keeps_the_text_and_both_calls_in_order(gateway: Gateway) -> None:
    """Bedrock answers a text block and then two `toolUse` blocks.

    The caller must get `message.content` equal to the text and two tool calls in Bedrock's order
    (`tooluse_a` then `tooluse_b`), each with its own JSON arguments.
    """

    def respond(_: Request) -> Reply:
        return converse_reply(
            content=[
                {"text": "Checking both."},
                {"toolUse": {"toolUseId": "tooluse_a", "name": "get_weather", "input": {"city": "Oslo"}}},
                {"toolUse": {"toolUseId": "tooluse_b", "name": "get_weather", "input": {"city": "Lima"}}},
            ],
            stop_reason="tool_use",
        )

    with bedrock(gateway, respond, model=MODEL) as rig:
        response: Final = rig.chat(tools=[WEATHER_TOOL])
        assert response.status_code == 200, response.text
        message: Final = response.json()["choices"][0]["message"]
        assert message["content"] == "Checking both."
        assert [call["id"] for call in message["tool_calls"]] == ["tooluse_a", "tooluse_b"]
        assert [json.loads(call["function"]["arguments"]) for call in message["tool_calls"]] == [
            {"city": "Oslo"},
            {"city": "Lima"},
        ]


def test_a_tool_name_bedrock_rejects_is_rewritten_on_the_wire_and_restored_for_the_caller(gateway: Gateway) -> None:
    """A tool named `lookup.city` breaks Bedrock's `[a-zA-Z][a-zA-Z0-9_-]*` rule.

    Bedrock must receive the spec as `lookup_city` (`make_valid_bedrock_tool_name`, `factory.py:5882`), and
    when Bedrock calls `lookup_city` the caller must see the original name `lookup.city` again, because
    the rewrite is stored in `litellm.bedrock_tool_name_mappings`. UNVERIFIED: the reverse lookup on the
    response path; on a multi-worker proxy the in-memory mapping lives only in the worker that sent it.
    """
    tool: Final[dict[str, JsonValue]] = {
        "type": "function",
        "function": {"name": "lookup.city", "parameters": {"type": "object", "properties": {}}},
    }

    def respond(_: Request) -> Reply:
        return converse_reply(
            content=[{"toolUse": {"toolUseId": "tooluse_dot", "name": "lookup_city", "input": {}}}],
            stop_reason="tool_use",
        )

    with bedrock(gateway, respond, model=MODEL) as rig:
        response: Final = rig.chat(tools=[tool])
        assert response.status_code == 200, response.text
        sent: Final = body_of(rig.one_post())
        assert sent["toolConfig"]["tools"][0]["toolSpec"]["name"] == "lookup_city"  # pyright: ignore[reportIndexIssue, reportArgumentType, reportCallIssue, reportOptionalSubscript]
        call: Final = response.json()["choices"][0]["message"]["tool_calls"][0]
        assert call["function"]["name"] == "lookup.city"


def test_a_replayed_tool_call_and_tool_result_become_tool_use_and_tool_result_blocks(gateway: Gateway) -> None:
    """The caller replays an assistant turn with one tool call and a `tool` message answering it.

    Bedrock must receive the assistant turn as a `toolUse` block with the same `toolUseId`, `name` and
    parsed `input`, followed by a user turn whose content holds a `toolResult` block with that
    `toolUseId` and the tool's text as `content: [{"text": "sunny"}]`.
    """
    with bedrock(gateway, _ok, model=MODEL) as rig:
        response: Final = rig.chat_body(
            {
                "tools": [WEATHER_TOOL],
                "messages": [
                    {"role": "user", "content": marker()},
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "tooluse_replay",
                                "type": "function",
                                "function": {"name": "get_weather", "arguments": '{"city": "Oslo"}'},
                            }
                        ],
                    },
                    {"role": "tool", "tool_call_id": "tooluse_replay", "content": "sunny"},
                ],
            }
        )
        assert response.status_code == 200, response.text
        messages: Final = body_of(rig.one_post())["messages"]
        assert isinstance(messages, list)
        assistant: Final = messages[1]
        assert assistant["role"] == "assistant"  # pyright: ignore[reportIndexIssue, reportArgumentType, reportCallIssue, reportOptionalSubscript]
        assert {
            "toolUse": {"toolUseId": "tooluse_replay", "name": "get_weather", "input": {"city": "Oslo"}}
        } in assistant[  # pyright: ignore[reportIndexIssue, reportArgumentType, reportCallIssue, reportOptionalSubscript, reportOperatorIssue]
            "content"
        ]
        result_turn: Final = messages[2]
        assert result_turn["role"] == "user"  # pyright: ignore[reportIndexIssue, reportArgumentType, reportCallIssue, reportOptionalSubscript]
        result: Final = result_turn["content"][0]["toolResult"]  # pyright: ignore[reportIndexIssue, reportArgumentType, reportCallIssue, reportOptionalSubscript]
        assert result["toolUseId"] == "tooluse_replay"  # pyright: ignore[reportIndexIssue, reportArgumentType, reportCallIssue, reportOptionalSubscript]
        assert result["content"] == [{"text": "sunny"}]  # pyright: ignore[reportIndexIssue, reportArgumentType, reportCallIssue, reportOptionalSubscript]


def test_converse_reasoning_content_becomes_reasoning_content_and_a_signed_thinking_block(gateway: Gateway) -> None:
    """Bedrock answers a `reasoningContent.reasoningText` block (`text`, `signature`) before the answer text.

    The caller must get `message.content: "42"`, `message.reasoning_content` equal to the reasoning
    text, and `message.thinking_blocks == [{"type": "thinking", "thinking": <text>, "signature": "sig-dec1"}]`
    so the signature can be replayed on the next turn.
    """

    def respond(_: Request) -> Reply:
        return converse_reply(
            content=[
                {"reasoningContent": {"reasoningText": {"text": "add six sevens", "signature": "sig-dec1"}}},
                {"text": "42"},
            ]
        )

    with bedrock(gateway, respond, model=MODEL) as rig:
        response: Final = rig.chat()
        assert response.status_code == 200, response.text
        message: Final = response.json()["choices"][0]["message"]
        assert message["content"] == "42"
        assert message["reasoning_content"] == "add six sevens"
        assert message["thinking_blocks"] == [
            {"type": "thinking", "thinking": "add six sevens", "signature": "sig-dec1"}
        ]


def test_reasoning_effort_low_on_sonnet_4_5_becomes_a_1024_token_thinking_budget(gateway: Gateway) -> None:
    """`reasoning_effort: "low"` on Claude Sonnet 4.5 (a budget, not adaptive, thinking model) is sent.

    Bedrock must receive `additionalModelRequestFields.thinking == {"type": "enabled", "budget_tokens": 1024}`
    (`DEFAULT_REASONING_EFFORT_LOW_THINKING_BUDGET`, `litellm/constants.py:371`) and no top-level
    `reasoning_effort`. Assumes the shared proxy leaves that env override unset.
    """
    with bedrock(gateway, _ok, model=MODEL) as rig:
        response: Final = rig.chat(reasoning_effort="low", max_tokens=4096)
        assert response.status_code == 200, response.text
        sent: Final = body_of(rig.one_post())
        assert sent["additionalModelRequestFields"]["thinking"] == {"type": "enabled", "budget_tokens": 1024}, sent  # pyright: ignore[reportIndexIssue, reportArgumentType, reportCallIssue]
        assert "reasoning_effort" not in sent


def test_cache_control_on_a_user_text_part_becomes_a_following_cache_point(gateway: Gateway) -> None:
    """A user text part carrying `cache_control: {"type": "ephemeral"}` is sent to a Claude Converse model.

    Bedrock must receive that turn as `[{"text": <text>}, {"cachePoint": {"type": "default"}}]`, so the
    prefix the caller marked is cached.
    """
    prompt: Final = marker()
    with bedrock(gateway, _ok, model=MODEL) as rig:
        response: Final = rig.chat_body(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": prompt, "cache_control": {"type": "ephemeral"}}],
                    }
                ]
            }
        )
        assert response.status_code == 200, response.text
        sent: Final = body_of(rig.one_post())
        assert sent["messages"][0]["content"] == [{"text": prompt}, {"cachePoint": {"type": "default"}}], sent  # pyright: ignore[reportIndexIssue, reportArgumentType, reportCallIssue, reportOptionalSubscript]


def test_converse_cache_usage_is_reported_in_openai_fields_and_priced(gateway: Gateway) -> None:
    """Bedrock reports `inputTokens: 10`, `cacheReadInputTokens: 5`, `cacheWriteInputTokens: 9`, `outputTokens: 4`.

    `transform_usage` (`converse_transformation.py:2664`) adds both cache counts to `prompt_tokens` (24),
    reports `prompt_tokens_details.cached_tokens: 5`, `cache_read_input_tokens: 5` and
    `cache_creation_input_tokens: 9`, and the `x-litellm-response-cost` header must price each class at the
    deployment's own rate: 10 * 0.001 + 5 * 0.0001 + 9 * 0.003 + 4 * 0.002 = 0.0455.
    """

    def respond(_: Request) -> Reply:
        return converse_reply(
            usage={
                "inputTokens": 10,
                "outputTokens": 4,
                "totalTokens": 28,
                "cacheReadInputTokens": 5,
                "cacheWriteInputTokens": 9,
            }
        )

    with bedrock(gateway, respond, model=MODEL, **PRICES) as rig:
        response: Final = rig.chat()
        assert response.status_code == 200, response.text
        usage: Final = response.json()["usage"]
        assert usage["prompt_tokens"] == 24
        assert usage["completion_tokens"] == 4
        assert usage["prompt_tokens_details"]["cached_tokens"] == 5
        assert usage["cache_read_input_tokens"] == 5
        assert usage["cache_creation_input_tokens"] == 9
        expected: Final = 10 * INPUT_COST + 5 * CACHE_READ_COST + 9 * CACHE_CREATION_COST + 4 * OUTPUT_COST
        assert float(response.headers["x-litellm-response-cost"]) == pytest.approx(expected)


def test_a_converse_call_writes_one_spend_row_naming_bedrock_and_the_priced_tokens(gateway: Gateway) -> None:
    """A successful Converse chat call with 11 input and 7 output tokens and deployment prices.

    One `LiteLLM_SpendLogs` row must land with `call_type` `acompletion`, `custom_llm_provider` `bedrock`,
    `prompt_tokens` 11, `completion_tokens` 7, `model_group` equal to the public name and
    `spend = 11 * 0.001 + 7 * 0.002 = 0.025`.
    """
    with bedrock(gateway, _ok, model=MODEL, **PRICES) as rig:
        response: Final = rig.chat()
        assert response.status_code == 200, response.text
        row: Final = spend_row(response.json()["id"])
        assert row["call_type"] == "acompletion"
        assert row["custom_llm_provider"] == "bedrock"
        assert (row["prompt_tokens"], row["completion_tokens"]) == (11, 7)
        assert row["model_group"] == rig.model
        assert float(str(row["spend"])) == pytest.approx(11 * INPUT_COST + 7 * OUTPUT_COST)


def test_deployment_guardrail_config_reaches_converse_and_wraps_the_last_user_turn_as_guard_content(
    gateway: Gateway,
) -> None:
    """The deployment sets `guardrailConfig: {"guardrailIdentifier": "gr-dec1", "guardrailVersion": "1"}`.

    Bedrock must receive that object as the top-level `guardrailConfig`, and the trailing user turn's text
    as `{"guardContent": {"text": {"text": <prompt>}}}` (`_convert_consecutive_user_messages_to_guarded_text`,
    `converse_transformation.py:267`), so only the caller's newest input is evaluated by the guardrail.
    """
    guardrail: Final[dict[str, JsonValue]] = {"guardrailIdentifier": "gr-dec1", "guardrailVersion": "1"}
    prompt: Final = marker()
    with bedrock(gateway, _ok, model=MODEL, guardrailConfig=guardrail) as rig:
        response: Final = rig.chat(prompt)
        assert response.status_code == 200, response.text
        sent: Final = body_of(rig.one_post())
        assert sent["guardrailConfig"] == guardrail, sent
        assert sent["messages"][-1]["content"] == [{"guardContent": {"text": {"text": prompt}}}], sent  # pyright: ignore[reportIndexIssue, reportArgumentType, reportCallIssue, reportOptionalSubscript]


def test_request_level_guardrail_config_is_passed_through_to_converse(gateway: Gateway) -> None:
    """A caller sends `guardrailConfig` with `trace: "enabled"` in the chat body on a plain deployment.

    Bedrock must receive the same object as the top-level `guardrailConfig`; it is a supported Converse
    param (`get_config_blocks`, `converse_transformation.py:259`), not an unknown one to drop.
    """
    guardrail: Final[dict[str, JsonValue]] = {
        "guardrailIdentifier": "gr-request",
        "guardrailVersion": "DRAFT",
        "trace": "enabled",
    }
    with bedrock(gateway, _ok, model=MODEL) as rig:
        response: Final = rig.chat(guardrailConfig=guardrail)
        assert response.status_code == 200, response.text
        assert body_of(rig.one_post())["guardrailConfig"] == guardrail


def test_a_guardrail_intervention_is_answered_with_finish_reason_content_filter(gateway: Gateway) -> None:
    """Bedrock answers `stopReason: guardrail_intervened` with the guardrail's canned text.

    The caller must get HTTP 200, the canned text as `message.content`, and `finish_reason: content_filter`
    (`core_helpers.py:478`), so clients can tell a block from a normal stop.
    """

    def respond(_: Request) -> Reply:
        return converse_reply(content=[{"text": "Sorry, I cannot help with that."}], stop_reason="guardrail_intervened")

    with bedrock(gateway, respond, model=MODEL) as rig:
        response: Final = rig.chat()
        assert response.status_code == 200, response.text
        choice: Final = response.json()["choices"][0]
        assert choice["message"]["content"] == "Sorry, I cannot help with that."
        assert choice["finish_reason"] == "content_filter"


def test_a_guardrail_trace_in_the_converse_answer_reaches_the_caller(gateway: Gateway) -> None:
    """Bedrock adds `trace: {"guardrail": {...}}` to its answer when the guardrail trace is enabled.

    `_transform_response` copies it onto the response (`converse_transformation.py:3140`), so the caller's
    JSON must carry a top-level `trace` equal to what Bedrock sent. UNVERIFIED that the proxy's response
    serializer keeps this non-OpenAI attribute.
    """
    trace: Final[dict[str, JsonValue]] = {"guardrail": {"inputAssessment": {"gr-dec1": {"invocationMetrics": {}}}}}

    def respond(_: Request) -> Reply:
        return json_reply({**converse_json(), "trace": trace})

    with bedrock(gateway, respond, model=MODEL) as rig:
        response: Final = rig.chat()
        assert response.status_code == 200, response.text
        assert response.json()["trace"] == trace


@pytest.mark.parametrize(
    ("requested", "sent"),
    [("standard", "default"), ("priority", "priority"), ("FLEX", "flex"), ("reserved", "reserved")],
)
def test_service_tier_is_sent_as_a_converse_service_tier_block(gateway: Gateway, requested: str, sent: str) -> None:
    """OpenAI `service_tier` maps to Converse `serviceTier: {"type": <tier>}` (`_map_service_tier_param`).

    `standard` is an alias of `default`, names are lowercased, and `priority`, `flex` and `reserved` pass
    as is (#50516), so Bedrock must see `serviceTier == {"type": <expected>}`.
    """
    with bedrock(gateway, _ok, model=MODEL) as rig:
        response: Final = rig.chat(service_tier=requested)
        assert response.status_code == 200, response.text
        assert body_of(rig.one_post())["serviceTier"] == {"type": sent}


def test_service_tier_auto_sends_no_service_tier(gateway: Gateway) -> None:
    """`service_tier: "auto"` means "let the provider choose", so Bedrock must receive no `serviceTier` key."""
    with bedrock(gateway, _ok, model=MODEL) as rig:
        response: Final = rig.chat(service_tier="auto")
        assert response.status_code == 200, response.text
        assert "serviceTier" not in body_of(rig.one_post())


def test_a_service_tier_bedrock_cannot_serve_is_refused_with_400_before_the_call(gateway: Gateway) -> None:
    """`service_tier: "scale"` is not one of Bedrock's tiers (#50516).

    The caller must get HTTP 400 whose message says the model `does not support service_tier='scale'`,
    and no request reaches Bedrock.
    """
    with bedrock(gateway, _ok, model=MODEL) as rig:
        response: Final = rig.chat(service_tier="scale")
        assert response.status_code == 400, response.text
        assert "does not support service_tier='scale'" in error_message(response), response.text
        assert rig.posts() == ()


def test_bedrock_reported_service_tier_becomes_the_openai_service_tier_string(gateway: Gateway) -> None:
    """Bedrock echoes `serviceTier: {"type": "priority"}` in its answer.

    The caller must get the OpenAI shape, a top-level string `service_tier: "priority"`
    (`converse_transformation.py:3147`).
    """

    def respond(_: Request) -> Reply:
        return json_reply({**converse_json(), "serviceTier": {"type": "priority"}})

    with bedrock(gateway, respond, model=MODEL) as rig:
        response: Final = rig.chat(service_tier="priority")
        assert response.status_code == 200, response.text
        assert response.json()["service_tier"] == "priority"


def test_request_metadata_is_forwarded_as_converse_request_metadata(gateway: Gateway) -> None:
    """A caller sends `requestMetadata: {"team": "dec1", "purpose": "contracts"}` in the chat body.

    Bedrock must receive the same map as the top-level `requestMetadata`, which AWS uses to tag
    invocation logs. UNVERIFIED: whether proxy-owned metadata forwarding (`request_metadata.py`) adds
    keys when it is enabled; the shared proxy leaves it off, so the map must arrive unchanged.
    """
    metadata: Final[dict[str, JsonValue]] = {"team": "dec1", "purpose": "contracts"}
    with bedrock(gateway, _ok, model=MODEL) as rig:
        response: Final = rig.chat(requestMetadata=metadata)
        assert response.status_code == 200, response.text
        assert body_of(rig.one_post())["requestMetadata"] == metadata


def test_a_data_url_image_becomes_a_converse_image_block_with_raw_base64_bytes(gateway: Gateway) -> None:
    """A user part `image_url: data:image/png;base64,<png>` is sent.

    Converse takes images as `{"image": {"format": "png", "source": {"bytes": <base64>}}}`, so Bedrock must
    receive exactly that block with the caller's base64 payload and the `data:` prefix removed.
    """
    with bedrock(gateway, _ok, model=MODEL) as rig:
        response: Final = rig.chat_body(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": marker()},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{PNG_BASE64}"}},
                        ],
                    }
                ]
            }
        )
        assert response.status_code == 200, response.text
        content: Final = body_of(rig.one_post())["messages"][0]["content"]  # pyright: ignore[reportIndexIssue, reportArgumentType, reportCallIssue, reportOptionalSubscript]
        assert {"image": {"format": "png", "source": {"bytes": PNG_BASE64}}} in content  # pyright: ignore[reportOperatorIssue]


def test_an_s3_image_url_becomes_a_converse_s3_location_source(gateway: Gateway) -> None:
    """A user part `image_url: s3://dec1-bucket/cat.png` is sent (#50577).

    Bedrock must receive `{"image": {"format": "png", "source": {"s3Location": {"uri": "s3://dec1-bucket/cat.png"}}}}`
    instead of the proxy trying to download the object. UNVERIFIED: whether the cloud storage guard
    (`litellm_core_utils/cloud_storage_security.py`) refuses buckets a key was not granted; if it does,
    this test needs a key with that bucket allowed.
    """
    uri: Final = "s3://dec1-bucket/cat.png"
    with bedrock(gateway, _ok, model=MODEL) as rig:
        response: Final = rig.chat_body(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": marker()},
                            {"type": "image_url", "image_url": {"url": uri}},
                        ],
                    }
                ]
            }
        )
        assert response.status_code == 200, response.text
        content: Final = body_of(rig.one_post())["messages"][0]["content"]  # pyright: ignore[reportIndexIssue, reportArgumentType, reportCallIssue, reportOptionalSubscript]
        assert {"image": {"format": "png", "source": {"s3Location": {"uri": uri}}}} in content  # pyright: ignore[reportOperatorIssue]


def test_nova_pro_chat_goes_to_converse_with_the_same_body_shape(gateway: Gateway) -> None:
    """`bedrock/amazon.nova-pro-v1:0` is a Converse model, so a chat call posts to `/model/amazon.nova-pro-v1%3A0/converse`.

    Bedrock must receive the same Converse shape as for Claude: a user turn of `[{"text": <prompt>}]`
    and `inferenceConfig.maxTokens` 32, and the Nova answer text must reach the caller.
    """
    prompt: Final = marker()
    with bedrock(gateway, _ok, model=f"bedrock/{NOVA_PRO}") as rig:
        response: Final = rig.chat(prompt, max_tokens=32)
        assert response.status_code == 200, response.text
        sent_request: Final = rig.one_post()
        assert path_of(sent_request) == f"/model/{NOVA_PRO}/converse"
        sent: Final = body_of(sent_request)
        assert sent["messages"] == [{"role": "user", "content": [{"text": prompt}]}]
        assert sent["inferenceConfig"]["maxTokens"] == 32  # pyright: ignore[reportIndexIssue, reportArgumentType, reportCallIssue]
        assert response.json()["choices"][0]["message"]["content"] == "pong"


def test_the_converse_prefix_forces_converse_for_a_model_that_would_otherwise_use_invoke(gateway: Gateway) -> None:
    """`bedrock/converse/mistral.mistral-7b-instruct-v0:2` names a model whose cost map row is `bedrock` (Invoke).

    The explicit `converse/` route wins (`named_bedrock_route`, `common_utils.py:2083`), so Bedrock must see
    `POST /model/mistral.mistral-7b-instruct-v0%3A2/converse` and a Converse `messages` body, never the
    Mistral Invoke `prompt` string.
    """
    with bedrock(gateway, _ok, model="bedrock/converse/mistral.mistral-7b-instruct-v0:2") as rig:
        response: Final = rig.chat()
        assert response.status_code == 200, response.text
        sent: Final = rig.one_post()
        assert path_of(sent) == "/model/mistral.mistral-7b-instruct-v0:2/converse"
        assert "messages" in body_of(sent) and "prompt" not in body_of(sent)
