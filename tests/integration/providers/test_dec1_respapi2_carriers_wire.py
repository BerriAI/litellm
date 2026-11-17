"""`/v1/responses` on Gemini (AI Studio), Bedrock Converse and Groq through the chat completions bridge.

None of these providers has a native Responses config (`litellm/utils.py:10709`), so the proxy rewrites the
Responses request into a chat completion (`litellm/responses/litellm_completion_transformation/transformation.py`)
and lets the provider's own chat transform build the wire request: Gemini `generateContent`, Bedrock `converse`
and Groq's OpenAI compatible `chat/completions`. The answer is rewritten back into a Responses object. The caller
must not be able to tell which carrier served it: the same `status`, `incomplete_details`, output items and usage
names come back. Round 1 pinned the bridge on OpenAI chat, DeepSeek and Anthropic; the Vertex and Bedrock suites
pin the basic Gemini and Converse mappings (`test_dec1_vertex3_responses_bridge_wire.py`,
`test_dec1_bedrock_responses_wire.py`). This file pins what those leave out: `tool_choice` and
`parallel_tool_calls` mapping, finish reasons, sampling params and Groq as a carrier.
"""

from __future__ import annotations

import json
from typing import Final

import pytest
from integration._support.client import Gateway, list_value, object_value
from integration._support.wire import Reply
from integration.providers.dec1_respapi2_support import (
    GEMINI_PATH,
    GROQ_BACKEND,
    WEATHER_TOOL,
    bedrock,
    converse_answer,
    every_respond,
    gemini,
    gemini_answer,
    groq,
    unique_text,
)
from integration.providers.dec1_respapi_support import (
    CHAT_PATH,
    PROVIDER_KEY,
    REPLY_TEXT,
    Call,
    chat_json,
    chat_messages,
    error_message,
    json_reply,
    output_items,
    output_text,
    peer,
    responses,
    served,
    tool_call,
)
from pydantic import JsonValue

_GROQ_SERVICE_TIER_BUG: Final = (
    "BUG: a Groq reply without service_tier fails with AttributeError on ModelResponse.service_tier"
    " (groq/chat/transformation.py:329 calls getattr without a default)"
)


def _tool_config(call: Call) -> dict[str, JsonValue]:
    return object_value(object_value(call.body["toolConfig"])["functionCallingConfig"])


# ---------------------------------------------------------------- Gemini


def test_gemini_instructions_become_the_system_instruction_not_a_content_turn(gateway: Gateway) -> None:
    """`instructions: "Answer in one word."` with a string input on a Gemini deployment. The bridge makes a
    leading system message (`transformation.py:2960`) and the Gemini transform lifts it out of `contents`
    (`vertex_ai/gemini/transformation.py:1244`), so the body must carry `system_instruction` holding that text and
    `contents` must be exactly one `user` turn with the input."""
    prompt: Final = unique_text()
    with peer(every_respond) as provider, gateway.scenario() as scenario:
        served(responses(gateway, gemini(scenario, provider), input=prompt, instructions="Answer in one word."))
        call: Final = provider.only()
    assert call.path == GEMINI_PATH
    assert "Answer in one word." in str(call.body["system_instruction"])
    assert [object_value(entry)["role"] for entry in list_value(call.body["contents"])] == ["user"]


def test_gemini_max_output_tokens_and_sampling_reach_the_generation_config(gateway: Gateway) -> None:
    """`max_output_tokens: 48`, `temperature: 0.3` and `top_p: 0.6` on a Gemini deployment become
    `generationConfig.max_output_tokens: 48`, `temperature: 0.3` and `top_p: 0.6` (the bridge maps
    `max_output_tokens` to chat `max_tokens`, `transformation.py:1424`, and Gemini renames it)."""
    with peer(every_respond) as provider, gateway.scenario() as scenario:
        served(responses(gateway, gemini(scenario, provider), max_output_tokens=48, temperature=0.3, top_p=0.6))
        config: Final = object_value(provider.only().body["generationConfig"])
    assert (config["max_output_tokens"], config["temperature"], config["top_p"]) == (48, 0.3, 0.6)


def test_gemini_tool_choice_none_becomes_function_calling_mode_none(gateway: Gateway) -> None:
    """`tool_choice: "none"` with the `get_weather` tool reaches Gemini as
    `toolConfig.functionCallingConfig.mode: "NONE"` (`vertex_and_google_ai_studio_gemini.py:1020`)."""
    with peer(every_respond) as provider, gateway.scenario() as scenario:
        served(responses(gateway, gemini(scenario, provider), tools=[WEATHER_TOOL], tool_choice="none"))
        assert _tool_config(provider.only())["mode"] == "NONE"


def test_gemini_named_tool_choice_forces_that_function_with_mode_any(gateway: Gateway) -> None:
    """`tool_choice: {"type": "function", "name": "get_weather"}` reaches Gemini as `mode: "ANY"` with
    `allowed_function_names: ["get_weather"]` (`vertex_and_google_ai_studio_gemini.py:1026`), so Gemini can only
    call that one function."""
    with peer(every_respond) as provider, gateway.scenario() as scenario:
        served(
            responses(
                gateway,
                gemini(scenario, provider),
                tools=[WEATHER_TOOL],
                tool_choice={"type": "function", "name": "get_weather"},
            )
        )
        config: Final = _tool_config(provider.only())
    assert (config["mode"], config["allowed_function_names"]) == ("ANY", ["get_weather"])


def test_gemini_max_tokens_finish_is_an_incomplete_response(gateway: Gateway) -> None:
    """Gemini answers `finishReason: MAX_TOKENS`, which is the chat `length` finish. The Responses answer must be
    `status: incomplete` with `incomplete_details: {"reason": "max_output_tokens"}` (`transformation.py:217`)."""

    def respond(call: Call) -> Reply:
        return json_reply(gemini_answer(finish="MAX_TOKENS")) if call.method == "POST" else every_respond(call)

    with peer(respond) as provider, gateway.scenario() as scenario:
        body: Final = served(responses(gateway, gemini(scenario, provider)))
    assert body["status"] == "incomplete"
    assert body["incomplete_details"] == {"reason": "max_output_tokens"}


def test_gemini_safety_finish_is_incomplete_for_content_filter(gateway: Gateway) -> None:
    """Gemini answers `finishReason: SAFETY` (chat `content_filter`). The Responses answer must be `status:
    incomplete` with `incomplete_details.reason: content_filter`, never `completed`."""

    def respond(call: Call) -> Reply:
        return json_reply(gemini_answer(finish="SAFETY")) if call.method == "POST" else every_respond(call)

    with peer(respond) as provider, gateway.scenario() as scenario:
        body: Final = served(responses(gateway, gemini(scenario, provider)))
    assert (body["status"], object_value(body["incomplete_details"])["reason"]) == ("incomplete", "content_filter")


def test_gemini_two_function_calls_come_back_as_two_function_call_items_in_order(gateway: Gateway) -> None:
    """Gemini answers two `functionCall` parts (`get_weather` Oslo then Rome) in one candidate. The Responses
    `output` must hold two `function_call` items named `get_weather` with arguments `{"city": "Oslo"}` then
    `{"city": "Rome"}` (as parsed JSON), in the order Gemini sent them."""
    parts: Final[list[dict[str, JsonValue]]] = [
        {"functionCall": {"name": "get_weather", "args": {"city": "Oslo"}}},
        {"functionCall": {"name": "get_weather", "args": {"city": "Rome"}}},
    ]

    def respond(call: Call) -> Reply:
        return json_reply(gemini_answer(parts=parts)) if call.method == "POST" else every_respond(call)

    with peer(respond) as provider, gateway.scenario() as scenario:
        body: Final = served(responses(gateway, gemini(scenario, provider), tools=[WEATHER_TOOL]))
    calls: Final = output_items(body, "function_call")
    assert [(item["name"], json.loads(str(item["arguments"]))) for item in calls] == [
        ("get_weather", {"city": "Oslo"}),
        ("get_weather", {"city": "Rome"}),
    ]


# ---------------------------------------------------------------- Bedrock Converse


def test_bedrock_required_tool_choice_becomes_converse_tool_choice_any(gateway: Gateway) -> None:
    """`tool_choice: "required"` with the `get_weather` tool on Bedrock Claude Sonnet 4.5 reaches Converse as
    `toolConfig.toolChoice: {"any": {}}` (`converse_transformation.py:804`, `:2264`)."""
    with peer(every_respond) as provider, gateway.scenario() as scenario:
        served(responses(gateway, bedrock(scenario, provider), tools=[WEATHER_TOOL], tool_choice="required"))
        call: Final = provider.only()
    assert call.path.endswith("/converse"), call.path
    assert object_value(call.body["toolConfig"])["toolChoice"] == {"any": {}}


def test_bedrock_named_tool_choice_becomes_converse_tool_choice_tool(gateway: Gateway) -> None:
    """`tool_choice: {"type": "function", "name": "get_weather"}` reaches Converse as
    `toolConfig.toolChoice: {"tool": {"name": "get_weather"}}` (`converse_transformation.py:810`)."""
    with peer(every_respond) as provider, gateway.scenario() as scenario:
        served(
            responses(
                gateway,
                bedrock(scenario, provider),
                tools=[WEATHER_TOOL],
                tool_choice={"type": "function", "name": "get_weather"},
            )
        )
        assert object_value(provider.only().body["toolConfig"])["toolChoice"] == {"tool": {"name": "get_weather"}}


def test_bedrock_tool_choice_none_is_refused_with_400_before_bedrock(gateway: Gateway) -> None:
    """Converse has no `none` tool choice. Without `drop_params` the Converse transform raises
    `Bedrock doesn't support tool_choice=none` with status 400 (`converse_transformation.py:796-803`), so the
    caller must get 400 naming it and the peer must receive no POST."""
    with peer(every_respond) as provider, gateway.scenario() as scenario:
        response: Final = responses(gateway, bedrock(scenario, provider), tools=[WEATHER_TOOL], tool_choice="none")
        assert provider.posts() == ()
    assert response.status_code == 400, response.text
    assert "tool_choice=none" in error_message(response)


def test_bedrock_tool_choice_none_is_dropped_under_drop_params(gateway: Gateway) -> None:
    """The same `tool_choice: "none"` on a Bedrock deployment with `drop_params: true` is served, and the
    Converse body carries no `toolChoice` (`converse_transformation.py:797`)."""
    with peer(every_respond) as provider, gateway.scenario() as scenario:
        served(
            responses(gateway, bedrock(scenario, provider, drop_params=True), tools=[WEATHER_TOOL], tool_choice="none")
        )
        tool_config: Final = object_value(provider.only().body.get("toolConfig") or {})
    assert "toolChoice" not in tool_config


def test_bedrock_sampling_params_reach_the_inference_config(gateway: Gateway) -> None:
    """`temperature: 0.4` on a Bedrock deployment must reach Converse as `inferenceConfig.temperature: 0.4`
    (`converse_transformation.py:1148-1168`). UNVERIFIED: whether Claude Sonnet 4.5 on Converse also keeps
    `top_p` beside `temperature`; only `temperature` is asserted."""
    with peer(every_respond) as provider, gateway.scenario() as scenario:
        served(responses(gateway, bedrock(scenario, provider), temperature=0.4))
        assert object_value(provider.only().body["inferenceConfig"])["temperature"] == 0.4


def test_bedrock_max_tokens_stop_is_an_incomplete_response(gateway: Gateway) -> None:
    """Converse answers `stopReason: max_tokens` (chat `length`, `converse_transformation.py:3119`). The
    Responses answer must be `status: incomplete` with `incomplete_details.reason: max_output_tokens` and still
    carry the partial text."""

    def respond(call: Call) -> Reply:
        return json_reply(converse_answer(stop_reason="max_tokens")) if call.method == "POST" else every_respond(call)

    with peer(respond) as provider, gateway.scenario() as scenario:
        body: Final = served(responses(gateway, bedrock(scenario, provider)))
    assert (body["status"], object_value(body["incomplete_details"])["reason"]) == ("incomplete", "max_output_tokens")
    assert output_text(body) == REPLY_TEXT


def test_bedrock_two_tool_uses_come_back_as_two_function_call_items(gateway: Gateway) -> None:
    """Converse answers two `toolUse` blocks (`tooluse_a` Oslo, `tooluse_b` Rome) with `stopReason: tool_use`.
    The Responses `output` must hold two `function_call` items with those `call_id`s in order, and the response
    is `completed`."""
    content: Final[list[dict[str, JsonValue]]] = [
        {"toolUse": {"toolUseId": "tooluse_a", "name": "get_weather", "input": {"city": "Oslo"}}},
        {"toolUse": {"toolUseId": "tooluse_b", "name": "get_weather", "input": {"city": "Rome"}}},
    ]

    def respond(call: Call) -> Reply:
        if call.method != "POST":
            return every_respond(call)
        return json_reply(converse_answer(content=content, stop_reason="tool_use"))

    with peer(respond) as provider, gateway.scenario() as scenario:
        body: Final = served(responses(gateway, bedrock(scenario, provider), tools=[WEATHER_TOOL]))
    assert body["status"] == "completed"
    assert [item["call_id"] for item in output_items(body, "function_call")] == ["tooluse_a", "tooluse_b"]


# ---------------------------------------------------------------- Groq


def test_groq_responses_reach_chat_completions_with_the_bare_model_and_bearer_key(gateway: Gateway) -> None:
    """A Groq deployment gets `/v1/responses`. The peer must receive one `POST /v1/chat/completions` with
    `Authorization: Bearer <provider key>` and body `model: llama-3.3-70b-versatile` (the `groq/` prefix is
    stripped), and the caller gets a completed response with the canned text.
    Skipped as a product bug until fixed, see `_GROQ_SERVICE_TIER_BUG`."""
    pytest.skip(_GROQ_SERVICE_TIER_BUG)
    with peer(every_respond) as provider, gateway.scenario() as scenario:
        body: Final = served(responses(gateway, groq(scenario, provider)))
        call: Final = provider.only()
    assert call.path == CHAT_PATH
    assert {name.lower(): value for name, value in call.headers.items()}["authorization"] == f"Bearer {PROVIDER_KEY}"
    assert call.body["model"] == GROQ_BACKEND
    assert (body["status"], output_text(body)) == ("completed", REPLY_TEXT)


def test_groq_instructions_and_max_output_tokens_become_a_system_message_and_max_tokens(gateway: Gateway) -> None:
    """`instructions: "Be brief."` and `max_output_tokens: 32` on Groq: the chat body must start with
    `{"role": "system", "content": "Be brief."}` followed by the user input, and carry `max_tokens: 32`
    (`transformation.py:1424`; Groq passes `max_tokens` through `OpenAIGPTConfig`).
    Skipped as a product bug until fixed, see `_GROQ_SERVICE_TIER_BUG`."""
    pytest.skip(_GROQ_SERVICE_TIER_BUG)
    prompt: Final = unique_text()
    with peer(every_respond) as provider, gateway.scenario() as scenario:
        served(
            responses(gateway, groq(scenario, provider), input=prompt, instructions="Be brief.", max_output_tokens=32)
        )
        call: Final = provider.only()
    assert [(message["role"], message["content"]) for message in chat_messages(call)] == [
        ("system", "Be brief."),
        ("user", prompt),
    ]
    assert call.body["max_tokens"] == 32


def test_groq_parallel_tool_calls_false_is_forwarded_with_the_tools(gateway: Gateway) -> None:
    """`parallel_tool_calls: false` with the `get_weather` tool on Groq must reach the chat body as
    `parallel_tool_calls: false` beside a `tools` list whose one entry is the nested chat function
    `get_weather`; the bridge only drops it when no tools are sent (`transformation.py:1433`).
    UNVERIFIED: that Groq's supported params keep `parallel_tool_calls` (inherited from `OpenAIGPTConfig`).
    Skipped as a product bug until fixed, see `_GROQ_SERVICE_TIER_BUG`."""
    pytest.skip(_GROQ_SERVICE_TIER_BUG)
    with peer(every_respond) as provider, gateway.scenario() as scenario:
        served(responses(gateway, groq(scenario, provider), tools=[WEATHER_TOOL], parallel_tool_calls=False))
        body: Final = provider.only().body
    assert body["parallel_tool_calls"] is False
    assert [object_value(object_value(tool)["function"])["name"] for tool in list_value(body["tools"])] == [
        "get_weather"
    ]


def test_groq_tool_call_answer_round_trips_as_a_function_call_item(gateway: Gateway) -> None:
    """Groq answers one tool call `call_dec1groq` to `get_weather` with `{"city":"Oslo"}`. The caller must get a
    `function_call` item with that `call_id`, `name` and argument string, and `status: completed`.
    Skipped as a product bug until fixed, see `_GROQ_SERVICE_TIER_BUG`."""
    pytest.skip(_GROQ_SERVICE_TIER_BUG)

    def respond(call: Call) -> Reply:
        if call.method != "POST":
            return every_respond(call)
        return json_reply(
            chat_json(
                call.body,
                content=None,
                finish_reason="tool_calls",
                message={"tool_calls": [tool_call("call_dec1groq", "get_weather", '{"city":"Oslo"}')]},
            )
        )

    with peer(respond) as provider, gateway.scenario() as scenario:
        body: Final = served(responses(gateway, groq(scenario, provider), tools=[WEATHER_TOOL]))
    calls: Final = output_items(body, "function_call")
    assert [(item["call_id"], item["name"], item["arguments"]) for item in calls] == [
        ("call_dec1groq", "get_weather", '{"city":"Oslo"}')
    ]
    assert body["status"] == "completed"


@pytest.mark.parametrize("carrier", ["gemini", "bedrock", "groq"])
def test_every_carrier_reports_usage_under_responses_names(gateway: Gateway, carrier: str) -> None:
    """Each carrier answers with 11 input and 7 output tokens in its own usage shape (`usageMetadata`,
    Converse `usage`, chat `usage`). The Responses answer must report `usage.input_tokens: 11`,
    `output_tokens: 7`, `total_tokens: 18` with no chat `prompt_tokens` key, whichever provider served it. The Groq
    cell is skipped as a product bug, see `_GROQ_SERVICE_TIER_BUG`."""
    if carrier == "groq":
        pytest.skip(_GROQ_SERVICE_TIER_BUG)
    with peer(every_respond) as provider, gateway.scenario() as scenario:
        deploy: Final = {"gemini": gemini, "bedrock": bedrock, "groq": groq}[carrier]
        body: Final = served(responses(gateway, deploy(scenario, provider)))
    usage: Final = object_value(body["usage"])
    assert (usage["input_tokens"], usage["output_tokens"], usage["total_tokens"]) == (11, 7, 18)
    assert "prompt_tokens" not in usage
