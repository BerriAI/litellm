"""`POST /v1/responses` carried by Vertex Gemini and Vertex Claude through the chat completions bridge.

Vertex has no native Responses config (`ProviderConfigManager.get_provider_responses_api_config` returns None
for `vertex_ai`), so the proxy turns a Responses request into a chat completion
(`litellm/responses/litellm_completion_transformation/`), sends it through the Vertex chat handlers (Gemini
`generateContent` or Claude `rawPredict`), and turns the chat answer back into Responses `output` items.
`test_dec1_vertex_gemini_stream_responses.py` pins the Gemini basics (string input, instructions,
max_output_tokens, reasoning effort, usage names, a text stream). This file adds the gaps the round 1 to 2
indexes listed: function tools and `function_call` / `function_call_output` items on Gemini, `text.format`
json_schema, `tool_choice`, and Vertex Claude as a Responses carrier (rawPredict URL, body, tools, tool results,
usage, streaming). A Rust port of the Responses front door or of the bridge has to keep each mapping.
"""

from __future__ import annotations

import json
from typing import Final

import pytest
from integration._support.client import Gateway, eventually, list_value, object_value, string_value
from integration._support.database import read_rows
from integration._support.wire import Reply
from integration.providers.dec1_vertex3_support import event_types, is_token_from, output_of_type
from integration.providers.dec1_vertex_support import (
    CLAUDE,
    anthropic_error,
    at,
    body_of,
    claude_echo,
    claude_json,
    claude_path,
    gemini_echo,
    gemini_reply,
    marker,
    path_of,
    query_of,
    vertex,
)
from pydantic import JsonValue

_FLAT_TOOL: Final[dict[str, JsonValue]] = {
    "type": "function",
    "name": "get_weather",
    "description": "Look up the weather for a city",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
}
_ROUND_TRIP: Final[list[JsonValue]] = [
    {"type": "function_call", "call_id": "call_dec1_1", "name": "get_weather", "arguments": '{"city": "Paris"}'},
    {"type": "function_call_output", "call_id": "call_dec1_1", "output": "sunny and 21C"},
]
_RAW: Final = claude_path() + ":rawPredict"


def _user(text: str) -> dict[str, JsonValue]:
    return {"role": "user", "content": text}


# ---------------------------------------------------------------- Vertex Gemini as the carrier


def test_a_flat_responses_function_tool_becomes_a_gemini_function_declaration(gateway: Gateway) -> None:
    """A Responses function tool (`{"type": "function", "name", "parameters"}`, no nested `function`) reaches
    `generateContent` as `tools[0].function_declarations[0]` with `name: "get_weather"` and the description."""
    with vertex(gateway, gemini_echo) as rig:
        response: Final = rig.post("/v1/responses", {"input": f"weather {marker()}", "tools": [_FLAT_TOOL]})
        assert response.status_code == 200, response.text
        declaration: Final = object_value(at(rig.peer.one_body(), "tools", 0, "function_declarations", 0))
        assert declaration["name"] == "get_weather"
        assert declaration["description"] == "Look up the weather for a city"


def test_a_gemini_function_call_comes_back_as_a_function_call_output_item(gateway: Gateway) -> None:
    """A candidate with a `functionCall {name: "get_weather", args: {"city": "Paris"}}` part is returned as
    one `output` item of `type: "function_call"` with that `name`, `arguments` equal to the JSON text of the
    args, and a non-empty `call_id` the caller can answer with `function_call_output`."""
    reply: Final = gemini_reply(parts=[{"functionCall": {"name": "get_weather", "args": {"city": "Paris"}}}])
    with vertex(gateway, lambda _r: reply) as rig:
        response: Final = rig.post("/v1/responses", {"input": f"weather {marker()}", "tools": [_FLAT_TOOL]})
        assert response.status_code == 200, response.text
        calls: Final = output_of_type(response, "function_call")
        assert len(calls) == 1, calls
        assert calls[0]["name"] == "get_weather"
        assert json.loads(string_value(calls[0]["arguments"])) == {"city": "Paris"}
        assert string_value(calls[0]["call_id"])


def test_function_call_and_output_items_become_function_call_and_function_response_parts(gateway: Gateway) -> None:
    """An input of a user message, a `function_call` item and its `function_call_output` reaches Gemini as
    a user turn, a `model` turn with a `functionCall {name: "get_weather", args: {"city": "Paris"}}` part, and
    a turn with a `functionResponse` part for `get_weather` whose response carries "sunny and 21C"."""
    with vertex(gateway, gemini_echo) as rig:
        response: Final = rig.post(
            "/v1/responses", {"input": [_user(f"weather {marker()}"), *_ROUND_TRIP], "tools": [_FLAT_TOOL]}
        )
        assert response.status_code == 200, response.text
        contents: Final = list_value(rig.peer.one_body()["contents"])
        assert object_value(contents[1])["role"] == "model"
        call_part: Final = object_value(at(contents[1], "parts", 0))
        assert object_value(call_part["functionCall"]) == {"name": "get_weather", "args": {"city": "Paris"}}
        answer_part: Final = object_value(at(contents[2], "parts", 0))
        function_response: Final = object_value(answer_part["functionResponse"])
        assert function_response["name"] == "get_weather"
        assert "sunny and 21C" in json.dumps(function_response["response"])


def test_a_text_format_json_schema_becomes_the_gemini_response_json_schema(gateway: Gateway) -> None:
    """`text.format: {"type": "json_schema", "name": "weather", "schema": {...}}` reaches Gemini 2.5 as
    `generationConfig.response_mime_type: "application/json"` and a `response_json_schema` holding the
    schema's `city` property (the same mapping chat's `response_format` gets)."""
    schema: Final[dict[str, JsonValue]] = {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
        "additionalProperties": False,
    }
    with vertex(gateway, gemini_echo) as rig:
        response: Final = rig.post(
            "/v1/responses",
            {
                "input": f"json please {marker()}",
                "text": {"format": {"type": "json_schema", "name": "weather", "schema": schema, "strict": True}},
            },
        )
        assert response.status_code == 200, response.text
        config: Final = object_value(rig.peer.one_body()["generationConfig"])
        assert config["response_mime_type"] == "application/json"
        assert "city" in json.dumps(config["response_json_schema"])


def test_responses_tool_choice_required_becomes_function_calling_mode_any(gateway: Gateway) -> None:
    """`tool_choice: "required"` on a Responses request is sent as `toolConfig.functionCallingConfig.mode:
    "ANY"`, so Gemini must call a tool."""
    with vertex(gateway, gemini_echo) as rig:
        response: Final = rig.post(
            "/v1/responses", {"input": f"q {marker()}", "tools": [_FLAT_TOOL], "tool_choice": "required"}
        )
        assert response.status_code == 200, response.text
        assert at(rig.peer.one_body(), "toolConfig", "functionCallingConfig", "mode") == "ANY"


def test_responses_sampling_params_reach_the_generation_config(gateway: Gateway) -> None:
    """`temperature: 0.2` and `top_p: 0.5` on a Responses request reach `generationConfig.temperature` and
    `generationConfig.top_p` unchanged."""
    with vertex(gateway, gemini_echo) as rig:
        assert (
            rig.post("/v1/responses", {"input": f"q {marker()}", "temperature": 0.2, "top_p": 0.5}).status_code == 200
        )
        config: Final = object_value(rig.peer.one_body()["generationConfig"])
        assert (config["temperature"], config["top_p"]) == (0.2, 0.5)


def test_a_gemini_thought_part_comes_back_as_a_reasoning_output_item(gateway: Gateway) -> None:
    """A candidate with a `thought: true` text part before the answer comes back with a `reasoning` output
    item before the `message` item. UNVERIFIED: the bridge turns chat `reasoning_content` into a `reasoning`
    item only when it is non-empty; look at `litellm/responses/litellm_completion_transformation/
    transformation.py` if the item is missing."""
    reply: Final = gemini_reply(parts=[{"text": "thinking it over", "thought": True}, {"text": "42"}])
    with vertex(gateway, lambda _r: reply) as rig:
        response: Final = rig.post("/v1/responses", {"input": f"q {marker()}", "reasoning": {"effort": "low"}})
        assert response.status_code == 200, response.text
        kinds: Final = [object_value(item)["type"] for item in list_value(response.json()["output"])]
        assert kinds.index("reasoning") < kinds.index("message"), kinds


def test_a_gemini_responses_call_writes_one_spend_row_under_vertex_ai(gateway: Gateway) -> None:
    """One bridged Responses call on Vertex Gemini writes exactly one SpendLogs row for the deployment with
    `custom_llm_provider = "vertex_ai"` and 11 prompt plus 7 completion tokens (the bridge must not log the
    inner chat call a second time)."""
    with vertex(gateway, gemini_echo) as rig:
        assert rig.post("/v1/responses", {"input": f"q {marker()}"}).status_code == 200
        rows: Final = eventually(
            lambda: read_rows(
                'SELECT custom_llm_provider, prompt_tokens, completion_tokens FROM "LiteLLM_SpendLogs" WHERE model_group=%s',
                (rig.model,),
            ),
            lambda found: len(found) >= 1,
            seconds=90,
        )
        assert [(row["custom_llm_provider"], row["prompt_tokens"], row["completion_tokens"]) for row in rows] == [
            ("vertex_ai", 11, 7)
        ]


# ---------------------------------------------------------------- Vertex Claude as the carrier


def test_a_claude_responses_call_posts_raw_predict_with_the_minted_token(gateway: Gateway) -> None:
    """`/v1/responses` on `vertex_ai/claude-sonnet-4-5@20250929` sends one `POST .../publishers/anthropic/
    models/claude-sonnet-4-5@20250929:rawPredict` with no query, a minted bearer, `anthropic_version:
    "vertex-2023-10-16"` and no `model` in the body."""
    with vertex(gateway, claude_echo, backend=CLAUDE) as rig:
        response: Final = rig.post("/v1/responses", {"input": f"q {marker()}", "max_output_tokens": 64})
        assert response.status_code == 200, response.text
        call: Final = rig.peer.one_call()
        assert (path_of(call), query_of(call)) == (_RAW, "")
        assert is_token_from(rig.peer.plan, call)
        body: Final = body_of(call)
        assert body["anthropic_version"] == "vertex-2023-10-16"
        assert "model" not in body


def test_claude_responses_instructions_and_max_output_tokens_reach_anthropic_fields(gateway: Gateway) -> None:
    """`instructions` is lifted into Anthropic's top level `system` (not a message) and
    `max_output_tokens: 64` is sent as `max_tokens: 64`."""
    with vertex(gateway, claude_echo, backend=CLAUDE) as rig:
        response: Final = rig.post(
            "/v1/responses", {"input": f"q {marker()}", "instructions": "answer in French", "max_output_tokens": 64}
        )
        assert response.status_code == 200, response.text
        body: Final = rig.peer.one_body()
        assert "answer in French" in json.dumps(body["system"])
        assert body["max_tokens"] == 64
        assert [object_value(item)["role"] for item in list_value(body["messages"])] == ["user"]


def test_claude_responses_function_tools_become_anthropic_tools(gateway: Gateway) -> None:
    """A flat Responses function tool reaches Claude as `{"name": "get_weather", "description": ...,
    "input_schema": <parameters>}`."""
    with vertex(gateway, claude_echo, backend=CLAUDE) as rig:
        response: Final = rig.post(
            "/v1/responses", {"input": f"q {marker()}", "tools": [_FLAT_TOOL], "max_output_tokens": 64}
        )
        assert response.status_code == 200, response.text
        tool: Final = object_value(list_value(rig.peer.one_body()["tools"])[0])
        assert tool["name"] == "get_weather"
        assert object_value(tool["input_schema"])["properties"] == {"city": {"type": "string"}}


def test_a_claude_tool_use_comes_back_as_a_function_call_item_with_its_id(gateway: Gateway) -> None:
    """A `tool_use` block `{id: "toolu_dec1", name: "get_weather", input: {"city": "Paris"}}` is returned
    as a `function_call` output item with `call_id: "toolu_dec1"`, the same name and the JSON text of the
    input as `arguments`."""
    content: Final[list[JsonValue]] = [
        {"type": "tool_use", "id": "toolu_dec1", "name": "get_weather", "input": {"city": "Paris"}}
    ]
    reply: Final = Reply(body=json.dumps(claude_json(content=content, stop_reason="tool_use")).encode())
    with vertex(gateway, lambda _r: reply, backend=CLAUDE) as rig:
        response: Final = rig.post(
            "/v1/responses", {"input": f"q {marker()}", "tools": [_FLAT_TOOL], "max_output_tokens": 64}
        )
        assert response.status_code == 200, response.text
        calls: Final = output_of_type(response, "function_call")
        assert [(call["call_id"], call["name"]) for call in calls] == [("toolu_dec1", "get_weather")]
        assert json.loads(string_value(calls[0]["arguments"])) == {"city": "Paris"}


def test_a_function_call_output_reaches_claude_as_a_tool_result_block(gateway: Gateway) -> None:
    """The `function_call` / `function_call_output` input pair is sent to Claude as an assistant
    `tool_use` block with `id: "call_dec1_1"` followed by a user `tool_result` block with
    `tool_use_id: "call_dec1_1"` holding "sunny and 21C"."""
    with vertex(gateway, claude_echo, backend=CLAUDE) as rig:
        response: Final = rig.post(
            "/v1/responses",
            {"input": [_user(f"weather {marker()}"), *_ROUND_TRIP], "tools": [_FLAT_TOOL], "max_output_tokens": 64},
        )
        assert response.status_code == 200, response.text
        messages: Final = [object_value(item) for item in list_value(rig.peer.one_body()["messages"])]
        assert [message["role"] for message in messages] == ["user", "assistant", "user"]
        tool_use: Final = next(
            object_value(block)
            for block in list_value(messages[1]["content"])
            if object_value(block)["type"] == "tool_use"
        )
        assert (tool_use["id"], tool_use["name"]) == ("call_dec1_1", "get_weather")
        tool_result: Final = next(
            object_value(block)
            for block in list_value(messages[2]["content"])
            if object_value(block)["type"] == "tool_result"
        )
        assert tool_result["tool_use_id"] == "call_dec1_1"
        assert "sunny and 21C" in json.dumps(tool_result["content"])


def test_claude_responses_usage_uses_input_and_output_token_names(gateway: Gateway) -> None:
    """Claude's `input_tokens: 11` and `output_tokens: 7` come back as Responses `usage {input_tokens: 11,
    output_tokens: 7, total_tokens: 18}` on a `status: "completed"` response."""
    with vertex(gateway, claude_echo, backend=CLAUDE) as rig:
        response: Final = rig.post("/v1/responses", {"input": f"q {marker()}", "max_output_tokens": 64})
        assert response.status_code == 200, response.text
        body: Final = object_value(response.json())
        assert body["status"] == "completed"
        usage: Final = object_value(body["usage"])
        assert (usage["input_tokens"], usage["output_tokens"], usage["total_tokens"]) == (11, 7, 18)


def test_a_streamed_claude_response_uses_stream_raw_predict_and_ends_completed(gateway: Gateway) -> None:
    """`stream: true` posts to `...:streamRawPredict` with `alt=sse`, and the caller's Responses event
    stream carries `response.created`, at least one `response.output_text.delta` and ends with
    `response.completed`."""
    with vertex(gateway, claude_echo, backend=CLAUDE) as rig:
        response, text = rig.stream("/v1/responses", {"input": f"q {marker()}", "max_output_tokens": 64})
        assert response.status_code == 200
        call: Final = rig.peer.one_call()
        assert (path_of(call), query_of(call)) == (claude_path() + ":streamRawPredict", "alt=sse")
        kinds: Final = event_types(text)
        assert kinds[0] == "response.created", kinds
        assert "response.output_text.delta" in kinds
        assert kinds[-1] == "response.completed", kinds


@pytest.mark.parametrize(("status", "error_type"), [(429, "rate_limit_error"), (400, "invalid_request_error")])
def test_a_claude_error_on_the_responses_route_keeps_its_status(gateway: Gateway, status: int, error_type: str) -> None:
    """A Vertex Claude error envelope (`{"type": "error", "error": {"type": ..., "message": ...}}`) leaves
    `/v1/responses` with the same status and the provider message in the body. UNVERIFIED: that the bridge
    re-raises the chat exception without rewrapping it as a 500."""
    message: Final = f"synthetic responses failure {status}"
    with vertex(gateway, lambda _r: anthropic_error(status, error_type, message), backend=CLAUDE) as rig:
        response: Final = rig.post("/v1/responses", {"input": f"q {marker()}", "max_output_tokens": 64})
        assert response.status_code == status, response.text
        assert message in response.text
