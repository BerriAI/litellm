"""`POST /v1/chat/completions` served by Claude on Vertex AI (`vertex_ai/claude-*`).

`VertexAIPartnerModels.completion` (`litellm/llms/vertex_ai/vertex_ai_partner_models/main.py`) mints
an access token, builds `{api_base}/v1/projects/{project}/locations/{location}/publishers/anthropic/
models/{model}:rawPredict` (`:streamRawPredict` when streaming, with `alt=sse` added because the
deployment has an `api_base`), sets `Authorization: Bearer <token>` on top of any deployment headers,
adds `anthropic_version: "vertex-2023-10-16"` and hands the OpenAI request to the Anthropic chat
transformation, which drops `model` from the body (`VertexAIAnthropicConfig.transform_request`). The
Anthropic answer is mapped back to an OpenAI chat completion with Anthropic's usage arithmetic
(`prompt_tokens = input + cache_creation + cache_read`).

The `/v1/messages` side of Claude on Vertex is pinned in `messages_endpoint/test_dec1_msgprov_vertex.py`;
this file covers the chat route, which a Rust port of the chat path or of Vertex auth must keep.
"""

from __future__ import annotations

import json
from typing import Final

import pytest
from integration._support.client import Gateway, eventually, list_value, object_value
from integration._support.database import read_rows
from integration._support.wire import Reply
from integration.providers.dec1_vertex_support import (
    CLAUDE,
    anthropic_error,
    choice,
    claude_echo,
    claude_json,
    claude_path,
    json_of,
    marker,
    message_of,
    path_of,
    query_of,
    streamed_text,
    vertex,
)
from pydantic import JsonValue

_RAW: Final = claude_path() + ":rawPredict"
_STREAM_RAW: Final = claude_path() + ":streamRawPredict"
_TOOL: Final[dict[str, JsonValue]] = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Look up the weather for a city",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
    },
}


def _claude_reply(**fields: object) -> Reply:
    return Reply(body=json.dumps(claude_json(**fields)).encode())  # pyright: ignore[reportArgumentType]  # test-local keyword passthrough


def test_claude_chat_posts_to_raw_predict_with_the_minted_token(gateway: Gateway) -> None:
    """A Claude chat request reaches the peer as one `POST /v1/projects/dec1-vertex-project/locations/
    us-east5/publishers/anthropic/models/claude-sonnet-4-5@20250929:rawPredict` with no query, a minted
    `Authorization: Bearer` token and no `x-api-key`."""
    with vertex(gateway, claude_echo, backend=CLAUDE) as rig:
        response: Final = rig.chat(max_tokens=32)
        assert response.status_code == 200, response.text
        call: Final = rig.peer.one_call()
        assert (path_of(call), query_of(call)) == (_RAW, "")
        assert call.headers["authorization"].removeprefix("Bearer ") in rig.peer.plan.issued
        assert "x-api-key" not in call.headers


def test_claude_chat_streaming_posts_to_stream_raw_predict_with_alt_sse(gateway: Gateway) -> None:
    """`stream: true` switches the method to `:streamRawPredict` and, because the deployment has an
    `api_base`, `_check_custom_proxy` appends the query `alt=sse`."""
    with vertex(gateway, claude_echo, backend=CLAUDE) as rig:
        response, _text = rig.stream(
            "/v1/chat/completions", {"max_tokens": 32, "messages": [{"role": "user", "content": f"s {marker()}"}]}
        )
        assert response.status_code == 200
        call: Final = rig.peer.one_call()
        assert (path_of(call), query_of(call)) == (_STREAM_RAW, "alt=sse")


def test_claude_chat_global_location_is_in_the_path(gateway: Gateway) -> None:
    """`vertex_location: global` builds `locations/global` in the rawPredict path."""
    with vertex(gateway, claude_echo, backend=CLAUDE, vertex_location="global") as rig:
        assert rig.chat(max_tokens=32).status_code == 200
        assert path_of(rig.peer.one_call()) == claude_path(location="global") + ":rawPredict"


def test_claude_chat_body_has_the_vertex_anthropic_version_and_no_model(gateway: Gateway) -> None:
    """The body carries `anthropic_version: "vertex-2023-10-16"`, the caller's `max_tokens`, and no
    `model` (Vertex takes the model from the URL and rejects it in the body)."""
    with vertex(gateway, claude_echo, backend=CLAUDE) as rig:
        assert rig.chat(max_tokens=40).status_code == 200
        body: Final = rig.peer.one_body()
        assert body["anthropic_version"] == "vertex-2023-10-16"
        assert body["max_tokens"] == 40
        assert "model" not in body


def test_claude_chat_system_message_moves_to_the_top_level_system(gateway: Gateway) -> None:
    """An OpenAI system message is lifted into Anthropic's top-level `system` and removed from
    `messages`, which then holds only the user turn."""
    with vertex(gateway, claude_echo, backend=CLAUDE) as rig:
        prompt: Final = f"q {marker()}"
        response: Final = rig.post(
            "/v1/chat/completions",
            {
                "max_tokens": 32,
                "messages": [{"role": "system", "content": "be terse"}, {"role": "user", "content": prompt}],
            },
        )
        assert response.status_code == 200, response.text
        body: Final = rig.peer.one_body()
        assert "be terse" in json.dumps(body["system"])
        assert [object_value(item)["role"] for item in list_value(body["messages"])] == ["user"]


def test_claude_chat_tools_are_sent_with_input_schema(gateway: Gateway) -> None:
    """An OpenAI function tool becomes an Anthropic tool `{"name": "get_weather", "description": ...,
    "input_schema": <the parameters>}`."""
    with vertex(gateway, claude_echo, backend=CLAUDE) as rig:
        assert rig.chat(max_tokens=32, tools=[_TOOL]).status_code == 200
        tool: Final = object_value(list_value(rig.peer.one_body()["tools"])[0])
        assert tool["name"] == "get_weather"
        assert tool["description"] == "Look up the weather for a city"
        assert object_value(tool["input_schema"])["properties"] == {"city": {"type": "string"}}


@pytest.mark.parametrize(
    ("tool_choice", "expected"),
    (
        ("auto", {"type": "auto"}),
        ("required", {"type": "any"}),
        ({"type": "function", "function": {"name": "get_weather"}}, {"type": "tool", "name": "get_weather"}),
    ),
    ids=("auto", "required", "named"),
)
def test_claude_chat_tool_choice_maps_to_anthropic_shapes(
    gateway: Gateway, tool_choice: JsonValue, expected: dict[str, JsonValue]
) -> None:
    """`tool_choice` auto, required and a named function become Anthropic's `{"type": "auto"}`,
    `{"type": "any"}` and `{"type": "tool", "name": ...}`. UNVERIFIED: Anthropic may add
    `disable_parallel_tool_use` keys; the test checks the listed keys only."""
    with vertex(gateway, claude_echo, backend=CLAUDE) as rig:
        assert rig.chat(max_tokens=32, tools=[_TOOL], tool_choice=tool_choice).status_code == 200
        sent: Final = object_value(rig.peer.one_body()["tool_choice"])
        assert {key: sent.get(key) for key in expected} == expected


def test_claude_chat_overwrites_a_deployment_authorization_header(gateway: Gateway) -> None:
    """A deployment `extra_headers` `Authorization` is replaced by the minted bearer token (main.py builds
    the headers as `{**headers, "Authorization": f"Bearer {access_token}"}`), so a stale header can
    never reach Google on the chat route."""
    with vertex(
        gateway, claude_echo, backend=CLAUDE, extra_headers={"Authorization": "Bearer stale-deployment-token"}
    ) as rig:
        assert rig.chat(max_tokens=32).status_code == 200
        assert rig.peer.one_call().headers["authorization"].removeprefix("Bearer ") in rig.peer.plan.issued


def test_claude_chat_text_answer_comes_back_as_an_openai_message(gateway: Gateway) -> None:
    """An Anthropic `end_turn` message with one text block is answered as an assistant message holding
    that text with `finish_reason: "stop"`."""
    with vertex(gateway, lambda _r: _claude_reply(text="bonjour claude"), backend=CLAUDE) as rig:
        response: Final = rig.chat(max_tokens=32)
        assert response.status_code == 200, response.text
        assert message_of(response)["content"] == "bonjour claude"
        assert choice(response)["finish_reason"] == "stop"


def test_claude_chat_max_tokens_stop_is_reported_as_length(gateway: Gateway) -> None:
    """`stop_reason: "max_tokens"` maps to the OpenAI `finish_reason: "length"`."""
    with vertex(gateway, lambda _r: _claude_reply(stop_reason="max_tokens"), backend=CLAUDE) as rig:
        assert choice(rig.chat(max_tokens=32))["finish_reason"] == "length"


def test_claude_chat_tool_use_comes_back_as_tool_calls(gateway: Gateway) -> None:
    """A `tool_use` block becomes a `tool_calls` entry with the block's `id`, the tool name and the JSON
    text of `input`, and `stop_reason: "tool_use"` maps to `finish_reason: "tool_calls"`."""
    content: Final[list[JsonValue]] = [
        {"type": "tool_use", "id": "toolu_vrtx_01", "name": "get_weather", "input": {"city": "Rome"}}
    ]
    with vertex(gateway, lambda _r: _claude_reply(content=content, stop_reason="tool_use"), backend=CLAUDE) as rig:
        response: Final = rig.chat(max_tokens=32, tools=[_TOOL])
        call: Final = object_value(list_value(message_of(response)["tool_calls"])[0])
        assert call["id"] == "toolu_vrtx_01"
        assert object_value(call["function"])["name"] == "get_weather"
        assert json.loads(str(object_value(call["function"])["arguments"])) == {"city": "Rome"}
        assert choice(response)["finish_reason"] == "tool_calls"


def test_claude_chat_usage_adds_cache_tokens_to_the_prompt_count(gateway: Gateway) -> None:
    """Usage `{input_tokens: 10, cache_creation_input_tokens: 5, cache_read_input_tokens: 30,
    output_tokens: 7}` is reported as `prompt_tokens = 10 + 5 + 30 = 45`, `completion_tokens = 7`,
    `prompt_tokens_details.cached_tokens = 30` and `cache_creation_input_tokens = 5`
    (`AnthropicConfig.calculate_usage`)."""
    usage_in: Final[dict[str, JsonValue]] = {
        "input_tokens": 10,
        "cache_creation_input_tokens": 5,
        "cache_read_input_tokens": 30,
        "output_tokens": 7,
    }
    with vertex(gateway, lambda _r: _claude_reply(usage=usage_in), backend=CLAUDE) as rig:
        usage: Final = object_value(json_of(rig.chat(max_tokens=32))["usage"])
        assert (usage["prompt_tokens"], usage["completion_tokens"], usage["total_tokens"]) == (45, 7, 52)
        assert object_value(usage["prompt_tokens_details"])["cached_tokens"] == 30
        assert usage["cache_creation_input_tokens"] == 5


def test_claude_chat_stream_is_relayed_as_chat_chunks(gateway: Gateway) -> None:
    """Vertex's Anthropic SSE events reach the caller as `chat.completion.chunk` frames whose deltas
    join to "hello stream", ending with `data: [DONE]`."""
    with vertex(gateway, claude_echo, backend=CLAUDE) as rig:
        response, text = rig.stream(
            "/v1/chat/completions", {"max_tokens": 32, "messages": [{"role": "user", "content": f"s {marker()}"}]}
        )
        assert response.status_code == 200
        assert streamed_text(text) == "hello stream"
        assert text.rstrip().endswith("data: [DONE]")


@pytest.mark.parametrize(
    ("status", "error_type", "expected"),
    ((429, "rate_limit_error", 429), (400, "invalid_request_error", 400), (404, "not_found_error", 404)),
)
def test_claude_chat_errors_keep_their_status(gateway: Gateway, status: int, error_type: str, expected: int) -> None:
    """Vertex Claude errors arrive in Anthropic's envelope and leave the chat route with the same status
    (429, 400, 404) and the provider message in the body (`_map_vertex_exception` status branches)."""
    message: Final = f"synthetic claude failure {status}"
    with vertex(gateway, lambda _r: anthropic_error(status, error_type, message), backend=CLAUDE) as rig:
        response: Final = rig.chat(max_tokens=32)
        assert response.status_code == expected, response.text
        assert message in response.text


def test_claude_chat_spend_row_names_vertex_ai(gateway: Gateway) -> None:
    """One Claude chat call writes one SpendLogs row with `custom_llm_provider = "vertex_ai"` and 11
    prompt plus 7 completion tokens, priced at the deployment's custom prices
    (`11 * 0.001 + 7 * 0.002 = 0.025`)."""
    with vertex(gateway, claude_echo, backend=CLAUDE, input_cost_per_token=0.001, output_cost_per_token=0.002) as rig:
        assert rig.chat(max_tokens=32).status_code == 200
        rows: Final = eventually(
            lambda: read_rows(
                'SELECT custom_llm_provider, prompt_tokens, completion_tokens, spend FROM "LiteLLM_SpendLogs" WHERE model_group=%s',
                (rig.model,),
            ),
            lambda found: len(found) >= 1,
            seconds=90,
        )
        assert len(rows) == 1, rows
        assert (rows[0]["custom_llm_provider"], rows[0]["prompt_tokens"], rows[0]["completion_tokens"]) == (
            "vertex_ai",
            11,
            7,
        )
        assert float(str(rows[0]["spend"])) == pytest.approx(0.025)
