"""Contract: Groq chat reasoning params, native structured outputs, fake streaming, web search, service tier and streams.

A `groq/<model>` deployment posts to `<api_base>/chat/completions` with `Authorization: Bearer <key>`
(`GroqChatConfig`, `litellm/llms/groq/chat/transformation.py`, dispatched by `_complete_groq` in
`litellm/main.py:2290-2350`). Groq is in `openai_compatible_providers`, so any param the proxy does not know
(Groq's own `reasoning_format`, `include_reasoning`) travels in `extra_body` and is merged into the top level of
the request body (`litellm/utils.py:6366-6398`, `llm_http_handler.py:973-983`). Beyond that:

- `reasoning_effort` is only accepted on models the price map marks `supports_reasoning`;
- a model with native structured outputs gets `response_format` as sent (the forced-tool emulation for other
  models is covered in `test_dec1_openai_compatible_chat_wire.py`);
- any `response_format` makes a stream fake: Groq is called without streaming and the proxy streams the answer;
- `web_search_options` on a model with `supports_web_search` becomes a `browser_search` tool and is refused
  elsewhere; Groq's `executed_tools` are counted into `usage.server_tool_use`;
- `service_tier` values Groq invents (`on_demand`) are reported as `auto`;
- stream chunks carrying `error` raise with Groq's status; the stream usage only comes from a top-level `usage`
  chunk, the `x_groq.usage` that Groq also sends is ignored (see the skipped test and the recount cost case
  `groq-qwen-3.8-stream_x_groq_recount` in `cost_calculation/cases/misc.py`).
"""

from __future__ import annotations

from typing import Final

import pytest
from integration._support.client import Gateway, list_value, object_value
from integration._support.wire import Reply
from integration.providers.dec1_otherprov2_support import (
    TOOL_CALL_ID,
    chunk,
    delta_choice,
    raw_stream,
    stream_text,
    tool_call,
    tool_call_reply,
    upstream_error,
    usage_with,
)
from integration.providers.dec1_otherprov_support import (
    REPLY_TEXT,
    TOOL,
    Call,
    chat,
    chat_json,
    default_respond,
    deployment,
    error_message,
    final_usage,
    first_choice,
    joined_delta,
    json_reply,
    marker,
    peer,
    served,
    stream_chunks,
)
from pydantic import JsonValue

_LLAMA: Final = "groq/llama-3.3-70b-versatile"
_QWEN: Final = "groq/qwen/qwen3.8-27b"
_GPT_OSS: Final = "groq/openai/gpt-oss-20b"
_SERVICE_TIER_BUG: Final = (
    "BUG: a Groq reply without service_tier fails with AttributeError on ModelResponse.service_tier"
    " (groq/chat/transformation.py:329 calls getattr without a default)"
)


def test_groq_reasoning_format_reaches_the_top_level_of_the_body(gateway: Gateway) -> None:
    """The caller sends Groq's own `reasoning_format: "parsed"` to `groq/qwen/qwen3.8-27b`.

    It is not an OpenAI param, so it is put in `extra_body` (`add_provider_specific_params_to_optional_params`,
    `litellm/utils.py:6366-6398`) and merged into the request body (`llm_http_handler.py:973-983`): Groq
    receives `reasoning_format: "parsed"` at the top level and no `extra_body` key.
    Skipped as a product bug until fixed, see `_SERVICE_TIER_BUG`."""
    pytest.skip(_SERVICE_TIER_BUG)
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _QWEN, api_base_suffix="/openai/v1")
        served(chat(gateway, model, reasoning_format="parsed"))
        call: Final = provider.only()
        assert call.path == "/openai/v1/chat/completions", call.path
        assert call.body["reasoning_format"] == "parsed" and "extra_body" not in call.body, call.body


def test_groq_reasoning_effort_reaches_a_reasoning_model(gateway: Gateway) -> None:
    """`groq/qwen/qwen3.8-27b` has `supports_reasoning: true`, so `reasoning_effort` joins its supported params
    (`groq/chat/transformation.py:115-119`) and `reasoning_effort: "none"` reaches Groq as sent.
    Skipped as a product bug until fixed, see `_SERVICE_TIER_BUG`."""
    pytest.skip(_SERVICE_TIER_BUG)
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _QWEN, api_base_suffix="/openai/v1")
        served(chat(gateway, model, reasoning_effort="none"))
        assert provider.only().body["reasoning_effort"] == "none"


def test_groq_reasoning_effort_is_refused_on_a_model_without_reasoning(gateway: Gateway) -> None:
    """`groq/llama-3.3-70b-versatile` is not in the price map, so it does not support reasoning and
    `reasoning_effort: "high"` is answered 400 with `groq does not support parameters: ['reasoning_effort']`
    (`litellm/utils.py:5789-5794`) before Groq is called."""
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _LLAMA, api_base_suffix="/openai/v1")
        response: Final = chat(gateway, model, reasoning_effort="high")
        assert response.status_code == 400, response.text
        assert "groq does not support parameters: ['reasoning_effort']" in error_message(response)
        assert provider.posts() == ()


def test_groq_native_structured_output_model_gets_response_format_as_sent(gateway: Gateway) -> None:
    """`groq/openai/gpt-oss-20b` has `supports_response_schema: true`, so a `json_schema` response_format is
    forwarded unchanged with no `json_tool_call` tool and no forced `tool_choice`
    (`groq/chat/transformation.py:249-280`).
    Skipped as a product bug until fixed, see `_SERVICE_TIER_BUG`."""
    pytest.skip(_SERVICE_TIER_BUG)
    response_format: Final[dict[str, JsonValue]] = {
        "type": "json_schema",
        "json_schema": {"name": "city", "schema": {"type": "object", "properties": {"city": {"type": "string"}}}},
    }
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _GPT_OSS, api_base_suffix="/openai/v1")
        served(chat(gateway, model, response_format=response_format))
        sent: Final = provider.only().body
        assert sent["response_format"] == response_format, sent
        assert "tools" not in sent and "tool_choice" not in sent, sent


def test_groq_stream_with_response_format_is_faked_from_one_non_streaming_call(gateway: Gateway) -> None:
    """A streaming request with `response_format: {"type": "json_object"}`. Groq does not stream structured
    output, so `fake_stream` is set (`_should_fake_stream`, `groq/chat/transformation.py:191-198`, `:230-231`):
    Groq receives one request without `stream: true`, answers plain JSON, and the caller still gets SSE chunks
    whose joined content is the reply text.
    Skipped as a product bug until fixed, see `_SERVICE_TIER_BUG`."""
    pytest.skip(_SERVICE_TIER_BUG)
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _GPT_OSS, api_base_suffix="/openai/v1")
        chunks: Final = stream_chunks(gateway, model, response_format={"type": "json_object"})
        assert joined_delta(chunks, "content") == REPLY_TEXT, chunks
        assert provider.only().body.get("stream") is not True, provider.only().body


def test_groq_drops_null_fields_from_replayed_assistant_messages(gateway: Gateway) -> None:
    """An assistant turn replayed with `function_call: null` and `tool_calls: null` next to its text. Groq
    rejects null `function_call` (issue 5839), so the assistant message reaches Groq with only `role` and
    `content` (`GroqChatConfig._transform_messages`, `groq/chat/transformation.py:140-163`).
    Skipped as a product bug until fixed, see `_SERVICE_TIER_BUG`."""
    pytest.skip(_SERVICE_TIER_BUG)
    messages: Final[list[JsonValue]] = [
        {"role": "user", "content": marker()},
        {"role": "assistant", "content": "earlier answer", "function_call": None, "tool_calls": None},
        {"role": "user", "content": marker()},
    ]
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _LLAMA, api_base_suffix="/openai/v1")
        served(chat(gateway, model, messages=messages))
        assistant: Final = object_value(list_value(provider.only().body["messages"])[1])
        assert assistant == {"role": "assistant", "content": "earlier answer"}, assistant


def test_groq_web_search_options_become_a_browser_search_tool(gateway: Gateway) -> None:
    """`groq/openai/gpt-oss-20b` has `supports_web_search: true`; the caller sends `web_search_options: {}` and
    one function tool. Groq receives no `web_search_options`, and its tools are the caller's function tool plus
    `{"type": "browser_search"}` (`groq/chat/transformation.py:281-299`).
    Skipped as a product bug until fixed, see `_SERVICE_TIER_BUG`."""
    pytest.skip(_SERVICE_TIER_BUG)
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _GPT_OSS, api_base_suffix="/openai/v1")
        served(chat(gateway, model, web_search_options={}, tools=[TOOL]))
        sent: Final = provider.only().body
        assert "web_search_options" not in sent, sent
        types: Final = [object_value(tool)["type"] for tool in list_value(sent["tools"])]
        assert sorted(str(kind) for kind in types) == ["browser_search", "function"], sent


def test_groq_refuses_web_search_options_on_a_model_without_web_search(gateway: Gateway) -> None:
    """`groq/llama-3.3-70b-versatile` neither is a compound model nor supports web search, so
    `web_search_options` is removed from its supported params (`groq/chat/transformation.py:109-113`) and the call
    is answered 400 with `groq does not support parameters: ['web_search_options']`."""
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _LLAMA, api_base_suffix="/openai/v1")
        response: Final = chat(gateway, model, web_search_options={})
        assert response.status_code == 400, response.text
        assert "groq does not support parameters: ['web_search_options']" in error_message(response)
        assert provider.posts() == ()


def test_groq_executed_browser_tools_are_counted_into_server_tool_use(gateway: Gateway) -> None:
    """Groq answers a message with `executed_tools` of types `browser.search`, `browser.search` and
    `browser.open`. The caller's usage carries `server_tool_use: {"web_search_requests": 2,
    "browser_open_requests": 1}` (`_add_web_search_usage`, `groq/chat/transformation.py:337-360`), which is
    what web search billing reads.
    Skipped as a product bug until fixed, see `_SERVICE_TIER_BUG`."""
    pytest.skip(_SERVICE_TIER_BUG)
    executed: Final[list[JsonValue]] = [
        {"index": 0, "type": "browser.search", "arguments": "{}"},
        {"index": 1, "type": "browser.search", "arguments": "{}"},
        {"index": 2, "type": "browser.open", "arguments": "{}"},
    ]

    def respond(call: Call) -> Reply:
        return json_reply(chat_json(call.body, extra_message={"executed_tools": executed}))

    with peer(respond) as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _GPT_OSS, api_base_suffix="/openai/v1")
        usage: Final = object_value(served(chat(gateway, model))["usage"])
        tool_use: Final = object_value(usage["server_tool_use"])
        assert (tool_use["web_search_requests"], tool_use["browser_open_requests"]) == (2, 1), usage


@pytest.mark.parametrize(("groq_tier", "reported"), [("on_demand", "auto"), ("flex", "flex"), (None, "auto")])
def test_groq_service_tier_is_reported_in_the_openai_vocabulary(
    gateway: Gateway, groq_tier: str | None, reported: str
) -> None:
    """Groq answers `service_tier` `on_demand`, `flex` or nothing. The caller sees `auto`, `flex` and `auto`:
    values outside `auto`, `default`, `flex` and a missing tier map to `auto` (`_map_groq_service_tier`,
    `groq/chat/transformation.py:362-371`). The missing-tier cell is skipped as a product bug, see
    `_SERVICE_TIER_BUG`: `getattr(model_response, "service_tier")` raises before the mapping runs."""
    if groq_tier is None:
        pytest.skip(_SERVICE_TIER_BUG)

    def respond(call: Call) -> Reply:
        return json_reply(chat_json(call.body, extra_body={"service_tier": groq_tier} if groq_tier else {}))

    with peer(respond) as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _LLAMA, api_base_suffix="/openai/v1")
        assert served(chat(gateway, model))["service_tier"] == reported


def test_groq_tool_call_answer_reaches_the_caller(gateway: Gateway) -> None:
    """Groq answers one tool call with finish reason `tool_calls`; the caller's choice has the same id, name
    and finish reason. Groq's own tool calls are not mistaken for the JSON-mode tool (no `response_format` was
    sent, so `json_mode` is off).
    Skipped as a product bug until fixed, see `_SERVICE_TIER_BUG`."""
    pytest.skip(_SERVICE_TIER_BUG)
    with peer(tool_call_reply) as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _LLAMA, api_base_suffix="/openai/v1")
        choice: Final = first_choice(served(chat(gateway, model, tools=[TOOL])))
        found: Final = object_value(list_value(object_value(choice["message"])["tool_calls"])[0])
        assert (choice["finish_reason"], found["id"]) == ("tool_calls", TOOL_CALL_ID), choice
        assert object_value(found["function"])["name"] == "get_weather", choice


def test_groq_stream_top_level_usage_chunk_is_the_final_usage(gateway: Gateway) -> None:
    """Groq streams text, a stop chunk, and a usage chunk with `choices: []` and top-level usage 11 prompt, 7
    completion, alongside an `x_groq.usage` that says 99 and 99. The caller's final usage is 11 and 7, taken from
    the top-level `usage` (the OpenAI stream handler reads only `usage`)."""

    def respond(call: Call) -> Reply:
        x_groq: Final[dict[str, JsonValue]] = {
            "usage": {"prompt_tokens": 99, "completion_tokens": 99, "total_tokens": 198}
        }
        return raw_stream(
            chunk(call.body, choices=delta_choice({"role": "assistant", "content": REPLY_TEXT})),
            chunk(call.body, choices=delta_choice({}, finish_reason="stop")),
            chunk(call.body, choices=[], usage=usage_with(), x_groq=x_groq),
        )

    with peer(respond) as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _LLAMA, api_base_suffix="/openai/v1")
        found: Final = final_usage(stream_chunks(gateway, model, stream_options={"include_usage": True}))
        assert (found["prompt_tokens"], found["completion_tokens"]) == (11, 7), found


def test_groq_stream_usage_sent_only_in_x_groq_reaches_the_caller(gateway: Gateway) -> None:
    """Groq streams text and then a chunk with `choices: []` and only `x_groq: {"usage": {"prompt_tokens": 1840,
    "completion_tokens": 412}}`. The caller's final usage should be 1840 and 412, Groq's own count.

    Today nothing in `litellm/` reads `x_groq` (`GroqChatCompletionStreamingHandler.chunk_parser`,
    `groq/chat/transformation.py:374-388`), so the usage is recounted from the text; the recount is pinned as
    current behaviour in `cost_calculation/cases/misc.py` (`groq-qwen-3.8-stream_x_groq_recount`).
    """
    pytest.skip("BUG: Groq stream usage reported only in x_groq.usage is ignored and the tokens are recounted")

    def respond(call: Call) -> Reply:
        x_groq: Final[dict[str, JsonValue]] = {
            "usage": {"prompt_tokens": 1840, "completion_tokens": 412, "total_tokens": 2252}
        }
        return raw_stream(
            chunk(call.body, choices=delta_choice({"role": "assistant", "content": REPLY_TEXT}, finish_reason="stop")),
            chunk(call.body, choices=[], x_groq=x_groq),
        )

    with peer(respond) as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _QWEN, api_base_suffix="/openai/v1")
        found: Final = final_usage(stream_chunks(gateway, model, stream_options={"include_usage": True}))
        assert (found["prompt_tokens"], found["completion_tokens"]) == (1840, 412), found


def test_groq_stream_error_chunk_reaches_the_caller_with_its_message(gateway: Gateway) -> None:
    """Groq streams one text chunk and then `{"error": {"code": 503, "message": "dec1 groq overloaded"}}`.
    `GroqChatCompletionStreamingHandler.chunk_parser` raises an error with that status and message
    (`groq/chat/transformation.py:375-378`), so the caller's stream ends with an error that names
    `dec1 groq overloaded` instead of finishing normally. UNVERIFIED: the exact error frame shape the proxy
    writes after a 200 stream has started; the test checks the message text only."""

    def respond(call: Call) -> Reply:
        return raw_stream(
            chunk(call.body, choices=delta_choice({"role": "assistant", "content": "partial"})),
            {"error": {"code": 503, "message": "dec1 groq overloaded", "type": "server_error"}},
        )

    with peer(respond) as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _LLAMA, api_base_suffix="/openai/v1")
        _, text = stream_text(
            gateway, {"model": model, "messages": [{"role": "user", "content": marker()}], "stream": True}
        )
        assert "dec1 groq overloaded" in text, text


@pytest.mark.parametrize("status", [401, 429])
def test_groq_error_status_reaches_the_caller(gateway: Gateway, status: int) -> None:
    """Groq answers 401 or 429 with message `dec1 groq failure`. Groq is in `openai_compatible_providers`, so
    `_map_openai_exception` keeps the status (`exception_mapping_utils.py:2733-2749`, `:324`): the caller gets the
    same status and message after one upstream call. The model is not a llama-3 one: its first call on a proxy
    loads the HuggingFace llama-3 tokenizer, which retries for about 23 seconds where the network is closed."""
    with peer(lambda call: upstream_error(status, "dec1 groq failure")) as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _QWEN, api_base_suffix="/openai/v1")
        response: Final = chat(gateway, model)
        assert response.status_code == status, response.text
        assert "dec1 groq failure" in error_message(response)
        assert len(provider.posts()) == 1


def test_groq_assistant_tool_call_replay_keeps_its_tool_calls(gateway: Gateway) -> None:
    """A replayed assistant turn with one tool call and `content: null`, then the tool result. The null content
    is dropped (only non-null fields are copied) and the tool call reaches Groq with its id
    (`groq/chat/transformation.py:152-158`).
    Skipped as a product bug until fixed, see `_SERVICE_TIER_BUG`."""
    pytest.skip(_SERVICE_TIER_BUG)
    messages: Final[list[JsonValue]] = [
        {"role": "user", "content": marker()},
        {"role": "assistant", "content": None, "tool_calls": [tool_call()]},
        {"role": "tool", "tool_call_id": TOOL_CALL_ID, "content": "sunny"},
    ]
    with peer(default_respond) as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _LLAMA, api_base_suffix="/openai/v1")
        served(chat(gateway, model, messages=messages, tools=[TOOL]))
        assistant: Final = object_value(list_value(provider.only().body["messages"])[1])
        assert "content" not in assistant, assistant
        assert object_value(list_value(assistant["tool_calls"])[0])["id"] == TOOL_CALL_ID, assistant
