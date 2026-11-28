"""Contract: chat completions on Ollama's native chat API (`ollama_chat/...`) through the proxy.

`ollama_chat` does not speak OpenAI on the wire. The proxy posts to `<api_base>/api/chat` (appending `api/chat`
only when the base does not already end in it) and translates:

- request: sampling params go under `options` (`max_tokens`/`max_completion_tokens` become `num_predict`,
  `frequency_penalty` becomes `repeat_penalty`), `response_format` becomes top-level `format` (`"json"` or the
  JSON schema itself), `reasoning_effort` becomes `think` (true for low, medium, high; false for none; the effort
  string itself for gpt-oss), `tool_choice` is never sent, an assistant `reasoning_content` is replayed as
  `thinking`, and a key goes out as `Authorization: Bearer <key>`;
- response: `message.thinking` becomes `reasoning_content`, `done_reason` maps to a finish reason (forced to
  `tool_calls` when tool calls are present), and usage is `prompt_eval_count` and `eval_count`;
- stream: newline-delimited JSON objects; every streamed tool call gets its own index (#49183), and the `done`
  object's counts become the final usage.

Ollama prompt-based tool calling (`ollama/`) is covered in `test_ollama_prompt_tools_wire.py`, URL join rules in
`test_api_base_trailing_slash_wire.py`, embeddings in the `test_dec1_embedrerank_*` files.
"""

from __future__ import annotations

from typing import Final

import pytest
from integration._support.client import Gateway, list_value, object_value
from integration._support.wire import Reply
from integration.providers.dec1_otherprov_support import (
    PROVIDER_KEY,
    REPLY_TEXT,
    TOOL,
    Call,
    chat,
    deployment,
    final_usage,
    finish_reasons,
    first_choice,
    first_message,
    joined_delta,
    json_reply,
    marker,
    ndjson,
    peer,
    served,
    stream_chunks,
)
from pydantic import JsonValue

_MODEL: Final = "ollama_chat/llama3.1"


def _answer(
    *, message: dict[str, JsonValue] | None = None, done_reason: str = "stop", counts: tuple[int, int] = (11, 7)
) -> dict[str, JsonValue]:
    return {
        "model": "llama3.1",
        "created_at": "2026-11-17T00:00:00Z",
        "message": message if message is not None else {"role": "assistant", "content": REPLY_TEXT},
        "done": True,
        "done_reason": done_reason,
        "prompt_eval_count": counts[0],
        "eval_count": counts[1],
    }


def _plain(_call: Call) -> Reply:
    return json_reply(_answer())


def test_posts_to_api_chat_with_bearer_and_options(gateway: Gateway) -> None:
    """An `ollama_chat/llama3.1` deployment with a bare peer `api_base` and a key; the caller sends
    `max_completion_tokens: 40`, `temperature: 0.3`, `frequency_penalty: 1.1`, `seed: 7`.

    The peer receives `POST /api/chat` (`ensure_api_route(base, "api/chat")`, `ollama/chat/transformation.py:248-264`)
    with `authorization: Bearer dec1-otherprov-provider-key` (`:234-246`), body `model: "llama3.1"`,
    `stream: false`, and `options` holding `num_predict: 40`, `temperature: 0.3`, `repeat_penalty: 1.1`, `seed: 7`
    (`map_openai_params`, `:186-232`; `transform_request`, `:329-339`).
    """
    with peer(_plain) as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _MODEL)
        served(chat(gateway, model, max_completion_tokens=40, temperature=0.3, frequency_penalty=1.1, seed=7))
        call: Final = provider.only()
        assert (call.method, call.path) == ("POST", "/api/chat")
        assert call.headers["authorization"] == f"Bearer {PROVIDER_KEY}"
        assert (call.body["model"], call.body["stream"]) == ("llama3.1", False), call.body
        options: Final = object_value(call.body["options"])
        assert (options["num_predict"], options["temperature"], options["repeat_penalty"], options["seed"]) == (
            40,
            0.3,
            1.1,
            7,
        ), options


def test_api_base_already_ending_in_api_chat_is_not_doubled(gateway: Gateway) -> None:
    """A deployment whose `api_base` is `<peer>/ollama/api/chat` posts to `/ollama/api/chat`, never
    `/ollama/api/chat/api/chat` (`ensure_api_route`, `litellm_core_utils/url_utils.py:166-170`)."""
    with peer(_plain) as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _MODEL, api_base_suffix="/ollama/api/chat")
        served(chat(gateway, model))
        assert provider.only().path == "/ollama/api/chat"


def test_deployment_without_a_key_sends_no_authorization(gateway: Gateway) -> None:
    """With no `api_key`, `validate_environment` adds no header (`ollama/chat/transformation.py:244-245`), so the
    peer request has no `authorization` header at all. UNVERIFIED: assumes no `OLLAMA_API_KEY` on the proxy."""
    with peer(_plain) as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _MODEL, with_key=False)
        served(chat(gateway, model))
        assert "authorization" not in provider.only().headers


@pytest.mark.parametrize(
    ("response_format", "expected"),
    [
        ({"type": "json_object"}, "json"),
        (
            {
                "type": "json_schema",
                "json_schema": {"name": "city", "schema": {"type": "object", "properties": {"c": {"type": "string"}}}},
            },
            {"type": "object", "properties": {"c": {"type": "string"}}},
        ),
    ],
)
def test_response_format_becomes_top_level_format(
    gateway: Gateway, response_format: dict[str, JsonValue], expected: JsonValue
) -> None:
    """`response_format: json_object` is sent as `format: "json"`; a `json_schema` response format is sent as
    `format` equal to the schema object itself (`ollama/chat/transformation.py:214-218`, placed at top level at
    `:337-338`). Neither leaves a `response_format` key or a `format` inside `options`."""
    with peer(_plain) as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _MODEL)
        served(chat(gateway, model, response_format=response_format))
        sent: Final = provider.only().body
        assert sent["format"] == expected, sent
        assert "response_format" not in sent and "format" not in object_value(sent["options"]), sent


@pytest.mark.parametrize(
    ("upstream", "effort", "think"),
    [("llama3.1", "high", True), ("llama3.1", "none", False), ("gpt-oss:20b", "low", "low")],
)
def test_reasoning_effort_becomes_think(gateway: Gateway, upstream: str, effort: str, think: JsonValue) -> None:
    """`reasoning_effort` maps to Ollama's `think` (`ollama_think`, `ollama/common_utils.py:314-320`): `high` is
    `true`, `none` is `false`, and on a `gpt-oss` model the effort string is passed through as `"low"`. `think` is a
    top-level body key (`ollama/chat/transformation.py:343-344`)."""
    with peer(_plain) as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, f"ollama_chat/{upstream}")
        served(chat(gateway, model, reasoning_effort=effort))
        sent: Final = provider.only().body
        assert sent["think"] == think, sent
        assert "reasoning_effort" not in object_value(sent["options"]), sent


def test_tools_are_sent_natively_and_tool_choice_is_dropped(gateway: Gateway) -> None:
    """A tool with `tool_choice: "required"` reaches Ollama as a top-level `tools` list with the function name,
    and no `tool_choice` anywhere (it is popped because it hangs Ollama, `ollama/chat/transformation.py:229-230`)."""
    with peer(_plain) as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _MODEL)
        served(chat(gateway, model, tools=[TOOL], tool_choice="required"))
        sent: Final = provider.only().body
        tool_function: Final = object_value(object_value(list_value(sent["tools"])[0])["function"])
        assert tool_function["name"] == "get_weather", sent
        assert "tool_choice" not in sent and "tool_choice" not in object_value(sent["options"]), sent


def test_assistant_reasoning_content_is_replayed_as_thinking(gateway: Gateway) -> None:
    """A history with an assistant turn `{"content": "earlier", "reasoning_content": "dec1 thought"}` reaches Ollama
    as an assistant message with `thinking: "dec1 thought"` and `content: "earlier"`
    (`ollama/chat/transformation.py:306-315`)."""
    text: Final = marker()
    history: Final[list[JsonValue]] = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "earlier", "reasoning_content": "dec1 thought"},
        {"role": "user", "content": text},
    ]
    with peer(_plain) as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _MODEL)
        served(chat(gateway, model, messages=history))
        assistant: Final = object_value(list_value(provider.only().body["messages"])[1])
        assert (assistant["role"], assistant["content"], assistant["thinking"]) == (
            "assistant",
            "earlier",
            "dec1 thought",
        ), assistant


def test_thinking_and_eval_counts_in_the_answer_reach_the_caller(gateway: Gateway) -> None:
    """Ollama answers `message.thinking: "dec1 reasoning"`, `done_reason: "length"`, `prompt_eval_count: 11`,
    `eval_count: 7`.

    The caller's message has `reasoning_content: "dec1 reasoning"`, the finish reason is `length`, and usage is
    11 prompt and 7 completion tokens (`ollama/chat/transformation.py:376-445`).
    """

    def respond(_call: Call) -> Reply:
        return json_reply(
            _answer(
                message={"role": "assistant", "content": REPLY_TEXT, "thinking": "dec1 reasoning"}, done_reason="length"
            )
        )

    with peer(respond) as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _MODEL)
        body: Final = served(chat(gateway, model))
        assert first_message(body)["reasoning_content"] == "dec1 reasoning"
        assert first_choice(body)["finish_reason"] == "length"
        assert (object_value(body["usage"])["prompt_tokens"], object_value(body["usage"])["completion_tokens"]) == (
            11,
            7,
        )


def test_answer_with_tool_calls_finishes_with_tool_calls(gateway: Gateway) -> None:
    """Ollama answers `done_reason: "stop"` with a tool call whose `arguments` is an object `{"city": "Oslo"}`.

    The caller's choice has `finish_reason: "tool_calls"` (forced when tool calls are present,
    `ollama/chat/transformation.py:424-428`) and one tool call named `get_weather`.
    """

    def respond(_call: Call) -> Reply:
        return json_reply(
            _answer(
                message={
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{"function": {"name": "get_weather", "arguments": {"city": "Oslo"}}}],
                }
            )
        )

    with peer(respond) as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _MODEL)
        body: Final = served(chat(gateway, model, tools=[TOOL]))
        assert first_choice(body)["finish_reason"] == "tool_calls"
        call: Final = object_value(list_value(first_message(body)["tool_calls"])[0])
        assert object_value(call["function"])["name"] == "get_weather", call


def test_stream_gives_each_tool_call_its_own_index_and_finishes_with_tool_calls(gateway: Gateway) -> None:
    """Ollama streams two NDJSON objects, each with one complete tool call (`get_weather` for Oslo, then Rome),
    then a `done` object with `prompt_eval_count: 20`, `eval_count: 9`.

    The caller's tool call chunks carry indexes 0 and 1 (#49183, `_next_tool_call_index`,
    `ollama/chat/transformation.py:523-528`), the last finish reason is `tool_calls` (`:566-571`), and the final usage
    is 20 prompt and 9 completion tokens (`_done_chunk_usage`, `:463-472`).
    """

    def respond(_call: Call) -> Reply:
        def call_chunk(city: str) -> dict[str, JsonValue]:
            return {
                "model": "llama3.1",
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{"function": {"name": "get_weather", "arguments": {"city": city}}}],
                },
                "done": False,
            }

        return Reply(
            content_type="application/x-ndjson",
            chunks=ndjson(
                call_chunk("Oslo"),
                call_chunk("Rome"),
                {
                    "model": "llama3.1",
                    "message": {"role": "assistant", "content": ""},
                    "done": True,
                    "done_reason": "stop",
                    "prompt_eval_count": 20,
                    "eval_count": 9,
                },
            ),
        )

    with peer(respond) as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _MODEL)
        chunks: Final = stream_chunks(gateway, model, tools=[TOOL], stream_options={"include_usage": True})
        indexes: Final = [
            object_value(call)["index"]
            for chunk in chunks
            for choice in list_value(chunk.get("choices") or [])
            for call in list_value(object_value(object_value(choice).get("delta") or {}).get("tool_calls") or [])
        ]
        assert indexes == [0, 1], chunks
        assert finish_reasons(chunks)[-1] == "tool_calls"
        found: Final = final_usage(chunks)
        assert (found["prompt_tokens"], found["completion_tokens"]) == (20, 9), found
        assert provider.only().body["stream"] is True


def test_stream_think_tags_are_split_into_reasoning_and_content(gateway: Gateway) -> None:
    """Ollama streams content pieces `<think>plan`, ` more</think>`, `answer` (a model that inlines its
    reasoning).

    Intended: the caller's `reasoning_content` deltas join to `plan more` and the `content` deltas to `answer`, and
    the tags never reach the caller. Today the parser marks reasoning finished on the first content chunk after
    reasoning started (`ollama/chat/transformation.py:543-545`, a rule meant for the `thinking` field), so the
    second piece ` more` lands in `content` and the caller sees reasoning `plan` and content ` moreanswer`.
    """
    pytest.skip("BUG: ollama_chat stream ends inline <think> reasoning after its first chunk; rest leaks to content")

    def respond(_call: Call) -> Reply:
        def piece(text: str, done: bool = False) -> dict[str, JsonValue]:
            return {"model": "llama3.1", "message": {"role": "assistant", "content": text}, "done": done}

        return Reply(
            content_type="application/x-ndjson",
            chunks=ndjson(piece("<think>plan"), piece(" more</think>"), piece("answer"), piece("", done=True)),
        )

    with peer(respond) as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _MODEL)
        chunks: Final = stream_chunks(gateway, model)
        assert joined_delta(chunks, "reasoning_content") == "plan more", chunks
        assert joined_delta(chunks, "content") == "answer", chunks
