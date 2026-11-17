"""Contract: reasoning fields and provider quirks on xAI, Mistral (Magistral), DeepSeek and Fireworks AI.

These are the places where an OpenAI-compatible provider differs in ways a generic Rust client would flatten:

- xAI reports `reasoning_tokens` OUTSIDE `completion_tokens` but inside `total_tokens`; the proxy folds them into
  `completion_tokens` so `total = prompt + completion` holds (streaming and not); `grok-3-mini` and `grok-4` take no
  `stop`, and `grok-4` takes no `frequency_penalty`;
- Magistral answers `content` as a list of `thinking` and `text` blocks; the proxy splits it into
  `reasoning_content` and a string `content`. `thinking: {"type": "disabled"}` is sent as
  `reasoning_effort: "none"` (#51329), and Mistral's empty-string content becomes `null`;
- DeepSeek streams `delta.reasoning_content` before the answer, and the caller sees it chunk by chunk;
- Fireworks AI model names without an account path get `accounts/fireworks/models/` prepended, `tool_choice:
  "required"` is sent as `"any"`, a stream always asks for `include_usage`, a tool call returned as JSON text
  in `content` is lifted into `tool_calls`, file content blocks are refused, and the answer's `model` is prefixed
  with `fireworks_ai/`.

Generic param mapping for these providers is in `test_dec1_openai_compatible_chat_wire.py`; their cost rows in
`cost_calculation/cases/`.
"""

from __future__ import annotations

from typing import Final

import pytest
from integration._support.client import Gateway, list_value, object_value
from integration._support.wire import Reply
from integration.providers.dec1_otherprov_support import (
    TOOL,
    Call,
    chat,
    chat_chunk,
    chat_json,
    chat_stream_reply,
    deployment,
    error_message,
    final_usage,
    first_message,
    joined_delta,
    json_reply,
    peer,
    served,
    sse,
    stream_chunks,
)
from pydantic import JsonValue

_XAI_REASONING_USAGE: Final[dict[str, JsonValue]] = {
    "prompt_tokens": 10,
    "completion_tokens": 5,
    "total_tokens": 35,
    "completion_tokens_details": {"reasoning_tokens": 20},
}


def test_xai_reasoning_tokens_are_folded_into_completion_tokens(gateway: Gateway) -> None:
    """xAI answers usage prompt 10, completion 5, total 35 with `reasoning_tokens: 20` (reported outside
    completion).

    The caller's usage has `completion_tokens` 25 and `total_tokens` 35, so prompt plus completion equals total
    (`fold_reasoning_tokens_into_completion`, `xai/chat/transformation.py:350-409`).
    """

    def respond(call: Call) -> Reply:
        return json_reply(chat_json(call.body, usage_block=_XAI_REASONING_USAGE))

    with peer(respond) as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, "xai/grok-4.3", api_base_suffix="/v1")
        found: Final = object_value(served(chat(gateway, model))["usage"])
        assert (found["prompt_tokens"], found["completion_tokens"], found["total_tokens"]) == (10, 25, 35), found


def test_xai_reasoning_tokens_already_inside_completion_are_not_added_twice(gateway: Gateway) -> None:
    """xAI answers prompt 10, completion 25, total 35, reasoning 20 (already folded). The fold is skipped when
    `total == prompt + completion` (`xai/chat/transformation.py:401-402`), so the caller sees completion 25, not 45."""

    def respond(call: Call) -> Reply:
        return json_reply(chat_json(call.body, usage_block={**_XAI_REASONING_USAGE, "completion_tokens": 25}))

    with peer(respond) as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, "xai/grok-4.3", api_base_suffix="/v1")
        assert object_value(served(chat(gateway, model))["usage"])["completion_tokens"] == 25


def test_xai_stream_usage_chunk_folds_reasoning_tokens(gateway: Gateway) -> None:
    """xAI streams the answer and then a usage chunk with `choices: []` carrying the unfolded usage (10/5/35,
    reasoning 20). The caller's final usage has completion 25 and total 35 (`XAIChatCompletionStreamingHandler
    .chunk_parser`, `xai/chat/transformation.py:456-484`)."""

    def respond(call: Call) -> Reply:
        plain: Final = chat_stream_reply(call.body, ({"role": "assistant", "content": "dec1"},))
        assert plain.chunks is not None
        usage_chunk: Final = sse(chat_chunk(call.body, "chatcmpl-dec1xai", usage=_XAI_REASONING_USAGE))
        return Reply(content_type="text/event-stream", chunks=(*plain.chunks[:2], usage_chunk, plain.chunks[-1]))

    with peer(respond) as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, "xai/grok-4.3", api_base_suffix="/v1")
        found: Final = final_usage(stream_chunks(gateway, model, stream_options={"include_usage": True}))
        assert (found["completion_tokens"], found["total_tokens"]) == (25, 35), found


@pytest.mark.parametrize(
    ("model_path", "param", "value"),
    [
        ("xai/grok-3-mini", "stop", ["END"]),
        ("xai/grok-4.3", "frequency_penalty", 0.5),
    ],
)
def test_xai_models_refuse_params_their_family_rejects(
    gateway: Gateway, model_path: str, param: str, value: JsonValue
) -> None:
    """`grok-3-mini` takes no `stop` (`_supports_stop_reason`, `xai/chat/transformation.py:198-201`) and `grok-4`
    takes no `frequency_penalty` (`_supports_frequency_penalty`, `:209-219`). Each is answered 400 with
    `xai does not support parameters: ['<param>']` and the peer is not called."""
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, model_path, api_base_suffix="/v1")
        response: Final = chat(gateway, model, **{param: value})
        assert response.status_code == 400, response.text
        assert f"xai does not support parameters: ['{param}']" in error_message(response)
        assert provider.posts() == ()


def test_magistral_thinking_blocks_become_reasoning_content(gateway: Gateway) -> None:
    """Magistral answers `content: [{"type": "thinking", "thinking": [{"type": "text", "text": "step one"},
    {"type": "text", "text": "step two"}]}, {"type": "text", "text": "dec1 answer"}]`.

    The caller's message has `content: "dec1 answer"` and `reasoning_content: "step one\\nstep two"`
    (`_handle_content_list_to_str_conversion`, `mistral/chat/transformation.py:905-940`).
    """
    content: Final[list[JsonValue]] = [
        {"type": "thinking", "thinking": [{"type": "text", "text": "step one"}, {"type": "text", "text": "step two"}]},
        {"type": "text", "text": "dec1 answer"},
    ]

    def respond(call: Call) -> Reply:
        return json_reply(chat_json(call.body, content=content))

    with peer(respond) as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, "mistral/magistral-medium-latest", api_base_suffix="/v1")
        message: Final = first_message(served(chat(gateway, model)))
        assert (message["content"], message["reasoning_content"]) == ("dec1 answer", "step one\nstep two"), message


@pytest.mark.parametrize(
    ("sent_params", "expected"),
    [({"reasoning_effort": "high"}, "high"), ({"thinking": {"type": "disabled"}}, "none")],
)
def test_magistral_reasoning_switch_is_sent_as_a_native_level(
    gateway: Gateway, sent_params: dict[str, JsonValue], expected: str
) -> None:
    """Magistral's row declares `reasoning_effort_levels` `["none", "high"]` (#51329). `reasoning_effort: "high"` is
    sent as `reasoning_effort: "high"`, and Anthropic-style `thinking: {"type": "disabled"}` is sent as
    `reasoning_effort: "none"` (`_thinking_as_reasoning_effort`, `mistral/chat/transformation.py:82-89`,
    `map_openai_params`, `:294-306`). No `thinking` key reaches Mistral."""
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, "mistral/magistral-medium-latest", api_base_suffix="/v1")
        served(chat(gateway, model, **sent_params))
        sent: Final = provider.only().body
        assert sent["reasoning_effort"] == expected and "thinking" not in sent, sent


def test_mistral_empty_string_content_reaches_the_caller_as_null(gateway: Gateway) -> None:
    """Mistral answers a tool call with `content: ""`. The caller's message has `content: null` and the tool call
    (`mistral/chat/transformation.py:880-894` turns empty strings into None)."""
    tool_call: Final[dict[str, JsonValue]] = {
        "id": "dec1mistral",
        "type": "function",
        "function": {"name": "get_weather", "arguments": '{"city":"Oslo"}'},
    }

    def respond(call: Call) -> Reply:
        return json_reply(
            chat_json(call.body, content="", finish_reason="tool_calls", extra_message={"tool_calls": [tool_call]})
        )

    with peer(respond) as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, "mistral/mistral-large-latest", api_base_suffix="/v1")
        message: Final = first_message(served(chat(gateway, model, tools=[TOOL])))
        assert message["content"] is None, message
        assert object_value(list_value(message["tool_calls"])[0])["id"] == "dec1mistral"


def test_deepseek_stream_reasoning_content_reaches_the_caller_before_the_answer(gateway: Gateway) -> None:
    """DeepSeek streams `delta.reasoning_content` pieces `think ` and `hard`, then content `dec1 answer`.

    The caller's `reasoning_content` deltas join to `think hard` and the `content` deltas to `dec1 answer`; the
    first reasoning delta arrives before the first content delta. UNVERIFIED: read from the generic OpenAI stream
    path that DeepSeek uses (`deepseek/chat/transformation.py`), not a DeepSeek-specific parser.
    """

    def respond(call: Call) -> Reply:
        return chat_stream_reply(
            call.body,
            (
                {"role": "assistant", "content": None, "reasoning_content": "think "},
                {"content": None, "reasoning_content": "hard"},
                {"content": "dec1 answer"},
            ),
        )

    with peer(respond) as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, "deepseek/deepseek-reasoner", api_base_suffix="/v1")
        chunks: Final = stream_chunks(gateway, model)
        assert joined_delta(chunks, "reasoning_content") == "think hard", chunks
        assert joined_delta(chunks, "content") == "dec1 answer", chunks


@pytest.mark.parametrize(
    ("model_path", "upstream"),
    [
        ("fireworks_ai/llama-v3p3-70b-instruct", "accounts/fireworks/models/llama-v3p3-70b-instruct"),
        ("fireworks_ai/models/llama-v3p3-70b-instruct", "accounts/fireworks/models/llama-v3p3-70b-instruct"),
        ("fireworks_ai/accounts/dec1acct/models/custom-1", "accounts/dec1acct/models/custom-1"),
    ],
)
def test_fireworks_model_names_are_resolved_to_account_resources(
    gateway: Gateway, model_path: str, upstream: str
) -> None:
    """A short Fireworks name gets `accounts/fireworks/models/` prepended, a `models/` name gets
    `accounts/fireworks/`, and an `accounts/...` name is sent unchanged (`resolve_fireworks_resource_name`,
    `fireworks_ai/common_utils.py:93-101`, used by `transform_request`, `fireworks_ai/chat/transformation.py:679`).
    The peer body's `model` is the resolved resource name."""
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, model_path, api_base_suffix="/inference/v1")
        served(chat(gateway, model))
        call: Final = provider.only()
        assert call.path == "/inference/v1/chat/completions"
        assert call.body["model"] == upstream, call.body


def test_fireworks_required_tool_choice_is_sent_as_any(gateway: Gateway) -> None:
    """On a Fireworks model whose row has `supports_tool_choice` (`deepseek-v4-flash`), `tool_choice: "required"`
    is sent as `"any"` and the tool loses `strict` (`map_openai_params`, `fireworks_ai/chat/transformation.py:350-356`;
    `_fireworks_tool`, `:493-506`)."""
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = deployment(
            scenario, provider, "fireworks_ai/accounts/fireworks/models/deepseek-v4-flash", api_base_suffix="/v1"
        )
        served(chat(gateway, model, tools=[TOOL], tool_choice="required"))
        sent: Final = provider.only().body
        assert sent["tool_choice"] == "any", sent
        assert "strict" not in object_value(object_value(list_value(sent["tools"])[0])["function"]), sent


def test_fireworks_stream_always_asks_for_usage(gateway: Gateway) -> None:
    """A streamed Fireworks call without `stream_options` reaches the peer with `stream_options: {"include_usage":
    true}` (`fireworks_ai/chat/transformation.py:686-694`), so spend is never estimated."""
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = deployment(
            scenario, provider, "fireworks_ai/accounts/fireworks/models/llama-v3p3-70b-instruct", api_base_suffix="/v1"
        )
        stream_chunks(gateway, model)
        assert provider.only().body["stream_options"] == {"include_usage": True}


def test_fireworks_tool_call_in_content_is_lifted_into_tool_calls(gateway: Gateway) -> None:
    """Fireworks answers a tool call as JSON text in `content`: `{"name": "get_weather", "arguments": "{\\"city\\":
    \\"Oslo\\"}"}`. Since `get_weather` is one of the request's tools, the caller's message has `content: null` and one
    tool call named `get_weather` (`_handle_message_content_with_tool_calls`, `fireworks_ai/chat/transformation.py:713-737`).
    The proxy answers with the caller's model group name as `model`, not the `fireworks_ai/` name the provider config
    sets (`:779-780`), as run 44176682961 showed."""

    def respond(call: Call) -> Reply:
        return json_reply(
            chat_json(call.body, content='{"name": "get_weather", "arguments": "{\\"city\\": \\"Oslo\\"}"}')
        )

    with peer(respond) as provider, gateway.scenario() as scenario:
        model: Final = deployment(
            scenario, provider, "fireworks_ai/accounts/fireworks/models/deepseek-v4-flash", api_base_suffix="/v1"
        )
        body: Final = served(chat(gateway, model, tools=[TOOL]))
        message: Final = first_message(body)
        assert message["content"] is None, message
        function: Final = object_value(object_value(list_value(message["tool_calls"])[0])["function"])
        assert function["name"] == "get_weather", message
        assert body["model"] == model, body["model"]


def test_fireworks_refuses_file_content_blocks_before_the_call(gateway: Gateway) -> None:
    """A user message with a `{"type": "file", ...}` block is answered 400 with `Fireworks AI chat completions does
    not support file content blocks` and the peer is not called (`_transform_messages_helper`,
    `fireworks_ai/chat/transformation.py:519-531`)."""
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = deployment(
            scenario, provider, "fireworks_ai/accounts/fireworks/models/deepseek-v4-flash", api_base_suffix="/v1"
        )
        response: Final = chat(
            gateway,
            model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "read this"},
                        {
                            "type": "file",
                            "file": {"file_data": "data:application/pdf;base64,JVBERi0=", "filename": "a.pdf"},
                        },
                    ],
                }
            ],
        )
        assert response.status_code == 400, response.text
        assert "does not support file content blocks" in error_message(response)
        assert provider.posts() == ()
