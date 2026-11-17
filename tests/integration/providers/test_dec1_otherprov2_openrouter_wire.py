"""Contract: OpenRouter provider routing, reasoning, cache control, attribution headers, reported cost and errors.

An `openrouter/<vendor>/<model>` deployment posts to `<api_base>/chat/completions` with `Authorization: Bearer
<key>`, `HTTP-Referer` and `X-Title` (`_complete_openrouter`, `litellm/main.py:3659-3730`) through
`OpenrouterConfig` (`litellm/llms/openrouter/chat/transformation.py`). OpenRouter is NOT in
`openai_compatible_providers`, so params the proxy does not know (`provider`, `reasoning`) are copied to the top
level of the body directly (`litellm/utils.py:6399-6404`). Contract points here:

- OpenRouter's `provider` routing object and `reasoning` object reach the body as sent;
- caller `extra_headers` override the default attribution headers;
- `cache_control` on a message moves into the last content block for Claude-like models;
- `usage.cost` may be a numeric string; the proxy still reports it as the response cost;
- streamed `delta.reasoning` is surfaced as `reasoning_content`; an `error` chunk in a stream raises with its code;
- HTTP 422 from OpenRouter reaches callers as 400, and an error choice with no code inside a 200 becomes 502 with a
  fixed message.

The default headers, `usage.include`, `transforms`/`models`/`route`, `max` effort and numeric `usage.cost` are in
`test_dec1_openai_compatible_chat_wire.py`.
"""

from __future__ import annotations

from typing import Final

import pytest
from integration._support.client import Gateway, list_value, object_value
from integration._support.wire import Reply
from integration.providers.dec1_otherprov2_support import (
    chunk,
    delta_choice,
    raw_stream,
    stream_text,
    upstream_error,
    usage_with,
)
from integration.providers.dec1_otherprov_support import (
    Call,
    chat,
    chat_json,
    deployment,
    error_message,
    joined_delta,
    json_reply,
    marker,
    peer,
    served,
    stream_chunks,
)
from pydantic import JsonValue

_GPT: Final = "openrouter/openai/gpt-4o-mini"
_CLAUDE: Final = "openrouter/anthropic/claude-sonnet-4.5"
_BASE: Final = "/api/v1"


def test_openrouter_provider_routing_object_reaches_the_body_as_sent(gateway: Gateway) -> None:
    """The caller sends OpenRouter's `provider: {"order": ["dec1-a", "dec1-b"], "allow_fallbacks": false,
    "data_collection": "deny"}`. It is not an OpenAI param and OpenRouter is not an OpenAI SDK provider, so it is
    copied to `optional_params` (`litellm/utils.py:6399-6404`) and OpenRouter receives the same object at the top
    level of `POST /api/v1/chat/completions`."""
    routing: Final[dict[str, JsonValue]] = {
        "order": ["dec1-a", "dec1-b"],
        "allow_fallbacks": False,
        "data_collection": "deny",
    }
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _GPT, api_base_suffix=_BASE)
        served(chat(gateway, model, provider=routing))
        call: Final = provider.only()
        assert call.path == "/api/v1/chat/completions", call.path
        assert call.body["provider"] == routing, call.body


def test_openrouter_reasoning_object_reaches_the_body_as_sent(gateway: Gateway) -> None:
    """The caller sends OpenRouter's unified `reasoning: {"max_tokens": 2048, "exclude": true}` to a Claude model.
    OpenRouter receives the same object at the top level (`litellm/utils.py:6399-6404`) and no `thinking` key is
    invented from it."""
    reasoning: Final[dict[str, JsonValue]] = {"max_tokens": 2048, "exclude": True}
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _CLAUDE, api_base_suffix=_BASE)
        served(chat(gateway, model, reasoning=reasoning))
        sent: Final = provider.only().body
        assert sent["reasoning"] == reasoning and "thinking" not in sent, sent


def test_openrouter_caller_extra_headers_override_the_attribution_headers(gateway: Gateway) -> None:
    """The caller sends `extra_headers: {"X-Title": "dec1-app"}`. The proxy starts from `HTTP-Referer:
    https://litellm.ai` and `X-Title: liteLLM` and then updates them with the caller's headers
    (`litellm/main.py:3684-3696`), so OpenRouter receives `x-title: dec1-app` and the default referer.
    UNVERIFIED: assumes the shared proxy has no `OR_SITE_URL`."""
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _GPT, api_base_suffix=_BASE)
        served(chat(gateway, model, extra_headers={"X-Title": "dec1-app"}))
        headers: Final = provider.only().headers
        assert (headers["x-title"], headers["http-referer"]) == ("dec1-app", "https://litellm.ai"), headers


def test_openrouter_moves_message_cache_control_into_the_last_content_block_for_claude(gateway: Gateway) -> None:
    """A user message with string content and message-level `cache_control: {"type": "ephemeral"}` on a Claude
    model. OpenRouter wants the marker inside content, so it receives `content: [{"type": "text", "text": <the
    text>, "cache_control": {"type": "ephemeral"}}]` and no message-level `cache_control`
    (`_move_cache_control_to_content`, `openrouter/chat/transformation.py:130-170`, `:186-187`)."""
    text: Final = marker()
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _CLAUDE, api_base_suffix=_BASE)
        served(
            chat(
                gateway,
                model,
                messages=[{"role": "user", "content": text, "cache_control": {"type": "ephemeral"}}],
            )
        )
        message: Final = object_value(list_value(provider.only().body["messages"])[0])
        assert "cache_control" not in message, message
        assert message["content"] == [{"type": "text", "text": text, "cache_control": {"type": "ephemeral"}}]


def test_openrouter_cache_control_marks_only_the_last_block_of_a_list(gateway: Gateway) -> None:
    """A Claude user message with two text blocks and message-level `cache_control`. Only the second block gets
    the marker, to stay within Anthropic's four breakpoints (`openrouter/chat/transformation.py:146-156`)."""
    first: Final = marker()
    second: Final = marker()
    content: Final[list[JsonValue]] = [{"type": "text", "text": first}, {"type": "text", "text": second}]
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _CLAUDE, api_base_suffix=_BASE)
        served(
            chat(
                gateway, model, messages=[{"role": "user", "content": content, "cache_control": {"type": "ephemeral"}}]
            )
        )
        sent: Final = list_value(object_value(list_value(provider.only().body["messages"])[0])["content"])
        assert object_value(sent[0]) == {"type": "text", "text": first}, sent
        assert object_value(sent[1]) == {"type": "text", "text": second, "cache_control": {"type": "ephemeral"}}


def test_openrouter_string_usage_cost_is_the_response_cost(gateway: Gateway) -> None:
    """OpenRouter answers `usage.cost: "0.0042"` as a string. `transform_response` accepts `str | int | float`
    and stores `finite_cost(float(...))` as the provider response cost (`openrouter/chat/transformation.py:
    247-267`), so the caller's `x-litellm-response-cost` is 0.0042, not the 0.025 the deployment prices give."""

    def respond(call: Call) -> Reply:
        return json_reply(chat_json(call.body, usage_block=usage_with(cost="0.0042")))

    with peer(respond) as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _GPT, api_base_suffix=_BASE)
        response: Final = chat(gateway, model)
        served(response)
        assert float(response.headers["x-litellm-response-cost"]) == pytest.approx(0.0042), response.headers


def test_openrouter_stream_reasoning_delta_reaches_the_caller_as_reasoning_content(gateway: Gateway) -> None:
    """OpenRouter streams `delta.reasoning: "dec1 thinking"` before its text. `OpenRouterChatCompletionStreamingHandler`
    copies `reasoning` to `reasoning_content` on every choice (`openrouter/chat/transformation.py:321-324`), so the
    caller's joined `reasoning_content` is `dec1 thinking` and its joined content is `answer`."""

    def respond(call: Call) -> Reply:
        return raw_stream(
            chunk(call.body, choices=delta_choice({"role": "assistant", "content": "", "reasoning": "dec1 thinking"})),
            chunk(call.body, choices=delta_choice({"content": "answer"})),
            chunk(call.body, choices=delta_choice({}, finish_reason="stop"), usage=usage_with()),
        )

    with peer(respond) as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _CLAUDE, api_base_suffix=_BASE)
        chunks: Final = stream_chunks(gateway, model)
        assert joined_delta(chunks, "reasoning_content") == "dec1 thinking", chunks
        assert joined_delta(chunks, "content") == "answer", chunks


def test_openrouter_stream_error_chunk_reaches_the_caller_with_its_message(gateway: Gateway) -> None:
    """OpenRouter streams a text chunk and then `{"error": {"code": 429, "message": "dec1 or throttled"}}` (how it
    reports a provider failure after the 200). The stream handler raises with message `Message: dec1 or throttled,
    Metadata: {}, User ID: ` (`openrouter/chat/transformation.py:303-319`), so the caller's stream carries an
    error naming `dec1 or throttled`. UNVERIFIED: the error frame shape after a started stream; only the text is
    pinned."""

    def respond(call: Call) -> Reply:
        return raw_stream(
            chunk(call.body, choices=delta_choice({"role": "assistant", "content": "partial"})),
            {"error": {"code": 429, "message": "dec1 or throttled"}},
        )

    with peer(respond) as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _GPT, api_base_suffix=_BASE)
        _, text = stream_text(
            gateway, {"model": model, "messages": [{"role": "user", "content": marker()}], "stream": True}
        )
        assert "dec1 or throttled" in text, text


def test_openrouter_error_choice_without_a_code_becomes_502_with_the_fixed_message(gateway: Gateway) -> None:
    """OpenRouter answers HTTP 200 with one choice whose `finish_reason` is `error` and no `error` object.
    `_non_streaming_error` falls back to status 502 and the message `OpenRouter returned finish_reason 'error':
    the provider failed during generation` (`openrouter/chat/transformation.py:37-57`)."""

    def respond(call: Call) -> Reply:
        return json_reply(chat_json(call.body, content="", finish_reason="error"))

    with peer(respond) as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _GPT, api_base_suffix=_BASE)
        response: Final = chat(gateway, model)
        assert response.status_code == 502, response.text
        assert "OpenRouter returned finish_reason 'error'" in error_message(response)


def test_openrouter_error_choice_with_an_out_of_range_code_becomes_502(gateway: Gateway) -> None:
    """The error choice carries `error: {"code": 200, "message": "dec1 bad code"}`. Codes outside 400 to 599 are
    replaced by 502 (`openrouter/chat/transformation.py:48-49`), while the message is kept."""

    def respond(call: Call) -> Reply:
        payload: Final = chat_json(call.body, content="", finish_reason="error")
        choice: Final = object_value(list_value(payload["choices"])[0])
        choice["error"] = {"code": 200, "message": "dec1 bad code"}
        return json_reply({**payload, "choices": [choice]})

    with peer(respond) as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _GPT, api_base_suffix=_BASE)
        response: Final = chat(gateway, model)
        assert response.status_code == 502, response.text
        assert "dec1 bad code" in error_message(response)


def test_openrouter_unprocessable_entity_reaches_the_caller_as_400(gateway: Gateway) -> None:
    """OpenRouter answers HTTP 422 `dec1 bad schema`. `_map_openrouter_exception` raises a `BadRequestError` for
    422 (`exception_mapping_utils.py:2335-2342`), so the caller gets 400 with the message."""
    with peer(lambda call: upstream_error(422, "dec1 bad schema")) as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _GPT, api_base_suffix=_BASE)
        response: Final = chat(gateway, model)
        assert response.status_code == 400, response.text
        assert "dec1 bad schema" in error_message(response)


@pytest.mark.parametrize("status", [401, 429, 503])
def test_openrouter_error_status_reaches_the_caller(gateway: Gateway, status: int) -> None:
    """OpenRouter answers 401, 429 or 503 with `dec1 or failure`; `_map_openrouter_exception` keeps each status
    (`exception_mapping_utils.py:2312-2358`) and the caller sees it with the message after one call."""
    with peer(lambda call: upstream_error(status, "dec1 or failure")) as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _GPT, api_base_suffix=_BASE)
        response: Final = chat(gateway, model)
        assert response.status_code == status, response.text
        assert "dec1 or failure" in error_message(response)
        assert len(provider.posts()) == 1


def test_openrouter_thinking_is_accepted_on_any_model(gateway: Gateway) -> None:
    """`OpenrouterConfig.get_supported_openai_params` always adds `reasoning_effort` and `thinking`
    (`openrouter/chat/transformation.py:76-81`), so `reasoning_effort: "low"` on `openai/gpt-4o-mini` (no
    reasoning flag in the price map) is not refused and reaches OpenRouter as sent."""
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _GPT, api_base_suffix=_BASE)
        served(chat(gateway, model, reasoning_effort="low"))
        assert provider.only().body["reasoning_effort"] == "low"
