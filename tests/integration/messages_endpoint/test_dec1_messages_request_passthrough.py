"""Request body contract of `POST /v1/messages` for an Anthropic deployment.

A caller sends an Anthropic Messages request to the proxy naming a proxy model. The proxy must send
the provider the same request with one change: `model` becomes the deployment's backend model. Every
Anthropic field the caller set (system, tools, tool_choice, thinking, metadata, stop_sequences,
sampling params, cache_control marks, image and document blocks, tool history) must arrive at
`POST <api_base>/v1/messages` with the same value, and nothing the proxy uses internally
(`litellm_metadata`, routing and logging fields, the caller's proxy key) may be added to that body.

Each test creates one `anthropic/claude-sonnet-4-5-20250929` deployment whose provider is an owned
TCP peer, sends one request with the master key, and reads the JSON the peer received. A rewrite
that re-serializes the request through typed structs is the likely way to lose or reorder a field
here, which is why most tests compare whole values and not just key presence.
"""

from __future__ import annotations

from typing import Final

import pytest
from integration._support.client import Gateway
from integration.messages_endpoint.dec1_messages_support import (
    BACKEND,
    NO_CACHE,
    deployment,
    echo_message,
    marker,
)
from pydantic import JsonValue

_TOOL: Final[dict[str, JsonValue]] = {
    "name": "get_weather",
    "description": "Look up synthetic weather for a city.",
    "input_schema": {
        "type": "object",
        "properties": {"city": {"type": "string", "description": "City name"}, "unit": {"enum": ["c", "f"]}},
        "required": ["city"],
    },
}
_PNG_BASE64: Final = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
_PDF_BASE64: Final = "JVBERi0xLjQKJSBzeW50aGV0aWMgcGRmIGZvciBhIHBhc3N0aHJvdWdoIHRlc3QKJSVFT0YK"


def _sent(gateway: Gateway, **fields: JsonValue) -> dict[str, JsonValue]:
    """Send one request carrying `fields` and return the JSON body the provider received."""
    with deployment(gateway, echo_message) as rig:
        response: Final = rig.post(rig.body(**fields))
        assert response.status_code == 200, response.text
        return rig.one_body()


def test_the_proxy_model_name_is_replaced_by_the_deployment_backend_model(gateway: Gateway) -> None:
    """A deployment is created under a random proxy model name that maps to `anthropic/claude-sonnet-4-5-20250929`.
    The caller names the proxy model; the provider must receive `"model": "claude-sonnet-4-5-20250929"`,
    without the `anthropic/` prefix and without the proxy model name."""
    assert _sent(gateway)["model"] == BACKEND


def test_the_request_is_sent_as_one_post_to_v1_messages_on_the_deployment_api_base(gateway: Gateway) -> None:
    """One caller request to `/v1/messages` must produce exactly one provider request, a `POST` whose
    path is `/v1/messages` on the deployment's `api_base`. No query string is added, even though the
    api_base has no `/v1` suffix."""
    with deployment(gateway, echo_message) as rig:
        assert rig.post(rig.body()).status_code == 200
        call: Final = rig.one_call()
        assert (call.method, call.target) == ("POST", "/v1/messages")


def test_max_tokens_and_messages_reach_the_provider_unchanged(gateway: Gateway) -> None:
    """The two required fields are forwarded as sent: `max_tokens` keeps its integer value and a
    string-content user message stays a string, it is not rewritten into a list of text blocks."""
    prompt: Final = f"ping {marker()}"
    with deployment(gateway, echo_message) as rig:
        assert rig.post(rig.body(prompt, max_tokens=321)).status_code == 200
        body: Final = rig.one_body()
        assert body["max_tokens"] == 321
        assert body["messages"] == [{"role": "user", "content": prompt}]


def test_a_request_without_optional_fields_adds_no_optional_fields_upstream(gateway: Gateway) -> None:
    """A request with only `model`, `max_tokens` and `messages` must reach the provider with only those
    keys plus an optional `"stream": false`. In particular no `system`, `tools`, `metadata`,
    `temperature` or `thinking` key may appear with a null or default value, because Anthropic rejects nulls."""
    body: Final = _sent(gateway)
    assert set(body) - {"stream"} == {"model", "max_tokens", "messages"}, body
    assert body.get("stream", False) is False


def test_a_string_system_prompt_is_forwarded_as_a_string(gateway: Gateway) -> None:
    """`system` given as a plain string must arrive as the same plain string under the top-level
    `system` key, not as a message with role system and not as a list of blocks."""
    system: Final = f"You are a synthetic policy {marker()}."
    assert _sent(gateway, system=system)["system"] == system


def test_system_blocks_keep_their_order_text_and_cache_control(gateway: Gateway) -> None:
    """`system` given as two text blocks, the second carrying `cache_control: {"type": "ephemeral", "ttl": "1h"}`,
    must arrive as the same two blocks in the same order with the cache mark and its ttl intact.
    Losing the mark silently disables prompt caching and raises the caller's bill."""
    system: Final[list[JsonValue]] = [
        {"type": "text", "text": f"first {marker()}"},
        {"type": "text", "text": "second", "cache_control": {"type": "ephemeral", "ttl": "1h"}},
    ]
    assert _sent(gateway, system=system)["system"] == system


def test_tools_reach_the_provider_with_name_description_and_schema_unchanged(gateway: Gateway) -> None:
    """A client tool with a name, description and a JSON schema containing nested `properties`, an
    `enum` and `required` must arrive byte-for-byte equal as the only element of `tools`. The proxy
    must not wrap it in an OpenAI `function` object or prune schema keywords on this route."""
    assert _sent(gateway, tools=[_TOOL])["tools"] == [_TOOL]


def test_a_tool_cache_control_mark_is_forwarded(gateway: Gateway) -> None:
    """A tool definition carrying `cache_control: {"type": "ephemeral"}` must reach the provider with
    that key still on the tool, since it is the cache breakpoint for the whole tool list."""
    tool: Final[dict[str, JsonValue]] = {**_TOOL, "cache_control": {"type": "ephemeral"}}
    assert _sent(gateway, tools=[tool])["tools"] == [tool]


@pytest.mark.parametrize(
    "tool_choice",
    (
        {"type": "auto"},
        {"type": "any"},
        {"type": "tool", "name": "get_weather"},
        {"type": "none"},
        {"type": "auto", "disable_parallel_tool_use": True},
    ),
    ids=("auto", "any", "named_tool", "none", "auto_without_parallel"),
)
def test_every_tool_choice_shape_is_forwarded_as_sent(gateway: Gateway, tool_choice: dict[str, JsonValue]) -> None:
    """Each Anthropic `tool_choice` shape (auto, any, a named tool, none, and auto with
    `disable_parallel_tool_use`) must arrive at the provider exactly as the caller wrote it, next to
    the unchanged tool list."""
    body: Final = _sent(gateway, tools=[_TOOL], tool_choice=tool_choice)
    assert body["tool_choice"] == tool_choice
    assert body["tools"] == [_TOOL]


def test_enabled_thinking_with_a_budget_is_forwarded_unchanged(gateway: Gateway) -> None:
    """On a model that takes thinking budgets, `thinking: {"type": "enabled", "budget_tokens": 2048}` with
    `max_tokens: 4096` must arrive with the same type and budget, and `max_tokens` must stay 4096."""
    body: Final = _sent(gateway, max_tokens=4096, thinking={"type": "enabled", "budget_tokens": 2048})
    assert body["thinking"] == {"type": "enabled", "budget_tokens": 2048}
    assert body["max_tokens"] == 4096


def test_metadata_user_id_is_forwarded_to_the_provider(gateway: Gateway) -> None:
    """Anthropic's `metadata: {"user_id": ...}` is a provider field on this route, unlike on OpenAI
    routes where `metadata` is proxy bookkeeping. It must arrive at the provider as
    `{"user_id": "<same value>"}` (`handler.py` `validate_anthropic_api_metadata`)."""
    user_id: Final = f"synthetic-user-{marker()}"
    assert _sent(gateway, metadata={"user_id": user_id})["metadata"] == {"user_id": user_id}


def test_litellm_metadata_is_never_sent_to_the_provider(gateway: Gateway) -> None:
    """`litellm_metadata` is how a caller passes proxy tags and an end user on this route. The provider
    body must contain neither a `litellm_metadata` key nor the tag value anywhere in a `metadata` key."""
    tag: Final = marker("tag")
    body: Final = _sent(gateway, litellm_metadata={"tags": [tag], "user": "synthetic-end-user"})
    assert "litellm_metadata" not in body, body
    assert "metadata" not in body, body


def test_stop_sequences_are_forwarded_in_order(gateway: Gateway) -> None:
    """`stop_sequences` with three strings, one of them containing a newline, must arrive as the
    same list in the same order."""
    stops: Final[list[JsonValue]] = ["\n\nHuman:", "END", "###"]
    assert _sent(gateway, stop_sequences=stops)["stop_sequences"] == stops


def test_top_k_is_forwarded(gateway: Gateway) -> None:
    """`top_k: 40` is an Anthropic-only sampling field. On a model whose catalog row does not forbid
    sampling params it must arrive as the integer 40."""
    assert _sent(gateway, top_k=40)["top_k"] == 40


def test_temperature_is_forwarded(gateway: Gateway) -> None:
    """`temperature: 0.3` must arrive as 0.3 on a model that accepts sampling params and has no
    thinking enabled in the request."""
    assert _sent(gateway, temperature=0.3)["temperature"] == 0.3


def test_top_p_is_forwarded(gateway: Gateway) -> None:
    """`top_p: 0.9` must arrive as 0.9 on a model that accepts sampling params."""
    assert _sent(gateway, top_p=0.9)["top_p"] == 0.9


def test_a_top_level_cache_control_is_forwarded(gateway: Gateway) -> None:
    """Anthropic's automatic prompt caching is requested with a top-level
    `cache_control: {"type": "ephemeral"}`. It is in the proxy's list of Messages params
    (`AnthropicMessagesRequestOptionalParams.cache_control`) and must arrive unchanged at the top level."""
    assert _sent(gateway, cache_control={"type": "ephemeral"})["cache_control"] == {"type": "ephemeral"}


def test_a_message_block_cache_control_mark_is_forwarded(gateway: Gateway) -> None:
    """A user text block carrying `cache_control: {"type": "ephemeral"}` must reach the provider as a
    list-content message whose block still has the mark. The block list must not be collapsed to a string."""
    messages: Final[list[JsonValue]] = [
        {
            "role": "user",
            "content": [{"type": "text", "text": f"cached {marker()}", "cache_control": {"type": "ephemeral"}}],
        }
    ]
    assert _sent(gateway, messages=messages)["messages"] == messages


def test_a_base64_image_block_is_forwarded_with_its_media_type_and_data(gateway: Gateway) -> None:
    """A user turn with a text block and a base64 PNG image block must arrive with the image block's
    `source` unchanged: `type: base64`, `media_type: image/png` and the identical base64 string."""
    messages: Final[list[JsonValue]] = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f"describe {marker()}"},
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": _PNG_BASE64}},
            ],
        }
    ]
    assert _sent(gateway, messages=messages)["messages"] == messages


def test_a_url_image_block_is_forwarded_without_being_downloaded(gateway: Gateway) -> None:
    """An image block with `source: {"type": "url", "url": ...}` must arrive with the same URL source.
    The proxy must not fetch the URL and inline it as base64 on this route; the URL here does not resolve."""
    messages: Final[list[JsonValue]] = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f"describe {marker()}"},
                {"type": "image", "source": {"type": "url", "url": "https://images.invalid/synthetic.png"}},
            ],
        }
    ]
    assert _sent(gateway, messages=messages)["messages"] == messages


def test_a_base64_pdf_document_block_is_forwarded_with_title_context_and_citations(gateway: Gateway) -> None:
    """A document block with a base64 PDF source plus `title`, `context` and `citations: {"enabled": true}`
    must arrive with every one of those keys and the same base64 data."""
    messages: Final[list[JsonValue]] = [
        {
            "role": "user",
            "content": [
                {
                    "type": "document",
                    "source": {"type": "base64", "media_type": "application/pdf", "data": _PDF_BASE64},
                    "title": "Synthetic report",
                    "context": "A fixture document.",
                    "citations": {"enabled": True},
                },
                {"type": "text", "text": f"summarize {marker()}"},
            ],
        }
    ]
    assert _sent(gateway, messages=messages)["messages"] == messages


def test_a_plain_text_document_block_is_forwarded(gateway: Gateway) -> None:
    """A document block with `source: {"type": "text", "media_type": "text/plain", "data": ...}` must
    arrive unchanged, with the text kept under `source.data` and not turned into a text block."""
    messages: Final[list[JsonValue]] = [
        {
            "role": "user",
            "content": [
                {
                    "type": "document",
                    "source": {"type": "text", "media_type": "text/plain", "data": f"The sky is green. {marker()}"},
                },
                {"type": "text", "text": "what colour is the sky?"},
            ],
        }
    ]
    assert _sent(gateway, messages=messages)["messages"] == messages


def test_tool_use_and_tool_result_history_is_forwarded_with_ids_and_inputs(gateway: Gateway) -> None:
    """A three-turn history (user text, assistant `tool_use` with id `toolu_synthetic_01` and an input
    object, user `tool_result` naming that id with `is_error: false`) must arrive with the same
    roles, ids, input object and result content. The tool definition travels next to it."""
    messages: Final[list[JsonValue]] = [
        {"role": "user", "content": f"weather in Paris? {marker()}"},
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Let me check."},
                {"type": "tool_use", "id": "toolu_synthetic_01", "name": "get_weather", "input": {"city": "Paris"}},
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_synthetic_01",
                    "content": [{"type": "text", "text": "18C and clear"}],
                    "is_error": False,
                }
            ],
        },
    ]
    body: Final = _sent(gateway, messages=messages, tools=[_TOOL])
    assert body["messages"] == messages
    assert body["tools"] == [_TOOL]


def test_an_assistant_prefill_turn_is_forwarded_as_the_last_message(gateway: Gateway) -> None:
    """A conversation that ends with an assistant turn (a prefill) must arrive with that assistant
    turn still last and its text unchanged. The proxy must not append a user turn or drop the prefill."""
    messages: Final[list[JsonValue]] = [
        {"role": "user", "content": f"answer in JSON {marker()}"},
        {"role": "assistant", "content": "{"},
    ]
    assert _sent(gateway, messages=messages)["messages"] == messages


def test_stream_true_is_forwarded_as_stream_true(gateway: Gateway) -> None:
    """A caller asking for `stream: true` must make the provider request carry `"stream": true`, so
    the provider streams and the proxy does not fake a stream from a complete response."""
    with deployment(gateway, echo_message) as rig:
        response, _text = rig.stream(rig.body(stream=True))
        assert response.status_code == 200
        assert rig.one_body()["stream"] is True


def test_an_unknown_top_level_field_is_not_forwarded(gateway: Gateway) -> None:
    """A top-level key that is not an Anthropic Messages param (`frequency_penalty`, an OpenAI field)
    is filtered out before the provider call (`utils.py` `get_requested_anthropic_messages_optional_param`).
    The request still succeeds and the provider body has no such key."""
    body: Final = _sent(gateway, frequency_penalty=0.5)
    assert "frequency_penalty" not in body, body


def test_no_proxy_internal_field_leaks_into_the_provider_body(gateway: Gateway) -> None:
    """With tags, a proxy end user and a request timeout set by the caller, the provider body must
    contain none of the proxy's own keys: `litellm_metadata`, `litellm_params`, `litellm_call_id`,
    `litellm_logging_obj`, `proxy_server_request`, `timeout`, `user`, `tags`, `api_key`, `api_base`,
    `cache`, `custom_llm_provider`. A leaked key makes Anthropic answer 400 `Extra inputs are not permitted`."""
    body: Final = _sent(
        gateway,
        litellm_metadata={"tags": [marker("tag")], "user": "synthetic-end-user"},
        timeout=30,
        **NO_CACHE,
    )
    leaked: Final = {
        "litellm_metadata",
        "litellm_params",
        "litellm_call_id",
        "litellm_logging_obj",
        "proxy_server_request",
        "timeout",
        "user",
        "tags",
        "api_key",
        "api_base",
        "cache",
        "custom_llm_provider",
    }.intersection(body)
    assert leaked == set(), body


def test_a_combined_request_keeps_every_field_together(gateway: Gateway) -> None:
    """One request that sets system blocks, a tool, a named tool_choice, metadata, stop_sequences,
    top_k and temperature at once must arrive with all seven fields equal to what was sent. This
    guards against a rewrite that handles each field alone but drops one when several are present."""
    fields: Final[dict[str, JsonValue]] = {
        "system": [{"type": "text", "text": "policy", "cache_control": {"type": "ephemeral"}}],
        "tools": [_TOOL],
        "tool_choice": {"type": "tool", "name": "get_weather"},
        "metadata": {"user_id": f"synthetic-user-{marker()}"},
        "stop_sequences": ["END"],
        "top_k": 5,
        "temperature": 0.2,
    }
    body: Final = _sent(gateway, **fields)
    assert {name: body.get(name) for name in fields} == fields


def test_the_no_cache_control_reaches_the_provider_with_the_same_body_as_a_plain_call(gateway: Gateway) -> None:
    """`cache: {"no-cache": true}` is a proxy control that also takes the call off the Python-only
    response-cache path. Two requests that differ only in that control must produce the same provider
    body apart from their prompts, and the control itself must not be forwarded."""
    with deployment(gateway, echo_message) as rig:
        assert rig.post(rig.body("plain", system="s", tools=[_TOOL])).status_code == 200
        plain: Final = rig.one_body()
        assert rig.post(rig.body("plain", system="s", tools=[_TOOL], **NO_CACHE)).status_code == 200
        uncached: Final = rig.one_body()
        assert uncached == plain
        assert "cache" not in uncached
