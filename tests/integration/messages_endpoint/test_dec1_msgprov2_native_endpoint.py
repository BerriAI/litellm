"""`POST /v1/messages` on Fireworks and Baseten deployments that opt into the provider's own Messages endpoint.

Fireworks, Baseten and other OpenAI-compatible hosts can also serve the Anthropic Messages API. A
deployment opts in with `model_info.supported_endpoints: ["/v1/messages"]`; the proxy then picks
`OpenAILikeAnthropicMessagesConfig` (`litellm/llms/openai_like/messages/transformation.py`) instead of the
chat completions bridge, and forwards the caller's Anthropic body essentially unchanged. The contract:

The URL is `{api_base}/v1/messages`, whether `api_base` is the host, ends in `/v1`, has a trailing slash or
already ends in `/v1/messages`. The key goes in `Authorization: Bearer <key>` (never `x-api-key`),
`anthropic-version: 2023-06-01` and `content-type: application/json` are filled in, and the caller's own
`anthropic-version` and `anthropic-beta` headers are not forwarded, because the proxy scopes those headers
to `anthropic`, `bedrock`, `bedrock_mantle`, `vertex_ai`, and natively to `azure_ai` and `databricks`
(`litellm/proxy/litellm_pre_call_utils.py` `_ANTHROPIC_API_HEADER_PROVIDERS`). Betas that the body itself
implies (`output_format`, `context_management`) are still derived and sent. `cache_control` hints are
reduced to `{"type": ...}` unless the deployment sets `model_info.cache_control_ttl: true`.

The answer, the stream and the errors are Anthropic's own and are relayed. The SpendLogs row is an
`anthropic_messages` row named after the provider and priced from Anthropic usage, cache tokens included.
This target is `MESSAGES_PASSTHROUGH`, which `litellm/rust_bridge/catalog.py` keeps PYTHON_ONLY today; a Rust
port of these providers has to keep both edges.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from contextlib import AbstractContextManager
from typing import Final

import pytest
from integration._support.client import Gateway, object_value
from integration._support.wire import Reply, Request
from integration.messages_endpoint.dec1_messages_support import JSON_OBJECT, Rig, anthropic_error, sse_events
from integration.messages_endpoint.dec1_msgprov2_support import (
    CONTEXT_MANAGEMENT_BETA,
    DEFAULT_ANTHROPIC_VERSION,
    PASSTHROUGH_INFO,
    STRUCTURED_OUTPUTS_BETA,
    anthropic_answer,
    anthropic_message,
    anthropic_stream_with_usage,
    event_names,
)
from integration.messages_endpoint.dec1_msgprov_support import (
    BASETEN_BACKEND,
    BASETEN_KEY,
    CACHE_CREATION_COST,
    CACHE_READ_COST,
    FIREWORKS_BACKEND,
    FIREWORKS_KEY,
    INPUT_COST,
    OUTPUT_COST,
    PRICES,
    TOOL,
    anthropic_error_reply,
    only_spend_row,
    path_of,
    provider_deployment,
    spend_of,
    unique_text,
)
from pydantic import JsonValue

_BASETEN_ANSWER: Final = anthropic_answer(BASETEN_BACKEND)
_FIREWORKS_ANSWER: Final = anthropic_answer(FIREWORKS_BACKEND)


def _baseten(
    gateway: Gateway,
    respond: Callable[[Request], Reply] = _BASETEN_ANSWER,
    *,
    base_suffix: str = "",
    model_info: dict[str, JsonValue] | None = None,
    **params: JsonValue,
) -> AbstractContextManager[Rig]:
    return provider_deployment(
        gateway,
        respond,
        model=f"baseten/{BASETEN_BACKEND}",
        base_suffix=base_suffix,
        model_info=dict(PASSTHROUGH_INFO) if model_info is None else model_info,
        api_key=BASETEN_KEY,
        **params,
    )


def _fireworks(
    gateway: Gateway,
    respond: Callable[[Request], Reply] = _FIREWORKS_ANSWER,
    *,
    model: str = f"fireworks_ai/{FIREWORKS_BACKEND}",
    model_info: dict[str, JsonValue] | None = None,
    **params: JsonValue,
) -> AbstractContextManager[Rig]:
    return provider_deployment(
        gateway,
        respond,
        model=model,
        model_info=dict(PASSTHROUGH_INFO) if model_info is None else model_info,
        api_key=FIREWORKS_KEY,
        **params,
    )


# ---------------------------------------------------------------- URL and transport choice


@pytest.mark.parametrize(
    "suffix", ["", "/", "/v1", "/v1/", "/v1/messages"], ids=["host", "slash", "v1", "v1-slash", "full"]
)
def test_every_api_base_shape_reaches_v1_messages_once(gateway: Gateway, suffix: str) -> None:
    """A Baseten deployment with the `/v1/messages` opt-in and an `api_base` of the bare host, the host with a
    trailing slash, `/v1`, `/v1/` or the full `/v1/messages` URL is always called at exactly `POST /v1/messages`:
    `get_complete_url` strips one `/v1`, never doubles it, and keeps a full URL as is. A port that joins naively
    would post to `/v1/v1/messages` or `/v1/messages/v1/messages`."""
    with _baseten(gateway, base_suffix=suffix) as rig:
        response: Final = rig.post(rig.body())
        assert response.status_code == 200, response.text
        call: Final = rig.one_call()
        assert (call.method, path_of(call)) == ("POST", "/v1/messages")


def test_the_fireworks_opt_in_posts_messages_and_never_chat_completions(gateway: Gateway) -> None:
    """A Fireworks deployment with `supported_endpoints: ["/v1/messages"]` gets one `POST /v1/messages` and no
    `/chat/completions` call: the opt-in replaces the bridge rather than adding a second request."""
    with _fireworks(gateway) as rig:
        assert rig.post(rig.body()).status_code == 200
        assert [path_of(call) for call in rig.calls()] == ["/v1/messages"]


def test_a_supported_endpoints_list_without_messages_keeps_the_chat_bridge(gateway: Gateway) -> None:
    """`supported_endpoints: ["/v1/chat/completions"]` is not the opt-in: only a list containing the literal
    `/v1/messages` is (`_deployment_passes_through_anthropic_messages`), so the call goes through the chat
    bridge to `POST /chat/completions`."""
    from integration.messages_endpoint.dec1_msgprov_support import chat_echo

    with _baseten(
        gateway, chat_echo(BASETEN_BACKEND), model_info={"supported_endpoints": ["/v1/chat/completions"]}
    ) as rig:
        assert rig.post(rig.body()).status_code == 200
        assert path_of(rig.one_call()) == "/chat/completions"


# ---------------------------------------------------------------- Headers


def test_the_key_is_sent_as_a_bearer_token_and_never_as_x_api_key(gateway: Gateway) -> None:
    """The passthrough authenticates like an OpenAI-compatible host: `Authorization: Bearer synthetic-baseten-key`
    and no `x-api-key` header (`validate_anthropic_messages_environment`). Sending the key twice, or as
    `x-api-key` like first-party Anthropic, would be rejected by hosts that only read one of them."""
    with _baseten(gateway) as rig:
        assert rig.post(rig.body()).status_code == 200
        headers: Final = rig.one_call().headers
        assert headers["authorization"] == f"Bearer {BASETEN_KEY}"
        assert "x-api-key" not in headers


def test_anthropic_version_and_content_type_are_filled_in(gateway: Gateway) -> None:
    """A caller that sends no `anthropic-version` still reaches Fireworks with `anthropic-version: 2023-06-01`
    and `content-type: application/json`, which strict Messages implementations require."""
    with _fireworks(gateway) as rig:
        assert rig.post(rig.body()).status_code == 200
        headers: Final = rig.one_call().headers
        assert headers["anthropic-version"] == DEFAULT_ANTHROPIC_VERSION
        assert headers["content-type"].startswith("application/json")


def test_a_caller_anthropic_version_is_not_forwarded_to_the_passthrough(gateway: Gateway) -> None:
    """The caller sends `anthropic-version: 2026-10-21`. Fireworks is not in the providers that receive the
    caller's Anthropic headers (`_ANTHROPIC_API_HEADER_PROVIDERS` and the native-only Azure AI and Databricks
    list), so the upstream sees the default `2023-06-01`. A port that forwards every `anthropic-*` header would
    change the API version the host applies."""
    with _fireworks(gateway) as rig:
        assert rig.post(rig.body(), headers={"anthropic-version": "2026-10-21"}).status_code == 200
        assert rig.one_call().headers["anthropic-version"] == DEFAULT_ANTHROPIC_VERSION


def test_a_caller_anthropic_beta_is_not_forwarded_to_the_passthrough(gateway: Gateway) -> None:
    """The caller sends `anthropic-beta: interleaved-thinking-2025-05-14` and a body that implies no beta. Baseten
    receives no `anthropic-beta` header at all, for the same header scoping reason as the version. Pinned
    current behaviour: Anthropic-compatible hosts that need the beta never get it."""
    with _baseten(gateway) as rig:
        assert rig.post(rig.body(), headers={"anthropic-beta": "interleaved-thinking-2025-05-14"}).status_code == 200
        assert "anthropic-beta" not in rig.one_call().headers


def test_a_beta_the_body_implies_is_derived_and_sent(gateway: Gateway) -> None:
    """A body with `output_format: {"type": "json_schema", ...}` reaches Fireworks with
    `anthropic-beta: structured-outputs-2025-11-13`, derived from the body by `_update_headers_with_anthropic_beta`
    and not filtered (`should_filter_anthropic_beta_headers` is False for this config), and with `output_format`
    unchanged in the body."""
    output_format: Final[dict[str, JsonValue]] = {
        "type": "json_schema",
        "schema": {"type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"]},
    }
    with _fireworks(gateway) as rig:
        assert rig.post(rig.body(output_format=output_format)).status_code == 200
        call: Final = rig.one_call()
        assert call.headers["anthropic-beta"] == STRUCTURED_OUTPUTS_BETA
        assert JSON_OBJECT.validate_json(call.body)["output_format"] == output_format


def test_context_management_implies_its_beta_on_the_passthrough(gateway: Gateway) -> None:
    """A body with `context_management: {"edits": [{"type": "clear_tool_uses_20250919"}]}` reaches Baseten with
    `anthropic-beta: context-management-2025-06-27` (`context_management_betas`) and the edits in the body.
    UNVERIFIED: that the beta-gated field strip keeps `context_management` on this config."""
    edits: Final[dict[str, JsonValue]] = {"edits": [{"type": "clear_tool_uses_20250919"}]}
    with _baseten(gateway) as rig:
        assert rig.post(rig.body(context_management=edits)).status_code == 200
        call: Final = rig.one_call()
        assert CONTEXT_MANAGEMENT_BETA in call.headers["anthropic-beta"].split(",")
        assert JSON_OBJECT.validate_json(call.body)["context_management"] == edits


def test_the_callers_proxy_key_never_reaches_the_passthrough(gateway: Gateway) -> None:
    """The caller authenticates with the proxy master key as `Authorization: Bearer`. Fireworks receives its own
    deployment key in `authorization`, never the proxy key, in any header."""
    with _fireworks(gateway) as rig:
        assert rig.post(rig.body()).status_code == 200
        headers: Final = rig.one_call().headers
        assert headers["authorization"] == f"Bearer {FIREWORKS_KEY}"
        assert gateway.key not in json.dumps(dict(headers))


# ---------------------------------------------------------------- Body


def test_the_body_reaches_the_host_untranslated(gateway: Gateway) -> None:
    """System, tools, tool_choice, enabled thinking, stop_sequences, temperature and metadata arrive exactly as
    sent, next to `model: "zai-org/GLM-4.6"` (the backend name, not the proxy alias) and `max_tokens: 4096`.
    Nothing is turned into OpenAI shapes (`stop`, `functions`, `reasoning_effort`)."""
    prompt: Final = unique_text()
    fields: Final[dict[str, JsonValue]] = {
        "system": "You are terse.",
        "tools": [TOOL],
        "tool_choice": {"type": "auto"},
        "thinking": {"type": "enabled", "budget_tokens": 2048},
        "stop_sequences": ["###"],
        "temperature": 1,
        "metadata": {"user_id": "synthetic-user-02"},
    }
    with _baseten(gateway) as rig:
        assert rig.post(rig.body(prompt, max_tokens=4096, **fields)).status_code == 200
        body: Final = rig.one_body()
        assert (body["model"], body["max_tokens"]) == (BASETEN_BACKEND, 4096)
        assert body["messages"] == [{"role": "user", "content": prompt}]
        assert {name: body[name] for name in fields} == fields
        assert not {"stop", "reasoning_effort", "user", "functions"} & set(body), sorted(body)


@pytest.mark.parametrize("stream", [False, True], ids=["unary", "stream"])
def test_the_stream_flag_is_carried_in_the_body(gateway: Gateway, stream: bool) -> None:
    """The passthrough keeps one URL for both modes, so the caller's `stream` flag must reach the host in the
    body: `stream: true` for a streamed call, and no `stream: true` for a unary one."""
    with _fireworks(gateway) as rig:
        if stream:
            rig.stream(rig.body(stream=True))
        else:
            assert rig.post(rig.body()).status_code == 200
        assert (rig.one_body().get("stream") is True) is stream


def test_a_short_fireworks_name_is_sent_as_written_on_the_passthrough(gateway: Gateway) -> None:
    """`fireworks_ai/glm-5p3` with the opt-in reaches Fireworks' Messages endpoint as `model: "glm-5p3"`: the
    `accounts/fireworks/models/` resolution belongs to the chat config, which this route does not use.
    UNVERIFIED: no Fireworks-specific model resolution was found on `OpenAILikeAnthropicMessagesConfig`; if it
    is added, this expectation flips to the account path."""
    with _fireworks(gateway, model="fireworks_ai/glm-5p3") as rig:
        assert rig.post(rig.body()).status_code == 200
        assert rig.one_body()["model"] == "glm-5p3"


def test_cache_control_is_reduced_to_its_type(gateway: Gateway) -> None:
    """A text block with `cache_control: {"type": "ephemeral", "ttl": "1h"}` and a system block with
    `cache_control: {"ttl": "5m"}` reach Baseten as `{"type": "ephemeral"}` both times
    (`normalize_cache_control_in_anthropic_payload`), because strict Messages implementations reject `ttl`."""
    content: Final[list[JsonValue]] = [
        {"type": "text", "text": unique_text(), "cache_control": {"type": "ephemeral", "ttl": "1h"}}
    ]
    system: Final[list[JsonValue]] = [{"type": "text", "text": "You are terse.", "cache_control": {"ttl": "5m"}}]
    with _baseten(gateway) as rig:
        assert rig.post(rig.body(messages=[{"role": "user", "content": content}], system=system)).status_code == 200
        body: Final = rig.one_body()
        message_block: Final = object_value(object_value(body["messages"][0])["content"][0])  # pyright: ignore[reportIndexIssue]  # nested JSON
        system_block: Final = object_value(body["system"][0])  # pyright: ignore[reportIndexIssue]  # nested JSON
        assert message_block["cache_control"] == {"type": "ephemeral"}
        assert system_block["cache_control"] == {"type": "ephemeral"}


def test_cache_control_ttl_is_kept_when_the_deployment_declares_it(gateway: Gateway) -> None:
    """With `model_info: {"supported_endpoints": ["/v1/messages"], "cache_control_ttl": true}` the same block
    reaches the host with `cache_control: {"type": "ephemeral", "ttl": "1h"}` unchanged."""
    content: Final[list[JsonValue]] = [
        {"type": "text", "text": unique_text(), "cache_control": {"type": "ephemeral", "ttl": "1h"}}
    ]
    info: Final[dict[str, JsonValue]] = {**PASSTHROUGH_INFO, "cache_control_ttl": True}
    with _baseten(gateway, model_info=info) as rig:
        assert rig.post(rig.body(messages=[{"role": "user", "content": content}])).status_code == 200
        block: Final = object_value(object_value(rig.one_body()["messages"][0])["content"][0])  # pyright: ignore[reportIndexIssue]  # nested JSON
        assert block["cache_control"] == {"type": "ephemeral", "ttl": "1h"}


# ---------------------------------------------------------------- Answer, stream and errors


def test_the_hosts_message_is_relayed_unchanged(gateway: Gateway) -> None:
    """Fireworks answers a message with a text block and a `tool_use` block, `stop_reason: tool_use` and usage
    11 and 7. The caller receives the same `content`, `stop_reason`, `usage` and `id`, with no OpenAI fields
    (`choices`, `object`) and no translation round trip."""
    content: Final[list[JsonValue]] = [
        {"type": "text", "text": "checking"},
        {"type": "tool_use", "id": "toolu_msgprov2", "name": "get_weather", "input": {"city": "Porto"}},
    ]
    reply: Final = anthropic_message(FIREWORKS_BACKEND, content=content, stop_reason="tool_use")
    sent_id: Final = JSON_OBJECT.validate_json(reply.body)["id"]
    with _fireworks(gateway, lambda _r: reply) as rig:
        response: Final = rig.post(rig.body(tools=[TOOL]))
        assert response.status_code == 200, response.text
        payload: Final = JSON_OBJECT.validate_json(response.content)
        assert (payload["id"], payload["content"], payload["stop_reason"]) == (sent_id, content, "tool_use")
        assert object_value(payload["usage"])["input_tokens"] == 11
        assert not {"choices", "object"} & set(payload)


def test_the_hosts_stream_is_relayed_as_the_same_six_events(gateway: Gateway) -> None:
    """Baseten streams the six Anthropic events. The caller gets `message_start`, `content_block_start`,
    `content_block_delta`, `content_block_stop`, `message_delta`, `message_stop` in that order with the text
    `hello stream`, and an `text/event-stream` content type."""
    with _baseten(gateway) as rig:
        response, text = rig.stream(rig.body(stream=True))
        assert response.status_code == 200, text
        assert response.headers["content-type"].startswith("text/event-stream")
        events: Final = sse_events(text)
        assert event_names(events) == (
            "message_start",
            "content_block_start",
            "content_block_delta",
            "content_block_stop",
            "message_delta",
            "message_stop",
        ), text
        delta: Final = object_value(events[2][1]["delta"])
        assert delta == {"type": "text_delta", "text": "hello stream"}


@pytest.mark.parametrize(
    ("status", "error_type"),
    [
        (400, "invalid_request_error"),
        (401, "authentication_error"),
        (429, "rate_limit_error"),
        (529, "overloaded_error"),
    ],
)
def test_a_host_error_keeps_its_status_and_anthropic_type(gateway: Gateway, status: int, error_type: str) -> None:
    """An Anthropic-shaped error from the host leaves the proxy with the same HTTP status and the same Anthropic
    `error.type`, with the host's message text inside `error.message`. UNVERIFIED for 529: that the passthrough
    error path keeps 529 rather than mapping it to 503 as the chat bridge does."""
    message: Final = f"synthetic passthrough {status}"
    with _fireworks(gateway, lambda _r: anthropic_error_reply(status, error_type, message)) as rig:
        response: Final = rig.post(rig.body())
        assert response.status_code == status, response.text
        error: Final = anthropic_error(response)
        assert error["type"] == error_type
        assert message in str(error["message"])


# ---------------------------------------------------------------- Spend and count tokens


@pytest.mark.parametrize(("provider", "factory"), [("baseten", _baseten), ("fireworks_ai", _fireworks)])
def test_the_spend_row_prices_anthropic_usage_with_cache_tokens(
    gateway: Gateway, provider: str, factory: Callable[..., AbstractContextManager[Rig]]
) -> None:
    """The host reports `input_tokens: 10`, `cache_read_input_tokens: 4`, `cache_creation_input_tokens: 2` and
    `output_tokens: 5`. One `anthropic_messages` row is written with `custom_llm_provider` naming the provider
    and `spend = 10 * 0.001 + 4 * 0.0001 + 2 * 0.003 + 5 * 0.002 = 0.0264` from the deployment prices."""
    usage: Final[dict[str, JsonValue]] = {
        "input_tokens": 10,
        "output_tokens": 5,
        "cache_read_input_tokens": 4,
        "cache_creation_input_tokens": 2,
    }
    backend: Final = BASETEN_BACKEND if provider == "baseten" else FIREWORKS_BACKEND
    reply: Final = anthropic_message(backend, usage=usage)
    with factory(gateway, lambda _r: reply, **PRICES) as rig:
        assert rig.post(rig.body()).status_code == 200
        row: Final = only_spend_row(rig)
        assert (row["call_type"], row["custom_llm_provider"]) == ("anthropic_messages", provider)
        assert spend_of(row) == pytest.approx(
            10 * INPUT_COST + 4 * CACHE_READ_COST + 2 * CACHE_CREATION_COST + 5 * OUTPUT_COST
        )


def test_a_streamed_passthrough_call_is_billed_from_message_start_and_message_delta(gateway: Gateway) -> None:
    """A streamed Fireworks call whose `message_start` reports 20 input and 6 cache read tokens and whose
    `message_delta` reports 9 output tokens writes one row with `prompt_tokens` counting all input and spend
    `20 * 0.001 + 6 * 0.0001 + 9 * 0.002 = 0.0386`. UNVERIFIED: whether `prompt_tokens` is 26 (input plus
    cache) or 20 on this route; the spend is what is pinned."""
    reply: Final = anthropic_stream_with_usage(FIREWORKS_BACKEND, input_tokens=20, output_tokens=9, cache_read=6)
    with _fireworks(gateway, lambda _r: reply, **PRICES) as rig:
        response, text = rig.stream(rig.body(stream=True))
        assert response.status_code == 200, text
        row: Final = only_spend_row(rig)
        assert spend_of(row) == pytest.approx(20 * INPUT_COST + 6 * CACHE_READ_COST + 9 * OUTPUT_COST)
        assert row["completion_tokens"] == 9


def test_count_tokens_on_a_passthrough_deployment_is_a_local_estimate(gateway: Gateway) -> None:
    """`POST /v1/messages/count_tokens` for a Baseten passthrough deployment is answered with HTTP 200, a
    positive `input_tokens` and `litellm_count_source: "local_estimate"` (`count_source_fields`), and the host's
    `/v1/messages/count_tokens` is never called: no Baseten token counter exists, and the passthrough config does
    not forward count requests."""
    with _baseten(gateway) as rig:
        body: Final[dict[str, JsonValue]] = {
            "model": rig.model,
            "messages": [{"role": "user", "content": unique_text()}],
        }
        response: Final = rig.post(body, path="/v1/messages/count_tokens")
        assert response.status_code == 200, response.text
        payload: Final = JSON_OBJECT.validate_json(response.content)
        assert isinstance(payload["input_tokens"], int) and payload["input_tokens"] > 0
        assert payload["litellm_count_source"] == "local_estimate"
        assert rig.calls() == ()


def test_the_passthrough_writes_a_failure_row_for_a_host_error(gateway: Gateway) -> None:
    """A Baseten 400 on the passthrough writes one SpendLogs row for the model group with `status: "failure"`
    and spend 0, so failed Messages calls stay visible per provider."""
    with _baseten(
        gateway, lambda _r: anthropic_error_reply(400, "invalid_request_error", "synthetic bad request"), **PRICES
    ) as rig:
        assert rig.post(rig.body()).status_code == 400
        row: Final = only_spend_row(rig)
        assert row["status"] == "failure"
        assert spend_of(row) == 0
