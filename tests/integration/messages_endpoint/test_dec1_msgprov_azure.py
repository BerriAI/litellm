"""`POST /v1/messages` served by Azure: Foundry's Anthropic endpoint (`azure_ai/claude-*`) and Azure OpenAI (`azure/`).

An `azure_ai/` deployment whose model id contains `claude` has a native Messages config,
`AzureAnthropicMessagesConfig` (`litellm/llms/azure_ai/anthropic/messages_transformation.py`). The body
goes to `{root}/anthropic/v1/messages` (`azure_anthropic_messages_url` accepts a bare resource root, a
`/anthropic` base or a full `/v1/messages` URL), authenticated with `x-api-key` (the Azure `api-key`
header is renamed), with `anthropic-version: 2023-06-01` unless the caller sent one, betas filtered
through the `azure_ai` row, `system` rebuilt as blocks and `cache_control.scope` removed. Unlike Bedrock
and Vertex, the body keeps `model` and `stream`. Count tokens goes to
`/anthropic/v1/messages/count_tokens` with both `x-api-key` and `api-key`.

An `azure/<deployment>` model has no Messages config, so the request is adapted to chat completions
and sent to `/openai/deployments/<deployment>/chat/completions?api-version=<version>` with the
`api-key` header. Both run on Python today (`litellm/rust_bridge/catalog.py` opts in only the
`anthropic` provider).
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
from integration.messages_endpoint.dec1_msgprov_support import (
    AZURE_AI_BACKEND,
    AZURE_AI_KEY,
    AZURE_API_VERSION,
    AZURE_DEPLOYMENT,
    AZURE_KEY,
    INPUT_COST,
    OUTPUT_COST,
    PRICES,
    TOOL,
    anthropic_body_echo,
    anthropic_error_reply,
    at,
    body_of,
    chat_echo,
    chat_reply,
    chat_stream_reply,
    only_spend_row,
    openai_error_reply,
    path_of,
    provider_deployment,
    query_of,
    spend_of,
    unique_text,
)
from pydantic import JsonValue

_FOUNDRY_MESSAGES: Final = "/anthropic/v1/messages"
_AZURE_CHAT: Final = f"/openai/deployments/{AZURE_DEPLOYMENT}/chat/completions"
_BILLING_BLOCK: Final[dict[str, JsonValue]] = {
    "type": "text",
    "text": "x-anthropic-billing-header: cc_version=2.1.0; cc_entrypoint=cli",
}


def _foundry(
    gateway: Gateway,
    respond: Callable[[Request], Reply] = anthropic_body_echo,
    *,
    base_suffix: str = "",
    **params: JsonValue,
) -> AbstractContextManager[Rig]:
    return provider_deployment(
        gateway, respond, model=f"azure_ai/{AZURE_AI_BACKEND}", base_suffix=base_suffix, api_key=AZURE_AI_KEY, **params
    )


_AZURE_ECHO: Final = chat_echo(AZURE_DEPLOYMENT)


def _azure(
    gateway: Gateway, respond: Callable[[Request], Reply] = _AZURE_ECHO, **params: JsonValue
) -> AbstractContextManager[Rig]:
    return provider_deployment(
        gateway,
        respond,
        model=f"azure/{AZURE_DEPLOYMENT}",
        api_key=AZURE_KEY,
        api_version=AZURE_API_VERSION,
        **params,
    )


# ---------------------------------------------------------------- Foundry: URL, method and auth


@pytest.mark.parametrize("base_suffix", ["", "/anthropic", "/anthropic/", "/anthropic/v1/messages"])
def test_every_foundry_api_base_shape_posts_to_anthropic_v1_messages(gateway: Gateway, base_suffix: str) -> None:
    """A resource root, a `/anthropic` base (with or without a trailing slash) and a full
    `/anthropic/v1/messages` URL all reach the same `POST /anthropic/v1/messages`, never a doubled
    `/anthropic/anthropic` or `/v1/messages/v1/messages`."""
    with _foundry(gateway, base_suffix=base_suffix) as rig:
        assert rig.post(rig.body()).status_code == 200
        call: Final = rig.one_call()
        assert (call.method, path_of(call)) == ("POST", _FOUNDRY_MESSAGES)


def test_foundry_authenticates_with_x_api_key_and_a_default_anthropic_version(gateway: Gateway) -> None:
    """The Foundry key goes in `x-api-key` (the Azure `api-key` header is renamed, not duplicated) and
    `anthropic-version: 2023-06-01` is set when the caller sent none. The proxy key never appears."""
    with _foundry(gateway) as rig:
        assert rig.post(rig.body()).status_code == 200
        call: Final = rig.one_call()
        assert call.headers["x-api-key"] == AZURE_AI_KEY
        assert "api-key" not in call.headers
        assert call.headers["anthropic-version"] == "2023-06-01"
        assert all(gateway.key not in value for value in call.headers.values()), dict(call.headers)


def test_a_caller_anthropic_version_is_forwarded_to_foundry(gateway: Gateway) -> None:
    """The proxy forwards the caller's `anthropic-version` to `azure_ai` on the native Messages route
    (`litellm_pre_call_utils.py`), and the config only fills the header when it is absent, so a caller
    value of `2026-10-21` reaches Foundry unchanged. UNVERIFIED: header forwarding to `azure_ai`."""
    with _foundry(gateway) as rig:
        assert rig.post(rig.body(), headers={"anthropic-version": "2026-10-21"}).status_code == 200
        assert rig.one_call().headers["anthropic-version"] == "2026-10-21"


# ---------------------------------------------------------------- Foundry: body translation


@pytest.mark.parametrize("stream", [False, True])
def test_the_foundry_body_keeps_model_and_stream(gateway: Gateway, stream: bool) -> None:
    """Foundry takes the deployment name in the body, so a plain request reaches it as exactly
    `{"model": "claude-sonnet-4-5", "max_tokens": 64, "messages": [...], "stream": <flag>}`, with the
    backend model name, not the proxy alias, and no `anthropic_version` body member."""
    with _foundry(gateway) as rig:
        body: Final = rig.body(stream=stream)
        if stream:
            rig.stream(body)
        else:
            assert rig.post(body).status_code == 200
        assert rig.one_body() == {
            "model": AZURE_AI_BACKEND,
            "max_tokens": 64,
            "messages": body["messages"],
            "stream": stream,
        }


def test_a_foundry_string_system_prompt_becomes_one_text_block(gateway: Gateway) -> None:
    """`system: "<text>"` is rebuilt as `[{"type": "text", "text": "<text>"}]`."""
    with _foundry(gateway) as rig:
        assert rig.post(rig.body(system="You are terse.")).status_code == 200
        assert rig.one_body()["system"] == [{"type": "text", "text": "You are terse."}]


def test_the_foundry_billing_header_block_is_stripped(gateway: Gateway) -> None:
    """A Claude Code billing header system block is removed before Foundry sees the request."""
    keep: Final[dict[str, JsonValue]] = {"type": "text", "text": "Keep this instruction."}
    with _foundry(gateway) as rig:
        assert rig.post(rig.body(system=[_BILLING_BLOCK, keep])).status_code == 200
        assert rig.one_body()["system"] == [keep]


def test_foundry_cache_control_loses_scope_and_keeps_ttl(gateway: Gateway) -> None:
    """`cache_control: {"type": "ephemeral", "ttl": "1h", "scope": "global"}` reaches Foundry as
    `{"type": "ephemeral", "ttl": "1h"}`; Foundry supports only `type` and `ttl`."""
    content: Final[list[JsonValue]] = [
        {"type": "text", "text": unique_text(), "cache_control": {"type": "ephemeral", "ttl": "1h", "scope": "global"}}
    ]
    with _foundry(gateway) as rig:
        assert rig.post(rig.body(messages=[{"role": "user", "content": content}])).status_code == 200
        assert at(rig.one_body(), "messages", 0, "content", 0, "cache_control") == {"type": "ephemeral", "ttl": "1h"}


def test_foundry_drops_a_beta_its_row_maps_to_null(gateway: Gateway) -> None:
    """`anthropic-beta: context-management-2025-06-27,advisor-tool-2026-03-01` reaches Foundry as
    `context-management-2025-06-27`; the `azure_ai` row maps the advisor beta to null."""
    with _foundry(gateway) as rig:
        headers: Final = {"anthropic-beta": "context-management-2025-06-27,advisor-tool-2026-03-01"}
        assert rig.post(rig.body(), headers=headers).status_code == 200
        assert rig.one_call().headers["anthropic-beta"] == "context-management-2025-06-27"


def test_foundry_forwards_native_fields_unchanged(gateway: Gateway) -> None:
    """`tools`, `tool_choice`, `thinking`, `stop_sequences` and `metadata` need no translation and
    arrive exactly as sent."""
    fields: Final[dict[str, JsonValue]] = {
        "tools": [TOOL],
        "tool_choice": {"type": "auto"},
        "thinking": {"type": "enabled", "budget_tokens": 1024},
        "stop_sequences": ["###"],
        "metadata": {"user_id": "synthetic-user-01"},
    }
    with _foundry(gateway) as rig:
        assert rig.post(rig.body(max_tokens=2048, **fields)).status_code == 200
        body: Final = rig.one_body()
        assert {key: body[key] for key in fields} == fields


# ---------------------------------------------------------------- Foundry: response, stream, errors, spend


def test_the_foundry_message_comes_back_with_the_public_model_name(gateway: Gateway) -> None:
    """Foundry's message is returned with `model` replaced by the proxy model name and the content
    and usage unchanged."""
    with _foundry(gateway) as rig:
        payload: Final = JSON_OBJECT.validate_json(rig.post(rig.body()).content)
        assert payload["model"] == rig.model
        assert payload["content"] == [{"type": "text", "text": "pong"}]
        usage: Final = object_value(payload["usage"])
        assert (usage["input_tokens"], usage["output_tokens"]) == (11, 7)


def test_the_foundry_stream_is_relayed_in_order(gateway: Gateway) -> None:
    """Foundry's SSE stream reaches the caller as the same six events in order, restamped with the
    proxy model name."""
    with _foundry(gateway) as rig:
        _response, text = rig.stream(rig.body(stream=True))
        events: Final = sse_events(text)
        assert [name for name, _data in events] == [
            "message_start",
            "content_block_start",
            "content_block_delta",
            "content_block_stop",
            "message_delta",
            "message_stop",
        ], text
        assert object_value(events[0][1]["message"])["model"] == rig.model


@pytest.mark.parametrize(
    ("status", "anthropic_type"),
    [(400, "invalid_request_error"), (401, "authentication_error"), (429, "rate_limit_error")],
)
def test_a_foundry_error_keeps_its_status_and_type(gateway: Gateway, status: int, anthropic_type: str) -> None:
    """Foundry errors arrive in Anthropic's envelope and leave the proxy with the same status and the
    matching Anthropic type, with the provider text in the message."""
    message: Final = f"synthetic foundry {status}"
    with _foundry(gateway, lambda _r: anthropic_error_reply(status, anthropic_type, message)) as rig:
        response: Final = rig.post(rig.body())
        assert response.status_code == status, response.text
        error: Final = anthropic_error(response)
        assert error["type"] == anthropic_type
        assert message in str(error["message"])


def test_the_foundry_spend_row_names_azure_ai(gateway: Gateway) -> None:
    """One Foundry call writes one `anthropic_messages` row with `custom_llm_provider = "azure_ai"` and
    `spend = 11 * 0.001 + 7 * 0.002 = 0.025`."""
    with _foundry(gateway, **PRICES) as rig:
        assert rig.post(rig.body()).status_code == 200
        row: Final = only_spend_row(rig)
        assert (row["call_type"], row["custom_llm_provider"]) == ("anthropic_messages", "azure_ai")
        assert spend_of(row) == pytest.approx(11 * INPUT_COST + 7 * OUTPUT_COST)


def _foundry_count(request: Request) -> Reply:
    if path_of(request).endswith("/count_tokens"):
        return Reply(body=json.dumps({"input_tokens": 37}).encode())
    return anthropic_body_echo(request)


def test_foundry_count_tokens_uses_the_count_path_and_both_key_headers(gateway: Gateway) -> None:
    """`POST /v1/messages/count_tokens` on a Foundry deployment calls
    `/anthropic/v1/messages/count_tokens` with the key in both `x-api-key` and `api-key` and
    `token-counting-2024-11-01` in `anthropic-beta`, and answers `input_tokens: 37`."""
    with _foundry(gateway, _foundry_count) as rig:
        body: Final[dict[str, JsonValue]] = {"model": rig.model, "messages": [{"role": "user", "content": unique_text()}]}
        response: Final = rig.post(body, path="/v1/messages/count_tokens")
        assert response.status_code == 200, response.text
        assert JSON_OBJECT.validate_json(response.content)["input_tokens"] == 37
        call: Final = rig.one_call()
        assert path_of(call) == "/anthropic/v1/messages/count_tokens"
        assert (call.headers["x-api-key"], call.headers["api-key"]) == (AZURE_AI_KEY, AZURE_AI_KEY)
        assert "token-counting-2024-11-01" in call.headers["anthropic-beta"].split(",")


# ---------------------------------------------------------------- Azure OpenAI through the chat bridge


def test_an_azure_deployment_posts_chat_completions_with_the_api_version(gateway: Gateway) -> None:
    """An `azure/gpt-5.4` deployment receives `POST /openai/deployments/gpt-5.4/chat/completions` with
    query `api-version=2025-04-01-preview` and the key in the `api-key` header."""
    with _azure(gateway) as rig:
        response: Final = rig.post(rig.body())
        assert response.status_code == 200, response.text
        call: Final = rig.one_call()
        assert (call.method, path_of(call), query_of(call)) == ("POST", _AZURE_CHAT, f"api-version={AZURE_API_VERSION}")
        assert call.headers["api-key"] == AZURE_KEY


def test_the_azure_chat_body_has_the_system_message_first_and_max_completion_tokens(gateway: Gateway) -> None:
    """The Anthropic `system` string becomes the first chat message with role `system`, the user turn
    follows, and `max_tokens: 64` is sent as `max_completion_tokens: 64` for a GPT-5 deployment,
    which rejects `max_tokens`."""
    prompt: Final = unique_text()
    with _azure(gateway) as rig:
        assert rig.post(rig.body(prompt, system="You are terse.")).status_code == 200
        body: Final = rig.one_body()
        assert body["messages"] == [
            {"role": "system", "content": "You are terse."},
            {"role": "user", "content": prompt},
        ]
        assert body["max_completion_tokens"] == 64 and "max_tokens" not in body


def test_the_azure_chat_tools_and_tool_choice_are_openai_functions(gateway: Gateway) -> None:
    """An Anthropic tool becomes `{"type": "function", "function": {"name", "description",
    "parameters"}}` and `tool_choice: {"type": "tool", "name": "get_weather"}` becomes
    `{"type": "function", "function": {"name": "get_weather"}}`."""
    with _azure(gateway) as rig:
        tool_choice: Final[dict[str, JsonValue]] = {"type": "tool", "name": "get_weather"}
        assert rig.post(rig.body(tools=[TOOL], tool_choice=tool_choice)).status_code == 200
        body: Final = rig.one_body()
        function: Final = object_value(at(body, "tools", 0, "function"))
        assert (at(body, "tools", 0, "type"), function["name"], function["parameters"]) == (
            "function",
            "get_weather",
            TOOL["input_schema"],
        )
        assert body["tool_choice"] == {"type": "function", "function": {"name": "get_weather"}}


def test_an_azure_chat_answer_becomes_an_anthropic_message(gateway: Gateway) -> None:
    """A chat completion with content `pong`, finish `stop` and 11 and 7 tokens becomes an Anthropic
    message with one text block, `stop_reason: end_turn` and the same usage."""
    with _azure(gateway, lambda _r: chat_reply("chatcmpl-msgprov", model=AZURE_DEPLOYMENT)) as rig:
        payload: Final = JSON_OBJECT.validate_json(rig.post(rig.body()).content)
        assert payload["content"] == [{"type": "text", "text": "pong"}]
        assert payload["stop_reason"] == "end_turn"
        usage: Final = object_value(payload["usage"])
        assert (usage["input_tokens"], usage["output_tokens"]) == (11, 7)


def test_an_azure_length_finish_becomes_max_tokens(gateway: Gateway) -> None:
    """Finish reason `length` maps to Anthropic `stop_reason: max_tokens`, so a caller can tell a cut
    answer from a finished one."""
    reply: Final = chat_reply("chatcmpl-msgprov", model=AZURE_DEPLOYMENT, finish_reason="length")
    with _azure(gateway, lambda _r: reply) as rig:
        assert JSON_OBJECT.validate_json(rig.post(rig.body()).content)["stop_reason"] == "max_tokens"


def test_an_azure_chat_stream_becomes_anthropic_sse(gateway: Gateway) -> None:
    """Azure chat chunks become an Anthropic stream that starts with `message_start`, ends with
    `message_stop` and carries `hello stream` across its text deltas."""
    reply: Final = chat_stream_reply("chatcmpl-msgprov", model=AZURE_DEPLOYMENT)
    with _azure(gateway, lambda _r: reply) as rig:
        _response, text = rig.stream(rig.body(stream=True))
        events: Final = sse_events(text)
        assert (events[0][0], events[-1][0]) == ("message_start", "message_stop"), text
        deltas: Final = [object_value(data["delta"]) for name, data in events if name == "content_block_delta"]
        assert "".join(str(delta.get("text", "")) for delta in deltas) == "hello stream"


def test_an_azure_429_is_an_anthropic_rate_limit_error(gateway: Gateway) -> None:
    """An Azure OpenAI 429 is answered with HTTP 429 and `error.type == "rate_limit_error"` in the
    Anthropic envelope, not the OpenAI one."""
    reply: Final = openai_error_reply(429, "Requests to the deployment have exceeded the rate limit.", "rate_limit_error")
    with _azure(gateway, lambda _r: reply) as rig:
        response: Final = rig.post(rig.body())
        assert response.status_code == 429, response.text
        assert anthropic_error(response)["type"] == "rate_limit_error"


def test_the_azure_spend_row_names_azure(gateway: Gateway) -> None:
    """One bridged Azure call writes one `anthropic_messages` row with `custom_llm_provider = "azure"`
    and `spend = 0.025` from the deployment prices."""
    with _azure(gateway, **PRICES) as rig:
        assert rig.post(rig.body()).status_code == 200
        row: Final = only_spend_row(rig)
        assert (row["call_type"], row["custom_llm_provider"]) == ("anthropic_messages", "azure")
        assert spend_of(row) == pytest.approx(11 * INPUT_COST + 7 * OUTPUT_COST)


def test_the_azure_chat_body_never_carries_anthropic_only_fields(gateway: Gateway) -> None:
    """The bridged body must not leak Anthropic members Azure OpenAI rejects: no `system`,
    `metadata`, `anthropic_version` or `max_tokens` at the top level. `stop_sequences` is left out
    because GPT-5 deployments list `stop` as unsupported; see the index's Not covered section."""
    with _azure(gateway) as rig:
        assert rig.post(rig.body(system="sys", metadata={"user_id": "synthetic-user-01"})).status_code == 200
        body: Final = body_of(rig.one_call())
        assert not {"system", "metadata", "anthropic_version", "max_tokens"} & set(body), sorted(body)
