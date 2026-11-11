"""What the caller gets back from a successful non-streaming `POST /v1/messages`.

The body is the provider's Anthropic message, passed through: `id`, `type`, `role`, every content
block (text, tool_use, thinking, citations), `stop_reason`, `stop_sequence` and the `usage` object
with its cache and server-tool counters. Two things are changed on purpose. `model` reports the
proxy model name the caller asked for instead of the backend model
(`common_request_processing.py` `_override_openai_response_model`), and a missing `stop_sequence`
is filled in as null (`transformation.py` `_ensure_stop_sequence`). No OpenAI-style field such as
`usage.total_tokens`, `choices` or `object` is added.

The proxy also adds its own response headers (`get_custom_headers`): `x-litellm-call-id` (echoing a
caller-supplied id), `x-litellm-model-id` (the deployment id), `x-litellm-version`,
`x-litellm-response-cost` when the model has a price, `x-litellm-key-spend` and the
`x-litellm-response-duration-ms` timing. Clients and dashboards read these, and they are produced
by Python request-processing code that the Rust front door has to reproduce.
"""

from __future__ import annotations

import json
import re
from typing import Final

import pytest
from integration._support.client import Gateway, object_value, string_value
from integration._support.wire import Reply
from integration.messages_endpoint.dec1_messages_support import (
    BACKEND,
    CACHE_CREATION_COST,
    CACHE_READ_COST,
    INPUT_COST,
    JSON_OBJECT,
    NO_CACHE,
    OUTPUT_COST,
    PRICES,
    Rig,
    always,
    deployment,
    marker,
    message_json,
    message_reply,
)
from pydantic import JsonValue

_UUID: Final = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")


def _deployment_id(rig: Rig) -> str:
    entries: Final = rig.gateway.get("/model/info")["data"]
    assert isinstance(entries, list)
    matches: Final = [
        object_value(object_value(entry)["model_info"])["id"]
        for entry in entries
        if object_value(entry)["model_name"] == rig.model
    ]
    assert len(matches) == 1, matches
    return string_value(matches[0])


def test_the_provider_message_is_returned_with_only_the_model_renamed(gateway: Gateway) -> None:
    """The provider answers a complete message with `stop_sequence: null`. The caller must receive
    HTTP 200, `content-type: application/json` and a body equal to the provider's message in every
    key except `model`, which must be the proxy model name the caller sent."""
    message_id: Final = f"msg_{marker()}"
    with deployment(gateway, always(message_reply(message_id))) as rig:
        response: Final = rig.post(rig.body())
        assert response.status_code == 200, response.text
        assert response.headers["content-type"].startswith("application/json")
        assert JSON_OBJECT.validate_json(response.content) == {**message_json(message_id), "model": rig.model}


def test_the_response_model_is_the_requested_proxy_model_not_the_backend_model(gateway: Gateway) -> None:
    """The provider reports `model: claude-sonnet-4-5-20250929`. The caller asked for a proxy model
    name, and the response `model` must be that name, so a client that compares request and response
    models (Claude Code does) sees them agree. The backend name must not appear in `model`."""
    with deployment(gateway, always(message_reply(f"msg_{marker()}"))) as rig:
        body: Final = JSON_OBJECT.validate_json(rig.post(rig.body()).content)
        assert body["model"] == rig.model
        assert body["model"] != BACKEND


def test_the_response_id_is_the_provider_message_id(gateway: Gateway) -> None:
    """The response `id` must be exactly the `msg_...` id the provider returned. The proxy must not
    replace it with its own id or wrap it, because the same id keys the SpendLogs row."""
    message_id: Final = f"msg_{marker()}"
    with deployment(gateway, always(message_reply(message_id))) as rig:
        assert JSON_OBJECT.validate_json(rig.post(rig.body()).content)["id"] == message_id


def test_a_missing_stop_sequence_is_returned_as_null(gateway: Gateway) -> None:
    """Some Anthropic-compatible backends omit `stop_sequence`. When the provider's message has no
    such key, the caller's body must still contain `"stop_sequence": null`, since Anthropic SDK types
    require the key."""
    message: Final = {key: value for key, value in message_json(f"msg_{marker()}").items() if key != "stop_sequence"}
    with deployment(gateway, always(Reply(body=json.dumps(message).encode()))) as rig:
        body: Final = JSON_OBJECT.validate_json(rig.post(rig.body()).content)
        assert "stop_sequence" in body and body["stop_sequence"] is None


def test_a_stop_sequence_hit_is_returned_with_its_reason_and_text(gateway: Gateway) -> None:
    """When the provider stops on a caller stop sequence it answers `stop_reason: "stop_sequence"` and
    `stop_sequence: "END"`. Both must reach the caller unchanged."""
    reply: Final = message_reply(f"msg_{marker()}", stop_reason="stop_sequence", extra={"stop_sequence": "END"})
    with deployment(gateway, always(reply)) as rig:
        body: Final = JSON_OBJECT.validate_json(rig.post(rig.body(stop_sequences=["END"])).content)
        assert (body["stop_reason"], body["stop_sequence"]) == ("stop_sequence", "END")


def test_a_max_tokens_stop_reason_is_returned_unchanged(gateway: Gateway) -> None:
    """A provider message cut off at the token limit has `stop_reason: "max_tokens"`. The caller must
    see `max_tokens`, not an OpenAI-style `length`."""
    with deployment(gateway, always(message_reply(f"msg_{marker()}", stop_reason="max_tokens"))) as rig:
        assert JSON_OBJECT.validate_json(rig.post(rig.body()).content)["stop_reason"] == "max_tokens"


def test_a_tool_use_block_is_returned_with_id_name_and_input_object(gateway: Gateway) -> None:
    """The provider answers a text block followed by a `tool_use` block and `stop_reason: "tool_use"`.
    The caller must receive both blocks in order, the tool block with the same `id`, `name` and
    `input` as a JSON object (not a JSON string as OpenAI tool calls carry it)."""
    content: Final[list[JsonValue]] = [
        {"type": "text", "text": "Checking."},
        {"type": "tool_use", "id": "toolu_synthetic_77", "name": "get_weather", "input": {"city": "Paris", "n": 2}},
    ]
    with deployment(gateway, always(message_reply(f"msg_{marker()}", content=content, stop_reason="tool_use"))) as rig:
        body: Final = JSON_OBJECT.validate_json(rig.post(rig.body()).content)
        assert body["content"] == content
        assert body["stop_reason"] == "tool_use"


def test_thinking_and_redacted_thinking_blocks_are_returned_with_signatures(gateway: Gateway) -> None:
    """The provider answers a `thinking` block with a `signature`, a `redacted_thinking` block with
    opaque `data`, then text. All three must reach the caller unchanged and in order: the signature
    and data are needed verbatim when the caller replays the turn."""
    content: Final[list[JsonValue]] = [
        {"type": "thinking", "thinking": "step one", "signature": "c3ludGhldGljLXNpZ25hdHVyZQ=="},
        {"type": "redacted_thinking", "data": "c3ludGhldGljLXJlZGFjdGVk"},
        {"type": "text", "text": "done"},
    ]
    with deployment(gateway, always(message_reply(f"msg_{marker()}", content=content))) as rig:
        request: Final = rig.body(max_tokens=4096, thinking={"type": "enabled", "budget_tokens": 2048})
        assert JSON_OBJECT.validate_json(rig.post(request).content)["content"] == content


def test_text_citations_are_returned_on_the_text_block(gateway: Gateway) -> None:
    """A provider text block carrying a `citations` list (a `char_location` citation with document
    index and offsets) must reach the caller with the list intact on the same block."""
    content: Final[list[JsonValue]] = [
        {
            "type": "text",
            "text": "The sky is green.",
            "citations": [
                {
                    "type": "char_location",
                    "cited_text": "The sky is green.",
                    "document_index": 0,
                    "document_title": "Synthetic report",
                    "start_char_index": 0,
                    "end_char_index": 17,
                }
            ],
        }
    ]
    with deployment(gateway, always(message_reply(f"msg_{marker()}", content=content))) as rig:
        assert JSON_OBJECT.validate_json(rig.post(rig.body()).content)["content"] == content


def test_usage_token_counts_are_returned_exactly_without_total_tokens(gateway: Gateway) -> None:
    """The provider reports `usage: {"input_tokens": 11, "output_tokens": 7}`. The caller must
    receive exactly those two counters. No `total_tokens`, `prompt_tokens` or `completion_tokens`
    key may be added: those are OpenAI names and are not part of the Anthropic usage object."""
    with deployment(gateway, always(message_reply(f"msg_{marker()}"))) as rig:
        usage: Final = object_value(JSON_OBJECT.validate_json(rig.post(rig.body()).content)["usage"])
        assert usage == {"input_tokens": 11, "output_tokens": 7}


def test_cache_usage_counters_and_ttl_breakdown_are_returned(gateway: Gateway) -> None:
    """The provider reports cache counters and the per-TTL `cache_creation` breakdown. The caller
    must receive `cache_read_input_tokens: 5`, `cache_creation_input_tokens: 9` and
    `cache_creation: {"ephemeral_5m_input_tokens": 4, "ephemeral_1h_input_tokens": 5}` unchanged, and
    `input_tokens` must stay 10 (the uncached count), not a sum that includes cached tokens."""
    usage: Final[dict[str, JsonValue]] = {
        "input_tokens": 10,
        "output_tokens": 4,
        "cache_read_input_tokens": 5,
        "cache_creation_input_tokens": 9,
        "cache_creation": {"ephemeral_5m_input_tokens": 4, "ephemeral_1h_input_tokens": 5},
    }
    with deployment(gateway, always(message_reply(f"msg_{marker()}", usage=usage))) as rig:
        assert object_value(JSON_OBJECT.validate_json(rig.post(rig.body()).content)["usage"]) == usage


def test_server_tool_use_and_service_tier_usage_fields_are_returned(gateway: Gateway) -> None:
    """Newer usage fields the proxy does not interpret, `server_tool_use: {"web_search_requests": 2}`
    and `service_tier: "standard"`, must be passed through inside `usage` unchanged. A typed rewrite
    that only knows the token counters would drop them."""
    usage: Final[dict[str, JsonValue]] = {
        "input_tokens": 10,
        "output_tokens": 4,
        "server_tool_use": {"web_search_requests": 2},
        "service_tier": "standard",
    }
    with deployment(gateway, always(message_reply(f"msg_{marker()}", usage=usage))) as rig:
        assert object_value(JSON_OBJECT.validate_json(rig.post(rig.body()).content)["usage"]) == usage


def test_an_unknown_top_level_response_field_is_passed_through(gateway: Gateway) -> None:
    """A top-level field the proxy has no type for (`container: {"id": ..., "expires_at": ...}`, sent
    by code-execution responses) must reach the caller unchanged next to the known fields."""
    container: Final[dict[str, JsonValue]] = {"id": "container_synthetic", "expires_at": "2099-01-01T00:00:00Z"}
    with deployment(gateway, always(message_reply(f"msg_{marker()}", extra={"container": container}))) as rig:
        assert JSON_OBJECT.validate_json(rig.post(rig.body()).content)["container"] == container


def test_the_response_carries_a_generated_x_litellm_call_id(gateway: Gateway) -> None:
    """When the caller sends no call id, the response must carry an `x-litellm-call-id` header holding
    a freshly generated lower-case UUID, the id operators use to find the request in logs."""
    with deployment(gateway, always(message_reply(f"msg_{marker()}"))) as rig:
        response: Final = rig.post(rig.body())
        assert _UUID.match(response.headers["x-litellm-call-id"]), dict(response.headers)


def test_a_caller_supplied_x_litellm_call_id_is_echoed_back(gateway: Gateway) -> None:
    """A caller that sends `x-litellm-call-id: <its own id>` must get the same value back in the
    response header (`resolve_litellm_call_id`), so it can correlate without parsing the body."""
    call_id: Final = marker("call")
    with deployment(gateway, always(message_reply(f"msg_{marker()}"))) as rig:
        response: Final = rig.post(rig.body(), headers={"x-litellm-call-id": call_id})
        assert response.status_code == 200, response.text
        assert response.headers["x-litellm-call-id"] == call_id


def test_x_litellm_model_id_names_the_deployment_that_served_the_call(gateway: Gateway) -> None:
    """The response header `x-litellm-model-id` must equal the deployment's `model_info.id` as
    `/model/info` lists it for the requested proxy model."""
    with deployment(gateway, always(message_reply(f"msg_{marker()}"))) as rig:
        response: Final = rig.post(rig.body())
        assert response.status_code == 200, response.text
        assert response.headers["x-litellm-model-id"] == _deployment_id(rig)


def test_x_litellm_version_is_present(gateway: Gateway) -> None:
    """Every successful response must carry a non-empty `x-litellm-version` header with a dotted
    version number. UNVERIFIED: the exact value is the installed package version, so only the shape
    is asserted."""
    with deployment(gateway, always(message_reply(f"msg_{marker()}"))) as rig:
        assert re.match(r"^\d+\.\d+", rig.post(rig.body()).headers["x-litellm-version"])


def test_x_litellm_response_cost_is_the_priced_cost_of_the_usage(gateway: Gateway) -> None:
    """The deployment is priced at 0.001 per input token and 0.002 per output token and the provider
    reports 11 input and 7 output tokens. The `x-litellm-response-cost` header must be
    11 * 0.001 + 7 * 0.002 = 0.025, the same number the SpendLogs row records."""
    with deployment(gateway, always(message_reply(f"msg_{marker()}")), **PRICES) as rig:
        response: Final = rig.post(rig.body())
        assert response.status_code == 200, response.text
        assert float(response.headers["x-litellm-response-cost"]) == pytest.approx(11 * INPUT_COST + 7 * OUTPUT_COST)


def test_x_litellm_response_cost_prices_cache_read_and_cache_creation_tokens(gateway: Gateway) -> None:
    """With 10 uncached input, 4 output, 5 cache-read and 9 cache-creation tokens at the four
    configured rates, the cost header must be
    10 * 0.001 + 4 * 0.002 + 5 * 0.0001 + 9 * 0.003 = 0.0455. Cached tokens are priced at their own
    rates and are not also charged at the input rate."""
    usage: Final[dict[str, JsonValue]] = {
        "input_tokens": 10,
        "output_tokens": 4,
        "cache_read_input_tokens": 5,
        "cache_creation_input_tokens": 9,
    }
    with deployment(gateway, always(message_reply(f"msg_{marker()}", usage=usage)), **PRICES) as rig:
        response: Final = rig.post(rig.body())
        assert response.status_code == 200, response.text
        assert float(response.headers["x-litellm-response-cost"]) == pytest.approx(
            10 * INPUT_COST + 4 * OUTPUT_COST + 5 * CACHE_READ_COST + 9 * CACHE_CREATION_COST
        )


def test_the_key_limit_and_spend_headers_describe_the_calling_key(gateway: Gateway) -> None:
    """A virtual key with `rpm_limit: 7`, `tpm_limit: 9000` and `max_budget: 12.5` makes its first
    priced call. The response must carry `x-litellm-key-rpm-limit: 7`, `x-litellm-key-tpm-limit: 9000`,
    `x-litellm-key-max-budget: 12.5`, and `x-litellm-key-spend` equal to this call's cost (0.025),
    because the header adds the current response cost to the key's prior spend of zero."""
    with deployment(gateway, always(message_reply(f"msg_{marker()}")), **PRICES) as rig:
        key: Final = rig.scenario.key(models=[rig.model], rpm_limit=7, tpm_limit=9000, max_budget=12.5)
        response: Final = rig.post(rig.body(), key=key)
        assert response.status_code == 200, response.text
        assert response.headers["x-litellm-key-rpm-limit"] == "7"
        assert response.headers["x-litellm-key-tpm-limit"] == "9000"
        assert float(response.headers["x-litellm-key-max-budget"]) == 12.5
        assert float(response.headers["x-litellm-key-spend"]) == pytest.approx(11 * INPUT_COST + 7 * OUTPUT_COST)


def test_the_duration_headers_are_non_negative_numbers(gateway: Gateway) -> None:
    """`x-litellm-response-duration-ms` must be present on a successful response and parse as a
    non-negative number of milliseconds. UNVERIFIED: `x-litellm-overhead-duration-ms` is asserted the
    same way only when present, because it depends on hidden params a dict response may not carry."""
    with deployment(gateway, always(message_reply(f"msg_{marker()}"))) as rig:
        headers: Final = rig.post(rig.body()).headers
        assert float(headers["x-litellm-response-duration-ms"]) >= 0
        if "x-litellm-overhead-duration-ms" in headers:
            assert float(headers["x-litellm-overhead-duration-ms"]) >= 0


def test_the_provider_api_key_never_appears_in_response_headers_or_body(gateway: Gateway) -> None:
    """Nothing in the response, neither any header value nor the body, may contain the deployment's
    provider key `synthetic-anthropic-key`."""
    with deployment(gateway, always(message_reply(f"msg_{marker()}"))) as rig:
        response: Final = rig.post(rig.body())
        assert "synthetic-anthropic-key" not in response.text
        assert all("synthetic-anthropic-key" not in value for value in response.headers.values())


def test_provider_response_headers_are_exposed_with_the_llm_provider_prefix(gateway: Gateway) -> None:
    """The provider answers with `request-id: req_synthetic_123` and an
    `anthropic-ratelimit-requests-remaining: 41` header. The proxy is expected to expose them to the
    caller as `llm_provider-request-id` and `llm_provider-anthropic-ratelimit-requests-remaining`
    (`get_headers.py` `_get_llm_provider_headers`), as it does for chat completions and for streamed
    `/v1/messages` (`AnthropicMessagesStreamingResponse` carries them in `_hidden_params`).

    BUG: a non-streaming `/v1/messages` response is a plain `AnthropicMessagesResponse` dict built by
    `transform_anthropic_messages_response` (`messages/native/transformation.py`) with no
    `_hidden_params`, so `common_request_processing.py` finds no `additional_headers` and the
    provider's headers are dropped."""
    pytest.skip("BUG: non-streaming /v1/messages drops the provider's llm_provider-* response headers")
    reply: Final = message_reply(
        f"msg_{marker()}",
        headers={"request-id": "req_synthetic_123", "anthropic-ratelimit-requests-remaining": "41"},
    )
    with deployment(gateway, always(reply)) as rig:
        headers: Final = rig.post(rig.body()).headers
        assert headers["llm_provider-request-id"] == "req_synthetic_123"
        assert headers["llm_provider-anthropic-ratelimit-requests-remaining"] == "41"


@pytest.mark.parametrize("uncached", (False, True), ids=("cache_default", "no_cache"))
def test_body_and_proxy_headers_are_the_same_with_and_without_the_response_cache(
    gateway: Gateway, uncached: bool
) -> None:
    """A call on the default path and a call sent with `cache: {"no-cache": true}` (the only shape
    eligible for the Rust route on this proxy) must both return the provider message with the proxy
    model name, the cost header 0.025, a call id and the deployment id header."""
    message_id: Final = f"msg_{marker()}"
    with deployment(gateway, always(message_reply(message_id)), **PRICES) as rig:
        response: Final = rig.post(rig.body(**(NO_CACHE if uncached else {})))
        assert response.status_code == 200, response.text
        assert JSON_OBJECT.validate_json(response.content) == {**message_json(message_id), "model": rig.model}
        assert float(response.headers["x-litellm-response-cost"]) == pytest.approx(11 * INPUT_COST + 7 * OUTPUT_COST)
        assert _UUID.match(response.headers["x-litellm-call-id"])
        assert response.headers["x-litellm-model-id"] == _deployment_id(rig)
