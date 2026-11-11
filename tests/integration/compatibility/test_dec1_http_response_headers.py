"""The response headers the proxy stamps on LLM calls, on success, on provider errors and on streams.

Callers and dashboards read `x-litellm-*` headers to correlate a response with its spend row
(`x-litellm-call-id`), to see which deployment served it (`x-litellm-model-id`, `x-litellm-model-api-base`), what
it cost (`x-litellm-response-cost`, `x-litellm-key-spend`) and how long the proxy took. Provider headers are
relayed under an `llm_provider-` prefix, with the OpenAI rate-limit headers also relayed under their own names.
All of it is assembled in `ProxyBaseLLMRequestProcessing.get_custom_headers` (`common_request_processing.py:2379`)
and the provider-header helpers in `litellm_core_utils`. A reimplemented front door has to produce the same names
and values, and must keep provider-supplied headers from overriding the proxy's own.
"""

from __future__ import annotations

import uuid
from collections.abc import Mapping
from functools import partial
from typing import Final

import pytest
from integration._support.client import JSON_OBJECT, Gateway, list_value, object_value
from integration._support.wire import Reply, Request
from integration.compatibility.dec1_http_support import (
    CHAT,
    chat_body,
    chat_completion,
    front_door,
    provider_error,
    scripted_reply,
    spend_rows,
)
from pydantic import JsonValue

INPUT_PRICE: Final = 0.001
OUTPUT_PRICE: Final = 0.002
SCRIPTED_COST: Final = 3 * INPUT_PRICE + 2 * OUTPUT_PRICE
PROVIDER_HEADERS: Final[Mapping[str, str]] = {
    "x-request-id": "req-dec1-upstream",
    "openai-processing-ms": "17",
    "x-ratelimit-limit-requests": "500",
    "x-ratelimit-remaining-requests": "499",
    "x-ratelimit-limit-tokens": "30000",
    "x-ratelimit-remaining-tokens": "29990",
}
HOSTILE_HEADERS: Final[Mapping[str, str]] = {
    "set-cookie": "upstream_session=1",
    "x-frame-options": "ALLOWALL",
    "access-control-allow-origin": "https://evil.example",
    "x-litellm-call-id": "spoofed-call-id",
    "x-litellm-response-cost": "999",
}


def _with_headers(headers: Mapping[str, str], request: Request) -> Reply:
    return Reply(body=chat_completion(), headers=headers)


def _failing(status: int, headers: Mapping[str, str], request: Request) -> Reply:
    failure: Final = provider_error(status)
    return Reply(status=failure.status, body=failure.body, headers=headers)


def _model_id(gateway: Gateway, name: str) -> JsonValue:
    entries: Final = tuple(object_value(entry) for entry in list_value(gateway.get("/model/info")["data"]))
    (entry,) = tuple(entry for entry in entries if entry["model_name"] == name)
    return object_value(entry["model_info"])["id"]


def test_a_json_success_is_application_json_with_an_exact_content_length(gateway: Gateway) -> None:
    """A chat model is registered and called without streaming. The 200 response must be labelled
    `content-type: application/json` and its `content-length` must equal the number of body bytes, so the body is
    not chunked. Clients on HTTP/1.0-style stacks and some gateways need the length."""
    with front_door(gateway) as door:
        response: Final = gateway.request("POST", CHAT, chat_body(door.chat_model()))
        assert response.status_code == 200, response.text
        assert response.headers["content-type"] == "application/json"
        assert int(response.headers["content-length"]) == len(response.content)
        assert "transfer-encoding" not in response.headers


def test_each_call_gets_its_own_call_id_in_uuid_form(gateway: Gateway) -> None:
    """A chat model is called twice. Each 200 response must carry an `x-litellm-call-id` that parses as a UUID,
    and the two values must differ (`resolve_litellm_call_id`, `common_request_processing.py:378`)."""
    with front_door(gateway) as door:
        model: Final = door.chat_model()
        identifiers: Final = tuple(
            gateway.request("POST", CHAT, chat_body(model)).headers["x-litellm-call-id"] for _ in range(2)
        )
        assert [str(uuid.UUID(identifier)) for identifier in identifiers] == list(identifiers)
        assert identifiers[0] != identifiers[1]


def test_a_call_id_the_client_sends_is_adopted_and_echoed(gateway: Gateway) -> None:
    """A chat model is called with the request header `x-litellm-call-id: dec1-<random>`. The proxy must use that
    value as the call id and echo it in the response's `x-litellm-call-id`, which lets a client pick the id it
    will later look up. The value is read at `common_request_processing.py:2749`."""
    with front_door(gateway) as door:
        chosen: Final = f"dec1-{uuid.uuid4().hex}"
        response: Final = gateway.request(
            "POST", CHAT, chat_body(door.chat_model()), headers={"x-litellm-call-id": chosen}
        )
        assert response.status_code == 200, response.text
        assert response.headers["x-litellm-call-id"] == chosen


def test_a_client_call_id_longer_than_256_characters_is_replaced(gateway: Gateway) -> None:
    """A chat model is called with an `x-litellm-call-id` request header of 257 characters. That exceeds
    `MAX_LITELLM_CALL_ID_LENGTH` (256), so the proxy must ignore it and answer with a fresh UUID call id instead
    of storing an oversized identifier."""
    with front_door(gateway) as door:
        oversized: Final = "c" * 257
        response: Final = gateway.request(
            "POST", CHAT, chat_body(door.chat_model()), headers={"x-litellm-call-id": oversized}
        )
        assert response.status_code == 200, response.text
        assert response.headers["x-litellm-call-id"] != oversized
        assert str(uuid.UUID(response.headers["x-litellm-call-id"])) == response.headers["x-litellm-call-id"]


def test_the_call_id_header_is_the_call_id_on_the_spend_row(gateway: Gateway) -> None:
    """A fresh virtual key calls a chat model once. After the spend row for that key lands in
    `LiteLLM_SpendLogs`, its `litellm_call_id` column must equal the `x-litellm-call-id` response header. This is
    the join support teams use to go from a client log line to the stored request."""
    with front_door(gateway) as door:
        key: Final = door.scenario.key()
        response: Final = gateway.request("POST", CHAT, chat_body(door.chat_model()), key=key)
        assert response.status_code == 200, response.text
        (row,) = spend_rows(gateway, key)
        assert row["litellm_call_id"] == response.headers["x-litellm-call-id"]


def test_model_id_and_api_base_headers_name_the_deployment_that_served(gateway: Gateway) -> None:
    """A chat model is registered behind a recording provider and called once. `x-litellm-model-id` must equal
    the deployment id that `/model/info` reports for that model, and `x-litellm-model-api-base` must equal the
    deployment's `api_base` (the provider URL ending in `/v1`)."""
    with front_door(gateway) as door:
        model: Final = door.chat_model()
        response: Final = gateway.request("POST", CHAT, chat_body(model))
        assert response.status_code == 200, response.text
        assert response.headers["x-litellm-model-id"] == _model_id(gateway, model)
        assert response.headers["x-litellm-model-api-base"] == f"{door.upstream.url}/v1"


def test_a_query_string_on_the_api_base_is_not_exposed_in_the_header(gateway: Gateway) -> None:
    """A chat model is registered whose `api_base` carries a query string with a synthetic secret. After one call
    `x-litellm-model-api-base` must stop at the `?`: the proxy strips query parameters before exposing the base
    (`common_request_processing.py:2429`) because they often carry credentials. The peer answers a chat
    completion on every path, since how the provider client joins `/chat/completions` onto a base that carries a
    query string is not what this test is about."""
    with front_door(gateway, partial(_with_headers, {})) as door:
        model: Final = door.chat_model(api_base=f"{door.upstream.url}/v1?sig=dec1-synthetic-secret")
        response: Final = gateway.request("POST", CHAT, chat_body(model))
        assert response.status_code == 200, response.text
        assert response.headers["x-litellm-model-api-base"] == f"{door.upstream.url}/v1"


def test_response_cost_header_is_the_token_usage_times_the_deployment_prices(gateway: Gateway) -> None:
    """A chat model priced at 0.001 per input token and 0.002 per output token is called once. The provider
    reports 3 prompt and 2 completion tokens, so `x-litellm-response-cost` must be 3 * 0.001 + 2 * 0.002 = 0.007."""
    with front_door(gateway) as door:
        model: Final = door.chat_model(input_cost_per_token=INPUT_PRICE, output_cost_per_token=OUTPUT_PRICE)
        response: Final = gateway.request("POST", CHAT, chat_body(model))
        assert response.status_code == 200, response.text
        assert float(response.headers["x-litellm-response-cost"]) == pytest.approx(SCRIPTED_COST)


def test_key_spend_header_includes_the_cost_of_the_call_it_is_on(gateway: Gateway) -> None:
    """A fresh virtual key with no spend calls a priced chat model once. `x-litellm-key-spend` on that response
    must already include this call's cost, 0.007, not the pre-call spend of 0
    (`common_request_processing.py:2408`). Clients that meter themselves from this header rely on that."""
    with front_door(gateway) as door:
        model: Final = door.chat_model(input_cost_per_token=INPUT_PRICE, output_cost_per_token=OUTPUT_PRICE)
        key: Final = door.scenario.key()
        response: Final = gateway.request("POST", CHAT, chat_body(model), key=key)
        assert response.status_code == 200, response.text
        assert float(response.headers["x-litellm-key-spend"]) == pytest.approx(SCRIPTED_COST)


def test_key_limit_headers_report_the_limits_set_on_the_key(gateway: Gateway) -> None:
    """A virtual key is created with `rpm_limit: 7`, `tpm_limit: 900` and `max_budget: 5.0` and calls a chat
    model. The response must carry `x-litellm-key-rpm-limit: 7`, `x-litellm-key-tpm-limit: 900` and
    `x-litellm-key-max-budget: 5.0`."""
    with front_door(gateway) as door:
        key: Final = door.scenario.key(rpm_limit=7, tpm_limit=900, max_budget=5.0)
        response: Final = gateway.request("POST", CHAT, chat_body(door.chat_model()), key=key)
        assert response.status_code == 200, response.text
        limits: Final = {
            name: response.headers.get(name)
            for name in ("x-litellm-key-rpm-limit", "x-litellm-key-tpm-limit", "x-litellm-key-max-budget")
        }
        assert limits == {
            "x-litellm-key-rpm-limit": "7",
            "x-litellm-key-tpm-limit": "900",
            "x-litellm-key-max-budget": "5.0",
        }


def test_key_limit_headers_are_omitted_when_the_key_has_no_limits(gateway: Gateway) -> None:
    """A virtual key with no rate limits and no budget calls a chat model. The three key-limit headers must be
    absent rather than sent as the text "None": `get_custom_headers` drops empty and "None" values
    (`common_request_processing.py:2397`)."""
    with front_door(gateway) as door:
        key: Final = door.scenario.key()
        response: Final = gateway.request("POST", CHAT, chat_body(door.chat_model()), key=key)
        assert response.status_code == 200, response.text
        present: Final = [
            name
            for name in ("x-litellm-key-rpm-limit", "x-litellm-key-tpm-limit", "x-litellm-key-max-budget")
            if name in response.headers
        ]
        assert present == []
        assert "None" not in [value for name, value in response.headers.items() if name.startswith("x-litellm-")]


def test_duration_headers_are_non_negative_milliseconds(gateway: Gateway) -> None:
    """A chat model is called once. `x-litellm-response-duration-ms` and `x-litellm-overhead-duration-ms` must
    both be present and parse as non-negative numbers, and the overhead must not exceed the whole duration.
    UNVERIFIED: both are read from the response's hidden timing values, assumed set for a plain chat call."""
    with front_door(gateway) as door:
        response: Final = gateway.request("POST", CHAT, chat_body(door.chat_model()))
        assert response.status_code == 200, response.text
        duration: Final = float(response.headers["x-litellm-response-duration-ms"])
        overhead: Final = float(response.headers["x-litellm-overhead-duration-ms"])
        assert 0 <= overhead <= duration


def test_provider_headers_are_relayed_under_the_llm_provider_prefix(gateway: Gateway) -> None:
    """The provider answers a chat call with `x-request-id: req-dec1-upstream` and `openai-processing-ms: 17`.
    The proxy response must carry `llm_provider-x-request-id: req-dec1-upstream` and
    `llm_provider-openai-processing-ms: 17`: every provider header is forwarded with the `llm_provider-` prefix
    (`core_helpers.py:729`, `get_headers.py:64`)."""
    with front_door(gateway, partial(_with_headers, PROVIDER_HEADERS)) as door:
        response: Final = gateway.request("POST", CHAT, chat_body(door.chat_model()))
        assert response.status_code == 200, response.text
        assert response.headers["llm_provider-x-request-id"] == "req-dec1-upstream"
        assert response.headers["llm_provider-openai-processing-ms"] == "17"


def test_provider_rate_limit_headers_are_relayed_under_their_own_names(gateway: Gateway) -> None:
    """The provider answers a chat call with the four OpenAI `x-ratelimit-*` headers. A key without its own
    limits makes the call. The proxy response must carry the same four headers, unprefixed and with the
    provider's values, so OpenAI SDK code that reads remaining quota keeps working. UNVERIFIED: assumes no proxy
    rate limiter overwrites them for a key that has no limits."""
    with front_door(gateway, partial(_with_headers, PROVIDER_HEADERS)) as door:
        response: Final = gateway.request("POST", CHAT, chat_body(door.chat_model()))
        assert response.status_code == 200, response.text
        relayed: Final = {name: response.headers.get(name) for name in PROVIDER_HEADERS if "ratelimit" in name}
        assert relayed == {name: value for name, value in PROVIDER_HEADERS.items() if "ratelimit" in name}


def test_a_provider_cannot_set_cookies_or_browser_policy_on_the_proxy_response(gateway: Gateway) -> None:
    """The provider answers a chat call with `set-cookie`, `x-frame-options: ALLOWALL` and
    `access-control-allow-origin: https://evil.example`. The proxy response must carry no `set-cookie`, must keep
    `x-frame-options: DENY`, and must carry no `access-control-allow-origin` (the caller sent no Origin).
    Provider headers only ever appear under the `llm_provider-` prefix."""
    with front_door(gateway, partial(_with_headers, HOSTILE_HEADERS)) as door:
        response: Final = gateway.request("POST", CHAT, chat_body(door.chat_model()))
        assert response.status_code == 200, response.text
        assert "set-cookie" not in response.headers
        assert response.headers["x-frame-options"] == "DENY"
        assert "access-control-allow-origin" not in response.headers


def test_a_provider_cannot_override_the_proxys_own_litellm_headers(gateway: Gateway) -> None:
    """The provider answers a priced chat call with `x-litellm-call-id: spoofed-call-id` and
    `x-litellm-response-cost: 999`. The proxy response must carry its own UUID call id and the computed cost
    0.007: headers that arrive from a provider are never treated as LiteLLM-owned (`core_helpers.py:718`)."""
    with front_door(gateway, partial(_with_headers, HOSTILE_HEADERS)) as door:
        model: Final = door.chat_model(input_cost_per_token=INPUT_PRICE, output_cost_per_token=OUTPUT_PRICE)
        response: Final = gateway.request("POST", CHAT, chat_body(model))
        assert response.status_code == 200, response.text
        assert "spoofed-call-id" not in response.headers.get_list("x-litellm-call-id")
        assert [float(value) for value in response.headers.get_list("x-litellm-response-cost")] == [
            pytest.approx(SCRIPTED_COST)
        ]


@pytest.mark.parametrize("status", (400, 404, 429, 500, 503))
def test_a_provider_error_response_still_carries_the_call_id(gateway: Gateway, status: int) -> None:
    """The provider fails a chat call with the given status and retries are turned off for the request. The
    proxy must answer with the same status, `content-type: application/json` and an `x-litellm-call-id`, so a
    failed request can be traced exactly like a successful one (`handle_llm_api_exception`,
    `common_request_processing.py:4643`)."""
    with front_door(gateway, partial(_failing, status, {})) as door:
        response: Final = gateway.request(
            "POST", CHAT, chat_body(door.chat_model()), headers={"x-litellm-num-retries": "0"}
        )
        assert response.status_code == status, response.text
        assert response.headers["content-type"] == "application/json"
        assert response.headers["x-litellm-call-id"]


@pytest.mark.parametrize("status", (400, 404, 429, 500, 503))
def test_a_provider_error_response_still_carries_the_call_id_and_version(gateway: Gateway, status: int) -> None:
    """The provider fails a chat call with the given status and retries are turned off for the request. The
    proxy must answer with the same status, `content-type: application/json`, an `x-litellm-call-id` and an
    `x-litellm-version`, so a failed request can be traced exactly like a successful one
    (`handle_llm_api_exception`, `common_request_processing.py:4643`).

    BUG: `chat_completion` in `proxy_server.py` calls `handle_llm_api_exception` without `version=version`,
    unlike `/completions` and `/embeddings` next to it, so `get_custom_headers` gets `version=None` and the
    header is dropped on every chat error. The call id half is covered by the test above."""
    pytest.skip("BUG: /chat/completions error responses carry no x-litellm-version header")
    with front_door(gateway, partial(_failing, status, {})) as door:
        response: Final = gateway.request(
            "POST", CHAT, chat_body(door.chat_model()), headers={"x-litellm-num-retries": "0"}
        )
        assert response.status_code == status, response.text
        assert response.headers["content-type"] == "application/json"
        assert response.headers["x-litellm-call-id"]
        assert response.headers["x-litellm-version"]


def test_a_provider_error_relays_the_providers_request_id(gateway: Gateway) -> None:
    """The provider fails a chat call with 400 and an `x-request-id: req-dec1-upstream` header. The proxy's 400
    must carry `llm_provider-x-request-id: req-dec1-upstream`, so the operator can quote the provider's id in a
    support ticket. The headers are taken from the provider response on the failure
    (`common_request_processing.py:4662`)."""
    with front_door(gateway, partial(_failing, 400, PROVIDER_HEADERS)) as door:
        response: Final = gateway.request(
            "POST", CHAT, chat_body(door.chat_model()), headers={"x-litellm-num-retries": "0"}
        )
        assert response.status_code == 400, response.text
        assert response.headers["llm_provider-x-request-id"] == "req-dec1-upstream"


def test_a_provider_error_cannot_replace_the_proxys_content_type_or_cookies(gateway: Gateway) -> None:
    """The provider fails a chat call with 400 and the headers `set-cookie` and `x-frame-options: ALLOWALL`. The
    proxy's error response must be `application/json`, carry no `set-cookie`, and keep `x-frame-options: DENY`:
    framing and browser-security headers from a provider are filtered out of error responses
    (`UNSAFE_PROXY_RESPONSE_HEADERS`, `constants.py:2270`)."""
    with front_door(gateway, partial(_failing, 400, HOSTILE_HEADERS)) as door:
        response: Final = gateway.request(
            "POST", CHAT, chat_body(door.chat_model()), headers={"x-litellm-num-retries": "0"}
        )
        assert response.status_code == 400, response.text
        assert response.headers["content-type"] == "application/json"
        assert "set-cookie" not in response.headers
        assert response.headers["x-frame-options"] == "DENY"


def test_a_streamed_response_carries_the_sse_and_litellm_headers_before_any_data(gateway: Gateway) -> None:
    """A chat model is called with `stream: true`. The response headers must be a 200 with `content-type`
    starting `text/event-stream`, `cache-control: no-cache`, `x-accel-buffering: no`, an `x-litellm-call-id`, an
    `x-litellm-model-id`, and no `content-length`. The two anti-buffering headers are what keep nginx and ALBs
    from holding the stream (`_sse_stream_headers`, `common_request_processing.py:1435`)."""
    with front_door(gateway) as door:
        body: Final = chat_body(door.chat_model(), stream=True)
        headers: Final = {"Authorization": f"Bearer {gateway.key}"}
        with gateway.client.stream("POST", CHAT, json=body, headers=headers) as response:
            assert response.status_code == 200
            assert response.headers["content-type"].startswith("text/event-stream")
            assert response.headers["cache-control"] == "no-cache"
            assert response.headers["x-accel-buffering"] == "no"
            assert response.headers["x-litellm-call-id"]
            assert response.headers["x-litellm-model-id"]
            assert "content-length" not in response.headers
            assert "data: [DONE]" in response.read().decode()


def test_an_identical_request_is_served_from_cache_with_the_same_cache_key_header(gateway: Gateway) -> None:
    """A chat model is called twice with a byte-identical body and no cache opt-out. The shared proxy has the
    Redis response cache on, so the provider must be called once, both responses must be 200, and the second must
    carry an `x-litellm-cache-key` header. UNVERIFIED: whether the first (miss) response also carries the key."""
    with front_door(gateway) as door:
        body: Final[dict[str, JsonValue]] = {
            "model": door.chat_model(),
            "messages": [{"role": "user", "content": f"dec1 cache {uuid.uuid4().hex}"}],
        }
        first: Final = gateway.request("POST", CHAT, body)
        second: Final = gateway.request("POST", CHAT, body)
        assert (first.status_code, second.status_code) == (200, 200), (first.text, second.text)
        assert second.headers["x-litellm-cache-key"]
        assert len(door.calls()) == 1
        answers: Final = tuple(JSON_OBJECT.validate_json(response.content)["choices"] for response in (first, second))
        assert answers[0] == answers[1]


def test_embeddings_and_chat_responses_share_the_core_litellm_headers(gateway: Gateway) -> None:
    """An embedding model and a chat model are each called once. Both 200 responses must carry
    `x-litellm-call-id`, `x-litellm-model-id`, `x-litellm-version` and `x-litellm-response-cost`. Every OpenAI
    surface goes through the same header builder, so a surface moved to a new implementation must not lose one."""
    core: Final = ("x-litellm-call-id", "x-litellm-model-id", "x-litellm-version", "x-litellm-response-cost")
    with front_door(gateway, scripted_reply) as door:
        chat: Final = gateway.request("POST", CHAT, chat_body(door.chat_model()))
        embedded: Final = gateway.request(
            "POST",
            "/v1/embeddings",
            {"model": door.embedding_model(), "input": f"dec1 {uuid.uuid4().hex}", "cache": {"no-cache": True}},
        )
        assert (chat.status_code, embedded.status_code) == (200, 200), (chat.text, embedded.text)
        missing: Final = {
            surface: [name for name in core if name not in response.headers]
            for surface, response in (("chat", chat), ("embeddings", embedded))
        }
        assert missing == {"chat": [], "embeddings": []}
