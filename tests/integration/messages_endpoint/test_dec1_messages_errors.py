"""Error contract of `POST /v1/messages` for an Anthropic deployment.

Every failure on this route is answered in Anthropic's envelope,
`{"type": "error", "error": {"type": <anthropic type>, "message": <text>}}`, plus a top-level
`request_id` when the caller sent `x-request-id`. The HTTP status of a provider failure is the
provider's own status, and `error.type` is the Anthropic type for that status
(`anthropic_interface/exceptions` `ANTHROPIC_ERROR_TYPE_MAP`): 400 `invalid_request_error`,
401 `authentication_error`, 403 `permission_error`, 404 `not_found_error`, 429 `rate_limit_error`,
500 `api_error`, 529 `overloaded_error`, anything else `api_error`. Two statuses are remapped by the
Anthropic exception mapper (`exception_mapping_utils.py` `_map_anthropic_exception`): a provider 413
is answered as 400, and a provider timeout as 408.

Request mistakes the proxy catches itself (unknown model, missing `max_tokens` or `messages`,
malformed JSON) are answered 400 `invalid_request_error` without calling the provider.

Today the Python proxy puts the provider's error text INSIDE its own message
(`litellm.BadRequestError: AnthropicException - {...}`), while the Rust gateway's `Error::body`
passes a provider envelope through verbatim. The tests therefore pin status, `error.type` and that
the provider's message text is contained in `error.message`, which both shapes satisfy; the index
lists the difference so a human can decide which one is the contract.
"""

from __future__ import annotations

import time
from typing import Final

import pytest
from integration._support.client import Gateway
from integration._support.timing import assert_cut_short, timed
from integration._support.wire import Reply, Request
from integration.messages_endpoint.dec1_messages_support import (
    JSON_OBJECT,
    NO_CACHE,
    UPSTREAM_KEY,
    always,
    anthropic_error,
    deployment,
    echo_message,
    marker,
    message_reply,
    upstream_error,
)

_AUTH: Final = "Authorization"
_STALL_SECONDS: Final = 6.0


@pytest.mark.parametrize(
    ("status", "error_type"),
    (
        (400, "invalid_request_error"),
        (401, "authentication_error"),
        (403, "permission_error"),
        (404, "not_found_error"),
        (429, "rate_limit_error"),
        (500, "api_error"),
        (529, "overloaded_error"),
    ),
)
@pytest.mark.parametrize("stream", (False, True), ids=("non_stream", "stream"))
def test_a_provider_error_status_is_answered_with_the_same_status_and_anthropic_type(
    gateway: Gateway, status: int, error_type: str, stream: bool
) -> None:
    """The provider rejects the call with each of 400, 401, 403, 404, 429, 500 and 529 in Anthropic's
    envelope, for a non-streaming and a `stream: true` request. The caller must get the same HTTP
    status as a JSON body (never a 200 SSE stream), with `error.type` equal to the Anthropic type for
    that status and the provider's message text contained in `error.message`."""
    text: Final = f"synthetic provider refusal {marker()}"
    with deployment(gateway, always(upstream_error(status, error_type, text))) as rig:
        response: Final = rig.post(rig.body(stream=stream))
        assert response.status_code == status, response.text
        assert response.headers["content-type"].startswith("application/json")
        error: Final = anthropic_error(response)
        assert error["type"] == error_type
        assert text in str(error["message"])


@pytest.mark.parametrize("status", (400, 401, 403, 404))
def test_a_provider_client_error_is_not_retried(gateway: Gateway, status: int) -> None:
    """A provider 400, 401, 403 or 404 is the caller's problem and must reach the provider exactly
    once. A retry would double-bill a rejected request and delay the error."""
    with deployment(gateway, always(upstream_error(status, "invalid_request_error", "synthetic refusal"))) as rig:
        assert rig.post(rig.body()).status_code == status
        assert len(rig.calls()) == 1


@pytest.mark.parametrize("status", (429, 500, 529))
def test_a_provider_server_error_is_attempted_once_without_a_retry_budget(gateway: Gateway, status: int) -> None:
    """The deployment has no `num_retries` and the proxy config sets none. A provider 429, 500 or 529
    must therefore be attempted exactly once and answered with that status. A rewrite that brings a
    default retry count would silently multiply provider load during an outage."""
    with deployment(gateway, always(upstream_error(status, "api_error", "synthetic outage"))) as rig:
        assert rig.post(rig.body()).status_code == status
        assert len(rig.calls()) == 1


def test_the_error_body_has_exactly_the_anthropic_envelope_keys(gateway: Gateway) -> None:
    """For a provider 400 and a caller that sends no `x-request-id`, the body must have exactly the
    top-level keys `type` and `error`, with `type == "error"`. No OpenAI-style `code` or `param` key
    may appear at the top level, and `error` must hold string `type` and `message`."""
    with deployment(gateway, always(upstream_error(400, "invalid_request_error", "synthetic refusal"))) as rig:
        response: Final = rig.post(rig.body())
        body: Final = JSON_OBJECT.validate_json(response.content)
        assert set(body) == {"type", "error"}, body
        assert body["type"] == "error"
        anthropic_error(response)


def test_a_provider_error_echoes_the_caller_x_request_id_as_request_id(gateway: Gateway) -> None:
    """The caller sends `x-request-id` and the provider answers 400. The error body must carry
    `"request_id": "<the caller's value>"` at the top level, as Anthropic's own error bodies do, so
    SDK exceptions expose a request id."""
    request_id: Final = f"req_{marker()}"
    with deployment(gateway, always(upstream_error(400, "invalid_request_error", "synthetic refusal"))) as rig:
        response: Final = rig.post(rig.body(), headers={"x-request-id": request_id})
        assert response.status_code == 400, response.text
        assert JSON_OBJECT.validate_json(response.content)["request_id"] == request_id


def test_a_provider_error_response_carries_x_litellm_call_id(gateway: Gateway) -> None:
    """An error response must carry the `x-litellm-call-id` header just like a success, and must echo
    the caller's own `x-litellm-call-id` when one was sent, so a failed request can be found in logs."""
    call_id: Final = marker("call")
    with deployment(gateway, always(upstream_error(500, "api_error", "synthetic outage"))) as rig:
        response: Final = rig.post(rig.body(), headers={"x-litellm-call-id": call_id})
        assert response.status_code == 500, response.text
        assert response.headers["x-litellm-call-id"] == call_id


def test_a_provider_error_does_not_leak_the_deployment_key(gateway: Gateway) -> None:
    """The provider's 401 body quotes the key it was given. The caller must still not find the
    deployment key `synthetic-anthropic-key` in the response headers, and the proxy must not add it
    to the message on its own. UNVERIFIED: whether the proxy masks a key the provider itself echoed;
    this asserts only the headers and that the proxy key is absent."""
    with deployment(gateway, always(upstream_error(401, "authentication_error", "invalid x-api-key"))) as rig:
        response: Final = rig.post(rig.body())
        assert response.status_code == 401, response.text
        assert all(UPSTREAM_KEY not in value for value in response.headers.values())
        assert gateway.key not in response.text


def test_a_provider_413_is_answered_as_400_invalid_request_error(gateway: Gateway) -> None:
    """The Anthropic exception mapper treats a provider 413 like a 400
    (`status_code == 400 or status_code == 413` raises `BadRequestError`). The caller must get HTTP
    400 with `error.type == "invalid_request_error"`, not 413 `request_too_large`."""
    with deployment(gateway, always(upstream_error(413, "request_too_large", "synthetic too large"))) as rig:
        response: Final = rig.post(rig.body())
        assert response.status_code == 400, response.text
        assert anthropic_error(response)["type"] == "invalid_request_error"


@pytest.mark.parametrize("status", (502, 503))
def test_a_provider_502_or_503_keeps_its_status_and_is_typed_api_error(gateway: Gateway, status: int) -> None:
    """A provider 502 or 503 must be answered with the same status. Neither has an Anthropic error
    type of its own, so `error.type` must be `api_error`."""
    with deployment(gateway, always(upstream_error(status, "api_error", "synthetic bad gateway"))) as rig:
        response: Final = rig.post(rig.body())
        assert response.status_code == status, response.text
        assert anthropic_error(response)["type"] == "api_error"


def test_a_prompt_too_long_rejection_stays_a_400_invalid_request_error(gateway: Gateway) -> None:
    """Anthropic rejects an oversized prompt with 400 and `prompt is too long: ...`. The proxy maps
    that text to its context-window error class, which must still be answered as HTTP 400 with
    `error.type == "invalid_request_error"` and the original text in the message."""
    text: Final = "prompt is too long: 250000 tokens > 200000 maximum"
    with deployment(gateway, always(upstream_error(400, "invalid_request_error", text))) as rig:
        response: Final = rig.post(rig.body())
        assert response.status_code == 400, response.text
        error: Final = anthropic_error(response)
        assert error["type"] == "invalid_request_error"
        assert text in str(error["message"])


def test_a_provider_error_body_that_is_not_json_is_still_wrapped_in_the_envelope(gateway: Gateway) -> None:
    """A load balancer in front of the provider answers 500 with an HTML page. The caller must get
    HTTP 500 and a valid Anthropic envelope with `error.type == "api_error"`, not the raw HTML and
    not a proxy crash."""
    html: Final = Reply(status=500, body=b"<html><body>synthetic bad day</body></html>", content_type="text/html")
    with deployment(gateway, always(html)) as rig:
        response: Final = rig.post(rig.body())
        assert response.status_code == 500, response.text
        assert anthropic_error(response)["type"] == "api_error"


def test_a_dropped_provider_connection_is_answered_500_api_error(gateway: Gateway) -> None:
    """The provider accepts the request and closes the connection without answering. The caller must
    get an Anthropic envelope with `error.type == "api_error"`. UNVERIFIED: status 500 is expected
    because litellm's `APIConnectionError` carries status 500; check
    `exception_mapping_utils.py` if a 503 comes back."""
    with deployment(gateway, always(Reply(drop_connection=True))) as rig:
        response: Final = rig.post(rig.body())
        assert response.status_code == 500, response.text
        assert anthropic_error(response)["type"] == "api_error"


def test_a_stalled_provider_is_cut_off_at_the_deployment_timeout_with_408_api_error(gateway: Gateway) -> None:
    """The deployment has `timeout: 1` and the provider waits 6 seconds before answering. The caller
    must get HTTP 408 in the Anthropic envelope with `error.type == "api_error"` (408 has no Anthropic
    type of its own), well before the provider would have answered, and the provider is called once."""

    def stall(_request: Request) -> Reply:
        time.sleep(_STALL_SECONDS)
        return message_reply(f"msg_{marker()}")

    with deployment(gateway, stall, timeout=1) as rig:
        response: Final = timed(lambda: rig.post(rig.body()))
        assert response.value.status_code == 408, response.value.text
        assert anthropic_error(response.value)["type"] == "api_error"
        assert_cut_short(response.seconds, injected_delay=_STALL_SECONDS)
        assert len(rig.calls()) == 1


@pytest.mark.parametrize("stream", (False, True), ids=("non_stream", "stream"))
def test_an_unknown_model_is_refused_with_400_and_the_models_hint(gateway: Gateway, stream: bool) -> None:
    """The caller names a model the proxy has no deployment for. The answer must be HTTP 400 as JSON
    in the Anthropic envelope with `error.type == "invalid_request_error"` and a message containing
    `Invalid model name passed in model=<name>. Call \\`/v1/models\\` to view available models for your key.`
    The Python proxy prefixes the message with `anthropic_messages: ` and the Rust gateway with
    `/v1/messages: `, so the prefix is not asserted."""
    unknown: Final = f"no-such-model-{marker()}"
    response: Final = gateway.request(
        "POST",
        "/v1/messages",
        {"model": unknown, "max_tokens": 16, "stream": stream, "messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 400, response.text
    error: Final = anthropic_error(response)
    assert error["type"] == "invalid_request_error"
    assert (
        f"Invalid model name passed in model={unknown}. Call `/v1/models` to view available models for your key."
        in str(error["message"])
    )


def test_the_unknown_model_message_names_the_route_as_anthropic_messages(gateway: Gateway) -> None:
    """Today's exact unknown-model message starts with the route type, `anthropic_messages: Invalid model
    name passed in model=...` (`route_llm_request.py`: `anthropic_messages` has no entry in
    `ROUTE_ENDPOINT_MAPPING`). The Rust gateway writes `/v1/messages: ...` instead
    (`gateway-inference/src/messages.rs`). This test pins today's prefix so the change is a decision."""
    unknown: Final = f"no-such-model-{marker()}"
    response: Final = gateway.request(
        "POST", "/v1/messages", {"model": unknown, "max_tokens": 16, "messages": [{"role": "user", "content": "hi"}]}
    )
    assert response.status_code == 400, response.text
    assert f"anthropic_messages: Invalid model name passed in model={unknown}." in str(
        anthropic_error(response)["message"]
    )


def test_a_missing_max_tokens_is_refused_with_400_before_the_provider(gateway: Gateway) -> None:
    """`max_tokens` is required by Anthropic. A request without it must be refused by the proxy with
    HTTP 400, `error.type == "invalid_request_error"` and a message containing
    `Missing required parameter: 'max_tokens'.` (`route_llm_request.py`
    `REQUIRED_BODY_PARAMS_BY_ROUTE`), and the provider must not be called."""
    with deployment(gateway, echo_message) as rig:
        response: Final = rig.post({"model": rig.model, "messages": [{"role": "user", "content": "hi"}]})
        assert response.status_code == 400, response.text
        error: Final = anthropic_error(response)
        assert error["type"] == "invalid_request_error"
        assert "Missing required parameter: 'max_tokens'." in str(error["message"])
        assert rig.calls() == ()


def test_a_null_max_tokens_is_refused_like_a_missing_one(gateway: Gateway) -> None:
    """`"max_tokens": null` is still refused with HTTP 400 and no provider call, but not by the proxy's
    missing-parameter check: `anthropic_messages` is listed in `REQUIRED_PRESENT_BODY_PARAMS_BY_ROUTE`
    (`route_llm_request.py`), which only asks that the key be present, so an explicit null passes it.
    The Anthropic request transformation then refuses the null with its own message,
    `max_tokens is required for Anthropic /v1/messages API` (`messages/native/transformation.py`),
    before any HTTP call is made."""
    with deployment(gateway, echo_message) as rig:
        response: Final = rig.post(rig.body(max_tokens=None))
        assert response.status_code == 400, response.text
        assert "max_tokens is required for Anthropic /v1/messages API" in str(anthropic_error(response)["message"])
        assert rig.calls() == ()


def test_a_missing_messages_field_is_refused_with_400_before_the_provider(gateway: Gateway) -> None:
    """A request with `model` and `max_tokens` but no `messages` must be refused with HTTP 400,
    `error.type == "invalid_request_error"` and a message containing
    `Missing required parameter: 'messages'.`, without a provider call."""
    with deployment(gateway, echo_message) as rig:
        response: Final = rig.post({"model": rig.model, "max_tokens": 16})
        assert response.status_code == 400, response.text
        error: Final = anthropic_error(response)
        assert error["type"] == "invalid_request_error"
        assert "Missing required parameter: 'messages'." in str(error["message"])
        assert rig.calls() == ()


def test_a_missing_model_is_refused_with_400(gateway: Gateway) -> None:
    """A request with `messages` and `max_tokens` but no `model` must be refused with HTTP 400 in the
    Anthropic envelope with `error.type == "invalid_request_error"`. UNVERIFIED: the message is
    expected to contain `Missing required parameter: 'model'` (`route_llm_request.py`
    `ProxyMissingRequiredParamError`); the Rust gateway says `model is required`, so only status and
    type are asserted strictly."""
    response: Final = gateway.request(
        "POST", "/v1/messages", {"max_tokens": 16, "messages": [{"role": "user", "content": "hi"}]}
    )
    assert response.status_code == 400, response.text
    assert anthropic_error(response)["type"] == "invalid_request_error"


def test_malformed_json_is_refused_with_400_invalid_json_payload(gateway: Gateway) -> None:
    """The body is `{"model": ` (truncated JSON) with a JSON content type. The answer must be HTTP 400
    in the Anthropic envelope, `error.type == "invalid_request_error"`, with a message that starts
    with `Invalid JSON payload` (`http_parsing_utils.py` `read_request_body`). The Rust gateway's
    text is `invalid JSON`, so the prefix check is the part to revisit at cutover."""
    with deployment(gateway, echo_message) as rig:
        response: Final = rig.post_raw(
            {_AUTH: f"Bearer {gateway.key}", "content-type": "application/json"}, content=b'{"model": '
        )
        assert response.status_code == 400, response.text
        error: Final = anthropic_error(response)
        assert error["type"] == "invalid_request_error"
        assert str(error["message"]).startswith("Invalid JSON payload")
        assert rig.calls() == ()


def test_a_json_array_body_is_refused_with_400_expected_an_object(gateway: Gateway) -> None:
    """The body is valid JSON but an array, `[1, 2]`. The answer must be HTTP 400 in the Anthropic
    envelope with a message containing `Invalid JSON payload: expected an object`. The Rust
    gateway's text for the same case is `expected a JSON object`."""
    response: Final = gateway.client.post(
        "/v1/messages",
        content=b"[1, 2]",
        headers={_AUTH: f"Bearer {gateway.key}", "content-type": "application/json"},
    )
    assert response.status_code == 400, response.text
    error: Final = anthropic_error(response)
    assert error["type"] == "invalid_request_error"
    assert "expected an object" in str(error["message"])


def test_a_get_on_the_messages_route_is_answered_405(gateway: Gateway) -> None:
    """`/v1/messages` only accepts `POST`. A `GET` with a valid key must be answered HTTP 405 and must
    not be treated as a model call."""
    response: Final = gateway.request("GET", "/v1/messages")
    assert response.status_code == 405, response.text


def test_a_failed_call_is_followed_by_a_working_call_on_the_same_deployment(gateway: Gateway) -> None:
    """The provider rejects prompts containing `sad` with 529 and answers everything else. A failing
    call followed by a healthy one on the same deployment must give 529 then 200: one overloaded
    answer must not put the deployment into cooldown on this proxy (`disable_cooldowns: true`) or
    poison the connection pool."""

    def respond(request: Request) -> Reply:
        if b"sad-" in request.body:
            return upstream_error(529, "overloaded_error", "Overloaded")
        return echo_message(request)

    with deployment(gateway, respond) as rig:
        assert rig.post(rig.body(f"sad-{marker()}")).status_code == 529
        assert rig.post(rig.body(f"fine-{marker()}")).status_code == 200
        assert len(rig.calls()) == 2


@pytest.mark.parametrize("uncached", (False, True), ids=("cache_default", "no_cache"))
def test_provider_errors_look_the_same_with_and_without_the_response_cache(gateway: Gateway, uncached: bool) -> None:
    """A provider 429 on the default path and on a `cache: {"no-cache": true}` call (the shape
    eligible for the Rust route) must both be answered HTTP 429 with `error.type == "rate_limit_error"`
    and the provider's text inside the message, after exactly one provider call."""
    text: Final = f"synthetic rate limit {marker()}"
    with deployment(gateway, always(upstream_error(429, "rate_limit_error", text))) as rig:
        response: Final = rig.post(rig.body(**(NO_CACHE if uncached else {})))
        assert response.status_code == 429, response.text
        error: Final = anthropic_error(response)
        assert error["type"] == "rate_limit_error"
        assert text in str(error["message"])
        assert len(rig.calls()) == 1
