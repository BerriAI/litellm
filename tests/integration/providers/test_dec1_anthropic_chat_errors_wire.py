"""Contract: how an Anthropic provider error on `POST /v1/chat/completions` reaches the OpenAI-shaped caller.

`_map_anthropic_exception` (`litellm/litellm_core_utils/exception_mapping_utils.py:615`) maps the provider's
status and text to litellm exceptions: 400 and 413 to BadRequestError, `prompt is too long` to
ContextWindowExceededError, 401 to AuthenticationError, 403 to PermissionDeniedError, 404 to NotFoundError, 429 to
RateLimitError, 500 to InternalServerError, 502 to BadGatewayError, 503 to ServiceUnavailableError, and any
`overloaded_error` (or a 529) to InternalServerError carrying `provider_status_code: 529`. On the OpenAI routes an
overload is answered `503` with `type: service_unavailable_error` and `code: server_is_overloaded`
(`litellm/proxy/anthropic_endpoints/answered_status.py`, `OverloadedProxyException`), so OpenAI SDK clients back
off instead of treating it as a hard server failure.

Deployments set `num_retries: 0` unless a test says otherwise, so the peer sees exactly one call per failure.
"""

from __future__ import annotations

from typing import Final

import pytest
from integration._support.client import Gateway
from integration._support.wire import Reply
from integration.providers.dec1_anthropic_support import (
    PROVIDER_KEY,
    Call,
    Respond,
    always,
    anthropic_error,
    chat,
    default_respond,
    deployment,
    error_message,
    error_of,
    json_reply,
    message_json,
    peer,
    served,
    spend_row_by_group,
)


def _first_fails_then_answers(failure: Reply) -> Respond:
    """Answer the first POST with `failure` and every later POST with the canned message."""
    posts: Final[list[Call]] = []  # mutable-ok: appended by the peer's handler thread

    def respond(call: Call) -> Reply:
        if call.method != "POST":
            return default_respond(call)
        posts.append(call)
        return failure if len(posts) == 1 else json_reply(message_json(call.body))

    return respond


def test_a_529_overloaded_error_is_answered_503_server_is_overloaded(gateway: Gateway) -> None:
    """The provider answers 529 `{"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}}`.

    The caller must get HTTP 503 with `error.type: service_unavailable_error` and `error.code: server_is_overloaded`,
    and the message must contain `Overloaded`. A port that answers 500 or relays 529 breaks OpenAI SDK back-off.
    """
    with (
        peer(always(anthropic_error(529, "overloaded_error", "Overloaded"))) as provider,
        gateway.scenario() as scenario,
    ):
        response: Final = chat(gateway, deployment(scenario, provider))
        assert response.status_code == 503, response.text
        error: Final = error_of(response)
        assert error["type"] == "service_unavailable_error"
        assert error["code"] == "server_is_overloaded"
        assert "Overloaded" in error_message(response)
        assert len(provider.posts()) == 1


def test_an_overloaded_error_body_on_a_500_is_also_answered_as_an_overload(gateway: Gateway) -> None:
    """The provider answers HTTP 500 with an `overloaded_error` body. Any 5xx carrying the overload text maps to
    `provider_status_code: 529` (`_anthropic_overload_status`), so the caller must again get 503 with
    `code: server_is_overloaded`.
    """
    with (
        peer(always(anthropic_error(500, "overloaded_error", "Overloaded"))) as provider,
        gateway.scenario() as scenario,
    ):
        response: Final = chat(gateway, deployment(scenario, provider))
        assert response.status_code == 503, response.text
        assert error_of(response)["code"] == "server_is_overloaded"


def test_a_streamed_request_whose_provider_is_overloaded_before_the_first_byte_gets_a_503_json_error(
    gateway: Gateway,
) -> None:
    """The caller streams; the provider answers 529 overloaded before sending any event.

    Nothing was streamed yet, so the caller must get a plain HTTP 503 JSON error with `code: server_is_overloaded`
    rather than a 200 stream that carries an error frame. UNVERIFIED: that the streaming path raises before the
    response starts for a non-2xx upstream status.
    """
    with (
        peer(always(anthropic_error(529, "overloaded_error", "Overloaded"))) as provider,
        gateway.scenario() as scenario,
    ):
        response: Final = chat(gateway, deployment(scenario, provider), stream=True)
        assert response.status_code == 503, response.text
        assert error_of(response)["code"] == "server_is_overloaded"


def test_a_429_rate_limit_keeps_its_status_and_the_retry_after_hint(gateway: Gateway) -> None:
    """The provider answers 429 `rate_limit_error` with `retry-after: 13` and
    `anthropic-ratelimit-requests-remaining: 0`.

    The caller must get 429, a message that starts with `litellm.RateLimitError` and contains the provider text, and
    the retry hint as `retry-after: 13` or `llm_provider-retry-after: 13` (the tolerance the OpenAI suite uses).
    """
    reply: Final = anthropic_error(
        429,
        "rate_limit_error",
        "dec1 rate limited",
        headers={"retry-after": "13", "anthropic-ratelimit-requests-remaining": "0"},
    )
    with peer(always(reply)) as provider, gateway.scenario() as scenario:
        response: Final = chat(gateway, deployment(scenario, provider))
        assert response.status_code == 429, response.text
        message: Final = error_message(response)
        assert message.startswith("litellm.RateLimitError") and "dec1 rate limited" in message
        assert (response.headers.get("retry-after") or response.headers.get("llm_provider-retry-after")) == "13"


def test_anthropic_rate_limit_headers_are_relayed_with_the_llm_provider_prefix(gateway: Gateway) -> None:
    """A successful provider reply carries `anthropic-ratelimit-requests-remaining: 41` and
    `request-id: req_dec1`.

    The caller's response must expose them as `llm_provider-anthropic-ratelimit-requests-remaining: 41` and
    `llm_provider-request-id: req_dec1` (`process_anthropic_headers`). UNVERIFIED: whether Anthropic's rate limit
    headers are also translated to the OpenAI `x-ratelimit-*` names; only the prefixed copies are pinned.
    """

    def respond(call: Call) -> Reply:
        if call.method != "POST":
            return default_respond(call)
        return json_reply(
            message_json(call.body),
            headers={"anthropic-ratelimit-requests-remaining": "41", "request-id": "req_dec1"},
        )

    with peer(respond) as provider, gateway.scenario() as scenario:
        response: Final = chat(gateway, deployment(scenario, provider))
        served(response)
        assert response.headers.get("llm_provider-anthropic-ratelimit-requests-remaining") == "41"
        assert response.headers.get("llm_provider-request-id") == "req_dec1"


@pytest.mark.parametrize(
    ("status", "kind", "prefix"),
    [
        (400, "invalid_request_error", "litellm.BadRequestError"),
        (401, "authentication_error", "litellm.AuthenticationError"),
        (403, "permission_error", "litellm.PermissionDeniedError"),
        (404, "not_found_error", "litellm.NotFoundError"),
        (500, "api_error", "litellm.InternalServerError"),
    ],
)
def test_a_provider_error_keeps_its_status_and_names_the_litellm_exception(
    gateway: Gateway, status: int, kind: str, prefix: str
) -> None:
    """The provider answers `status` with Anthropic's error envelope and message `dec1 provider said <status>`.

    The caller must get the same status, `error.code` equal to the status as a string, and a message that starts
    with the litellm exception name for that status and contains the provider text. The peer is called once.
    """
    text: Final = f"dec1 provider said {status}"
    with peer(always(anthropic_error(status, kind, text))) as provider, gateway.scenario() as scenario:
        response: Final = chat(gateway, deployment(scenario, provider))
        assert response.status_code == status, response.text
        assert str(error_of(response)["code"]) == str(status)
        message: Final = error_message(response)
        assert message.startswith(prefix) and text in message, message
        assert len(provider.posts()) == 1


def test_a_413_request_too_large_is_answered_400(gateway: Gateway) -> None:
    """The provider answers 413 `request_too_large`. `_map_anthropic_exception` maps 413 to BadRequestError, so the
    caller must get 400 with a message that starts with `litellm.BadRequestError` and keeps the provider text.
    """
    reply: Final = anthropic_error(413, "request_too_large", "dec1 request exceeds the maximum size")
    with peer(always(reply)) as provider, gateway.scenario() as scenario:
        response: Final = chat(gateway, deployment(scenario, provider))
        assert response.status_code == 400, response.text
        message: Final = error_message(response)
        assert message.startswith("litellm.BadRequestError") and "maximum size" in message


def test_a_prompt_too_long_rejection_is_a_context_window_exceeded_error(gateway: Gateway) -> None:
    """The provider answers 400 `invalid_request_error` with `prompt is too long: 210000 tokens > 200000 maximum`.

    The caller must get 400 and a message that starts with `litellm.ContextWindowExceededError`, which is what
    routers key `context_window_fallbacks` on.
    """
    reply: Final = anthropic_error(400, "invalid_request_error", "prompt is too long: 210000 tokens > 200000 maximum")
    with peer(always(reply)) as provider, gateway.scenario() as scenario:
        response: Final = chat(gateway, deployment(scenario, provider))
        assert response.status_code == 400, response.text
        assert error_message(response).startswith("litellm.ContextWindowExceededError")


def test_a_content_filtering_rejection_is_a_content_policy_violation(gateway: Gateway) -> None:
    """The provider answers 400 with a message containing `Output blocked by content filtering policy`.

    The caller must get 400 and a message that names `litellm.ContentPolicyViolationError`.

    Pinned current (run 44176682961): the message starts with `litellm.BadRequestError: litellm.ContentPolicyViolationError:`,
    because `ContentPolicyViolationError` prefixes its own name (litellm/exceptions.py:777) and then hands that
    message to `BadRequestError.__init__` (:782), which prefixes `litellm.BadRequestError: ` again.
    """
    reply: Final = anthropic_error(400, "invalid_request_error", "Output blocked by content filtering policy")
    with peer(always(reply)) as provider, gateway.scenario() as scenario:
        response: Final = chat(gateway, deployment(scenario, provider))
        assert response.status_code == 400, response.text
        assert error_message(response).startswith("litellm.BadRequestError: litellm.ContentPolicyViolationError: ")


def test_the_provider_error_never_leaks_the_deployment_key(gateway: Gateway) -> None:
    """The provider answers 401 `invalid x-api-key`. Neither the caller's error body nor its headers may contain
    the deployment key `dec1-anthropic-provider-key`.
    """
    with (
        peer(always(anthropic_error(401, "authentication_error", "invalid x-api-key"))) as provider,
        gateway.scenario() as scenario,
    ):
        response: Final = chat(gateway, deployment(scenario, provider))
        assert response.status_code == 401
        assert PROVIDER_KEY not in response.text
        assert all(PROVIDER_KEY not in value for value in response.headers.values())


def test_a_provider_error_response_still_carries_x_litellm_call_id(gateway: Gateway) -> None:
    """The provider answers 400. The caller's error response must still carry a non-empty `x-litellm-call-id`, so
    the failure can be traced to its SpendLogs failure row.
    """
    with (
        peer(always(anthropic_error(400, "invalid_request_error", "dec1 bad"))) as provider,
        gateway.scenario() as scenario,
    ):
        response: Final = chat(gateway, deployment(scenario, provider))
        assert response.status_code == 400
        assert response.headers.get("x-litellm-call-id")


def test_an_overloaded_call_is_retried_when_the_deployment_allows_retries(gateway: Gateway) -> None:
    """The deployment sets `num_retries: 1`; the provider answers 529 overloaded once, then a normal message.

    The caller must get 200 with the message, and the peer must have been called twice: overload is a retryable
    InternalServerError. UNVERIFIED: the router's retry backoff could make this slow but not change the counts.
    """
    with (
        peer(_first_fails_then_answers(anthropic_error(529, "overloaded_error", "Overloaded"))) as provider,
        gateway.scenario() as scenario,
    ):
        served(chat(gateway, deployment(scenario, provider, num_retries=1)))
        assert len(provider.posts()) == 2


def test_an_invalid_request_is_not_retried_even_with_retries_allowed(gateway: Gateway) -> None:
    """The deployment sets `num_retries: 2`; the provider answers 400 `invalid_request_error`.

    A client error cannot succeed on retry, so the caller must get 400 and the peer must have been called once.
    """
    with (
        peer(always(anthropic_error(400, "invalid_request_error", "dec1 bad"))) as provider,
        gateway.scenario() as scenario,
    ):
        response: Final = chat(gateway, deployment(scenario, provider, num_retries=2))
        assert response.status_code == 400
        assert len(provider.posts()) == 1


def test_a_failed_call_writes_a_failure_spend_row(gateway: Gateway) -> None:
    """The provider answers 500 `api_error`. A SpendLogs row for the model group must land with `status: failure`
    and spend 0. UNVERIFIED: that failure rows carry `model_group`; the messages suite pins failure rows by call id.
    """
    with peer(always(anthropic_error(500, "api_error", "dec1 boom"))) as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider)
        assert chat(gateway, model).status_code == 500
        row: Final = spend_row_by_group(gateway, model)
        assert row["status"] == "failure"
        assert float(str(row["spend"])) == 0
