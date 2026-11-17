"""Contract: how an OpenAI provider error on `POST /v1/chat/completions` reaches the caller.

When the provider answers with an error the proxy maps it to a litellm exception
(`litellm/litellm_core_utils/exception_mapping_utils.py`) and answers with the OpenAI error envelope
`{"error": {"message", "type", "param", "code"}}`. The caller-visible contract is: the HTTP status is the
provider's status, `error.code` is that status as a string, the message starts with litellm's exception prefix and
contains the provider's own message, provider response headers are relayed with an `llm_provider-` prefix
(`get_response_headers`), and every error answer still carries `x-litellm-call-id`. OpenAI's
`context_length_exceeded` code is promoted to `ContextWindowExceededError`, and a 429 keeps the provider's
`retry-after` so SDK clients back off correctly.

Every deployment here sets `num_retries: 0` and `max_retries: 0`, so the provider is called exactly once and the
status cannot be changed by a retry. The error type names (`throttling_error`, `authentication_error`,
`internal_server_error`) are the ones the embeddings suite pinned for the same exception classes
(`test_dec1_textapis_embeddings_openai.py`); for chat they are UNVERIFIED but expected to match because both
routes share `handle_llm_api_exception`.
"""

from __future__ import annotations

import uuid
from typing import Final

import pytest
from integration._support.client import Gateway, eventually
from integration._support.wire import Reply
from integration.providers.dec1_openai_support import (
    Call,
    Respond,
    chat,
    deployment,
    error_message,
    error_of,
    openai_error,
    peer,
)

_MODEL: Final = "openai/gpt-4o-mini"
_ERROR_TYPE: Final = {
    400: "invalid_request_error",
    401: "invalid_request_error",
    404: "invalid_request_error",
    429: "throttling_error",
    500: "internal_server_error",
    503: "internal_server_error",
}
_PREFIX: Final = {
    400: "litellm.BadRequestError",
    401: "litellm.AuthenticationError",
    404: "litellm.NotFoundError",
    429: "litellm.RateLimitError",
    500: "litellm.InternalServerError",
    503: "litellm.ServiceUnavailableError",
}


def _always(reply: Reply) -> Respond:
    def respond(call: Call) -> Reply:
        return reply

    return respond


@pytest.mark.parametrize("status", [400, 401, 404, 429, 500, 503])
def test_the_provider_status_and_message_reach_the_caller(gateway: Gateway, status: int) -> None:
    """The provider answers `status` with an OpenAI error body whose message is `dec1 provider said <status>`.

    The caller gets the same HTTP status, `error.code == "<status>"`, and a message that starts with the
    matching litellm exception class name and contains the provider's text. The provider is called once.
    The `type` strings are pinned from run 44176682961: 401 and 404 are `invalid_request_error` and 503 is
    `internal_server_error` (not `authentication_error`, `not_found_error`, `service_unavailable_error`).
    """
    text: Final = f"dec1 provider said {status}"
    with peer(_always(openai_error(status, text))) as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _MODEL)
        response: Final = chat(gateway, model)
        assert response.status_code == status, response.text
        error: Final = error_of(response)
        assert str(error["code"]) == str(status), response.text
        message: Final = error_message(response)
        assert message.startswith(_PREFIX[status]) and text in message, message
        assert error["type"] == _ERROR_TYPE[status], response.text
        assert len(provider.posts()) == 1


def test_a_429_keeps_the_provider_retry_after(gateway: Gateway) -> None:
    """The provider answers 429 with `retry-after: 7` and `x-ratelimit-remaining-requests: 0`.

    The caller gets 429 and the retry hint survives, either as `retry-after: 7` or as
    `llm_provider-retry-after: 7`, the same tolerance the streaming suite uses. Clients that obey retry-after
    would hammer the provider if a port dropped it. `x-ratelimit-remaining-requests` is relayed under its own
    name too (`get_response_headers` copies the four OpenAI rate limit headers unprefixed).
    """
    reply: Final = openai_error(
        429, "dec1 slow down", headers={"retry-after": "7", "x-ratelimit-remaining-requests": "0"}
    )
    with peer(_always(reply)) as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _MODEL)
        response: Final = chat(gateway, model)
        assert response.status_code == 429, response.text
        headers: Final = response.headers
        assert (headers.get("llm_provider-retry-after") or headers.get("retry-after")) == "7", headers
        assert (
            headers.get("x-ratelimit-remaining-requests") or headers.get("llm_provider-x-ratelimit-remaining-requests")
        ) == "0", headers


def test_an_invalid_request_error_keeps_the_provider_param(gateway: Gateway) -> None:
    """The provider answers 400 `{"type": "invalid_request_error", "param": "messages[0].content", "code":
    "invalid_value"}` with message `dec1 bad content`.

    The caller gets 400, `error.type == "invalid_request_error"`, and a message containing `dec1 bad content`.
    UNVERIFIED: whether `error.param` keeps `messages[0].content`; the proxy builds `param` from the exception, and
    the embeddings suite saw `None` there, so this test only pins the status, type and text.
    """
    reply: Final = openai_error(400, "dec1 bad content", param="messages[0].content", code="invalid_value")
    with peer(_always(reply)) as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _MODEL)
        response: Final = chat(gateway, model)
        assert response.status_code == 400, response.text
        assert error_of(response)["type"] == "invalid_request_error"
        assert "dec1 bad content" in error_message(response)


def test_context_length_exceeded_is_answered_as_a_context_window_error(gateway: Gateway) -> None:
    """The provider answers 400 with `code: "context_length_exceeded"` and OpenAI's wording `This model's maximum
    context length is 128000 tokens.`

    The caller gets 400 and a message that names `ContextWindowExceededError` and keeps the provider text, which
    is what context-window fallbacks key on (`exception_mapping_utils.py`).
    """
    reply: Final = openai_error(
        400,
        "This model's maximum context length is 128000 tokens. However, your messages resulted in 200000 tokens.",
        code="context_length_exceeded",
        param="messages",
    )
    with peer(_always(reply)) as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _MODEL)
        response: Final = chat(gateway, model)
        assert response.status_code == 400, response.text
        message: Final = error_message(response)
        assert "ContextWindowExceededError" in message, message
        assert "maximum context length is 128000 tokens" in message, message


def test_an_error_answer_carries_the_call_id_and_the_provider_request_id(gateway: Gateway) -> None:
    """The provider answers 500 with `x-request-id: req_dec1err`.

    The caller's error response has a UUID `x-litellm-call-id` and relays the provider request id as
    `llm_provider-x-request-id: req_dec1err`, so support can trace a failure to the provider log.
    """
    reply: Final = openai_error(500, "dec1 boom", headers={"x-request-id": "req_dec1err"})
    with peer(_always(reply)) as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _MODEL)
        response: Final = chat(gateway, model)
        assert response.status_code == 500, response.text
        assert uuid.UUID(response.headers["x-litellm-call-id"]), response.headers
        assert response.headers.get("llm_provider-x-request-id") == "req_dec1err", response.headers


def test_a_failed_call_writes_a_failure_spend_row_with_zero_spend(gateway: Gateway) -> None:
    """The provider answers 429. One SpendLogs row lands for the caller's `x-litellm-call-id` with
    `status = 'failure'` and `spend = 0`. UNVERIFIED: failure rows are keyed by the call id
    (`request_id = litellm_call_id`) as in `test_chat_completion_bad_model_with_spend_logs`."""
    with peer(_always(openai_error(429, "dec1 slow"))) as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _MODEL)
        response: Final = chat(gateway, model)
        assert response.status_code == 429, response.text
        call_id: Final = response.headers["x-litellm-call-id"]
        rows: Final = eventually(
            lambda: gateway.rows('SELECT status, spend FROM "LiteLLM_SpendLogs" WHERE request_id = %s', (call_id,)),
            lambda found: len(found) == 1,
            seconds=70,
        )
        assert (rows[0]["status"], float(str(rows[0]["spend"]))) == ("failure", 0.0), rows


def test_a_non_json_502_from_the_provider_is_answered_with_an_openai_envelope(gateway: Gateway) -> None:
    """The provider (or a gateway in front of it) answers 502 with an HTML body.

    The caller still gets an OpenAI-shaped JSON error envelope (an `error` object with a `message`) and a 5xx
    status, never the HTML page. UNVERIFIED: the exact status (502 kept, or mapped to 500) depends on
    `exception_mapping_utils.py`; this pins only the shape and the 5xx class.
    """
    reply: Final = Reply(status=502, body=b"<html><body>Bad gateway</body></html>", content_type="text/html")
    with peer(_always(reply)) as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _MODEL)
        response: Final = chat(gateway, model)
        assert 500 <= response.status_code < 600, response.text
        assert isinstance(error_of(response)["message"], str), response.text
        assert response.headers["content-type"].startswith("application/json"), response.headers


def test_a_stream_that_fails_on_open_is_answered_with_a_plain_error_status(gateway: Gateway) -> None:
    """The caller asks for `stream: true` and the provider answers 400 before any chunk.

    Because no byte was streamed yet the proxy answers HTTP 400 with the JSON error envelope (not a 200 SSE
    stream with an error event), so SDKs raise `BadRequestError`.
    """
    with peer(_always(openai_error(400, "dec1 bad stream request"))) as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _MODEL)
        response: Final = chat(gateway, model, stream=True)
        assert response.status_code == 400, response.text
        assert "dec1 bad stream request" in error_message(response)
