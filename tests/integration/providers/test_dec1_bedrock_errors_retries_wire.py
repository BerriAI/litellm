"""Bedrock error mapping, retries and fallbacks on `/v1/chat/completions`.

Bedrock answers errors as `{"message": ...}` with the AWS error class in `x-amzn-ErrorType`. The proxy maps
them in `_map_bedrock_exception` (`litellm/litellm_core_utils/exception_mapping_utils.py:987`) partly by
message text and partly by status: throttling and model-not-ready become 429 `RateLimitError`, validation
400 `BadRequestError` (or `ContextWindowExceededError` for "Input is too long"), `AccessDeniedException`
403 `PermissionDeniedError`, an invalid security token 401 `AuthenticationError`, "The provided model
identifier is invalid" on Converse 404 `NotFoundError`, 503 `ServiceUnavailableError`, and a Bedrock 500 is
answered as 503. The caller sees the mapped status and an OpenAI error envelope whose `code` is that status.

Retries are decided by the router core, already ported to Rust (`litellm/rust/crates/router-core/src/retry.rs:76`):
429, 408, 409 and 5xx are retried while a healthy deployment remains; 400 and 404 are not; 401 and 403 are
not when the group has a single deployment. `fallbacks` move a failed call to another model group, and
`context_window_fallbacks` catch context window errors. The shared proxy sets no `num_retries`, so every
test that counts upstream attempts passes `num_retries` explicitly.
"""

from __future__ import annotations

from collections.abc import Iterator
from itertools import count
from typing import Final

import pytest
from integration._support.client import Gateway
from integration._support.wire import Reply, Request
from integration.providers.dec1_bedrock_support import (
    CLAUDE_HAIKU_55,
    aws_error,
    bedrock,
    converse_reply,
    error_message,
)

MODEL: Final = f"bedrock/{CLAUDE_HAIKU_55}"


def _always(reply: Reply) -> object:
    def respond(_: Request) -> Reply:
        return reply

    return respond


@pytest.mark.parametrize(
    ("status", "error_type", "message", "expected"),
    [
        (429, "ThrottlingException", "Too many requests, please wait before trying again.", 429),
        (429, "ModelNotReadyException", "Model is not ready to serve inference requests.", 429),
        (400, "ValidationException", "Malformed input request: #: extraneous key [foo] is not permitted", 400),
        (503, "ServiceUnavailableException", "Service is temporarily unavailable.", 503),
        (404, "ResourceNotFoundException", "Requested resource not found.", 404),
    ],
    ids=["throttling", "model_not_ready", "validation", "unavailable", "not_found"],
)
def test_a_bedrock_error_status_reaches_the_caller_with_its_text(
    gateway: Gateway, status: int, error_type: str, message: str, expected: int
) -> None:
    """Bedrock answers the Converse call with `status` and `{"message": <text>}` and the AWS error class header.

    With `num_retries: 0` the caller must get HTTP `expected`, an OpenAI error envelope whose `error.code`
    is the same status as a string, and Bedrock's text inside `error.message`.
    """
    with bedrock(gateway, _always(aws_error(status, error_type, message)), model=MODEL) as rig:  # pyright: ignore[reportArgumentType]
        response: Final = rig.chat(num_retries=0)
        assert response.status_code == expected, response.text
        assert message in error_message(response), response.text
        assert str(response.json()["error"]["code"]) == str(expected), response.text


def test_a_bedrock_500_is_answered_as_503(gateway: Gateway) -> None:
    """Bedrock answers HTTP 500 InternalServerException.

    Pinned current behaviour: `_map_bedrock_exception` raises `ServiceUnavailableError` for any Bedrock 500
    (`exception_mapping_utils.py:1092`), so the caller gets HTTP 503, not 500.
    """
    with bedrock(gateway, _always(aws_error(500, "InternalServerException", "Internal failure")), model=MODEL) as rig:  # pyright: ignore[reportArgumentType]
        response: Final = rig.chat(num_retries=0)
        assert response.status_code == 503, response.text


def test_an_access_denied_exception_is_a_403_permission_error(gateway: Gateway) -> None:
    """Bedrock answers HTTP 403 whose text names `AccessDeniedException` (model access not granted).

    The caller must get HTTP 403 and a message containing `PermissionDeniedError`, so it can tell a missing
    model grant from a bad key.
    """
    message: Final = "AccessDeniedException: You don't have access to the model with the specified model ID."
    with bedrock(gateway, _always(aws_error(403, "AccessDeniedException", message)), model=MODEL) as rig:  # pyright: ignore[reportArgumentType]
        response: Final = rig.chat(num_retries=0)
        assert response.status_code == 403, response.text
        assert "PermissionDeniedError" in error_message(response), response.text


def test_real_bedrock_access_denied_body_without_the_class_name_is_still_a_403(gateway: Gateway) -> None:
    """Real Bedrock puts `AccessDeniedException` only in the `x-amzn-ErrorType` header; the body is just the sentence.

    `_map_bedrock_exception` matches the class name in the text and has no 403 status branch, so this
    relies on the generic status mapping after it. The caller must still get HTTP 403.
    UNVERIFIED: traced only to the end of `_map_bedrock_exception` (`exception_mapping_utils.py:1157`).
    """
    with bedrock(
        gateway,
        _always(
            aws_error(403, "AccessDeniedException", "You don't have access to the model with the specified model ID.")
        ),  # pyright: ignore[reportArgumentType]
        model=MODEL,
    ) as rig:
        response: Final = rig.chat(num_retries=0)
        assert response.status_code == 403, response.text


def test_an_invalid_security_token_is_a_401_authentication_error(gateway: Gateway) -> None:
    """Bedrock answers HTTP 403 UnrecognizedClientException `The security token included in the request is invalid.`.

    The text match comes first (`exception_mapping_utils.py:1053`), so the caller must get HTTP 401, not 403.
    """
    with bedrock(
        gateway,
        _always(
            aws_error(403, "UnrecognizedClientException", "The security token included in the request is invalid.")
        ),  # pyright: ignore[reportArgumentType]
        model=MODEL,
    ) as rig:
        response: Final = rig.chat(num_retries=0)
        assert response.status_code == 401, response.text
        assert "Invalid Authentication" in error_message(response), response.text


def test_input_too_long_is_a_400_context_window_error(gateway: Gateway) -> None:
    """Bedrock answers HTTP 400 `Input is too long for requested model.`.

    The caller must get HTTP 400 whose message contains `Context Window Error`, the marker routers and SDKs
    use to trigger context window fallbacks.
    """
    with bedrock(
        gateway,
        _always(aws_error(400, "ValidationException", "Input is too long for requested model.")),
        model=MODEL,  # pyright: ignore[reportArgumentType]
    ) as rig:
        response: Final = rig.chat(num_retries=0)
        assert response.status_code == 400, response.text
        assert "Context Window Error" in error_message(response), response.text


def test_an_invalid_model_identifier_on_converse_is_a_404(gateway: Gateway) -> None:
    """Converse answers HTTP 400 `The provided model identifier is invalid.` for a model the cost map routes to Converse.

    It is mapped to `NotFoundError` with status 404 (`exception_mapping_utils.py:1027`), so the caller gets
    HTTP 404 and the error is not retried.
    """
    with bedrock(
        gateway,
        _always(aws_error(400, "ValidationException", "The provided model identifier is invalid.")),
        model=MODEL,  # pyright: ignore[reportArgumentType]
    ) as rig:
        response: Final = rig.chat(num_retries=2)
        assert response.status_code == 404, response.text
        assert len(rig.posts()) == 1


def _throttle_then_ok(failures: int) -> object:
    calls: Final[Iterator[int]] = count()

    def respond(_: Request) -> Reply:
        if next(calls) < failures:
            return aws_error(429, "ThrottlingException", "Rate exceeded")
        return converse_reply()

    return respond


def test_a_throttled_call_is_retried_and_the_retry_answer_is_returned(gateway: Gateway) -> None:
    """Bedrock throttles the first call and answers the second, with `num_retries: 2` on the request.

    429 is retryable while the deployment is healthy (cooldowns are disabled on the shared proxy), so the
    caller must get HTTP 200 with the Converse text and the peer must have seen exactly two POSTs.
    """
    with bedrock(gateway, _throttle_then_ok(1), model=MODEL) as rig:  # pyright: ignore[reportArgumentType]
        response: Final = rig.chat(num_retries=2)
        assert response.status_code == 200, response.text
        assert response.json()["choices"][0]["message"]["content"] == "pong"
        assert len(rig.posts()) == 2


def test_throttling_on_every_attempt_exhausts_the_retries_and_returns_429(gateway: Gateway) -> None:
    """Bedrock throttles every call; the request sets `num_retries: 2`.

    The router must try 1 + 2 = 3 times and then answer HTTP 429.
    """
    with bedrock(gateway, _always(aws_error(429, "ThrottlingException", "Rate exceeded")), model=MODEL) as rig:  # pyright: ignore[reportArgumentType]
        response: Final = rig.chat(num_retries=2)
        assert response.status_code == 429, response.text
        assert len(rig.posts()) == 3


def test_a_validation_error_is_not_retried(gateway: Gateway) -> None:
    """Bedrock answers 400 ValidationException; the request sets `num_retries: 2`.

    400 is not retryable (`should_retry_status`), so the peer must see exactly one POST and the caller HTTP 400.
    """
    with bedrock(
        gateway,
        _always(aws_error(400, "ValidationException", "Malformed input request")),
        model=MODEL,  # pyright: ignore[reportArgumentType]
    ) as rig:
        response: Final = rig.chat(num_retries=2)
        assert response.status_code == 400, response.text
        assert len(rig.posts()) == 1


def test_access_denied_on_a_single_deployment_group_is_not_retried(gateway: Gateway) -> None:
    """Bedrock answers 403 AccessDeniedException on the only deployment of the group; `num_retries: 2`.

    `should_retry` refuses to retry auth and permission failures when the group has at most one deployment
    (`retry.rs:96`), so the peer must see exactly one POST.
    """
    with bedrock(
        gateway,
        _always(aws_error(403, "AccessDeniedException", "AccessDeniedException: no model access")),
        model=MODEL,  # pyright: ignore[reportArgumentType]
    ) as rig:
        response: Final = rig.chat(num_retries=2)
        assert response.status_code == 403, response.text
        assert len(rig.posts()) == 1


def test_a_service_unavailable_answer_is_retried(gateway: Gateway) -> None:
    """Bedrock answers 503 ServiceUnavailableException first and then succeeds; `num_retries: 1`.

    5xx is retryable, so the caller must get HTTP 200 after two POSTs.
    """
    calls: Final[Iterator[int]] = count()

    def respond(_: Request) -> Reply:
        if next(calls) == 0:
            return aws_error(503, "ServiceUnavailableException", "Service is temporarily unavailable.")
        return converse_reply()

    with bedrock(gateway, respond, model=MODEL) as rig:
        response: Final = rig.chat(num_retries=1)
        assert response.status_code == 200, response.text
        assert len(rig.posts()) == 2


def test_a_throttled_bedrock_group_falls_back_to_the_named_fallback_group(gateway: Gateway) -> None:
    """Group A (Bedrock) always throttles; the request names `fallbacks: [<group B>]` and `num_retries: 0`.

    The caller must get HTTP 200 with group B's answer, A must have been called once, and B once.
    """
    with (
        bedrock(gateway, _always(aws_error(429, "ThrottlingException", "Rate exceeded")), model=MODEL) as primary,
        bedrock(  # pyright: ignore[reportArgumentType]
            gateway, lambda _: converse_reply(content=[{"text": "from fallback"}]), model=MODEL
        ) as fallback,
    ):
        response: Final = primary.chat(num_retries=0, fallbacks=[fallback.model])
        assert response.status_code == 200, response.text
        assert response.json()["choices"][0]["message"]["content"] == "from fallback"
        assert len(primary.posts()) == 1
        assert len(fallback.posts()) == 1


def test_a_context_window_error_uses_context_window_fallbacks(gateway: Gateway) -> None:
    """Group A answers 400 `Input is too long for requested model.`; the request names `context_window_fallbacks: [{<A>: [<B>]}]`.

    The context window error must move the call to group B without retrying A, so the caller gets B's
    answer, A saw one POST and B one.
    """
    with (
        bedrock(
            gateway,
            _always(aws_error(400, "ValidationException", "Input is too long for requested model.")),
            model=MODEL,  # pyright: ignore[reportArgumentType]
        ) as primary,
        bedrock(gateway, lambda _: converse_reply(content=[{"text": "bigger window"}]), model=MODEL) as bigger,
    ):
        response: Final = primary.chat(num_retries=2, context_window_fallbacks=[{primary.model: [bigger.model]}])
        assert response.status_code == 200, response.text
        assert response.json()["choices"][0]["message"]["content"] == "bigger window"
        assert len(primary.posts()) == 1
        assert len(bigger.posts()) == 1


def test_a_failed_bedrock_call_writes_a_failure_spend_row(gateway: Gateway) -> None:
    """A Converse call that Bedrock throttles (num_retries 0) still leaves an audit trail.

    One `LiteLLM_SpendLogs` row with `status` `failure`, `spend` 0 and `model_group` equal to the group must
    land for the call id the proxy returned in `x-litellm-call-id`. UNVERIFIED: that the failure row's
    `request_id` equals the `x-litellm-call-id` header on an error answer.
    """
    from integration.providers.dec1_bedrock_support import spend_row

    with bedrock(gateway, _always(aws_error(429, "ThrottlingException", "Rate exceeded")), model=MODEL) as rig:  # pyright: ignore[reportArgumentType]
        response: Final = rig.chat(num_retries=0)
        assert response.status_code == 429, response.text
        row: Final = spend_row(response.headers["x-litellm-call-id"])
        assert row["status"] == "failure"
        assert float(str(row["spend"])) == 0
        assert row["model_group"] == rig.model
