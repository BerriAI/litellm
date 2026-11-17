"""Which retry budget a big provider's real error reads, through the Rust router core.

The router does not retry by HTTP status. It retries by the exception class litellm's provider mapping
raised (`litellm/litellm_core_utils/exception_mapping_utils.py`), and `retry_policy` names its budgets by
class (`RetryPolicy::retries_for` in `litellm/rust/crates/router-core/src/retry.rs:30`): the most specific
class the policy names wins, else `DefaultRetries`, else `num_retries`. Provider mapping therefore decides
which budget applies, and the providers disagree in ways a rewrite can easily flatten:

A Bedrock 500 `InternalServerException` is raised as `ServiceUnavailableError` (503), so it reads
`ServiceUnavailableErrorRetries`. Google's 503 overload text `The model is overloaded.` is raised as
`InternalServerError` (500) and reads `InternalServerErrorRetries`, while any other Google 503 reads
`ServiceUnavailableErrorRetries`. An Anthropic 529 `overloaded_error` is an `InternalServerError`. An Azure
500 without the text `Internal server error` is a bare `APIError` that no named budget covers, so it reads
`DefaultRetries`. A Vertex or Gemini 502 is an `APIConnectionError`. Every provider's throttle (Bedrock
`ThrottlingException` and `ModelNotReadyException`, Google `RESOURCE_EXHAUSTED`, Azure and Anthropic 429,
OpenAI `rate_limit_exceeded` and `insufficient_quota` alike) is a `RateLimitError`.

When a policy field applies, `should_retry_this_error` is skipped (`call.rs:1015`), so a policy can make the
router retry an error it would never retry on its own, such as a Google 400 `INVALID_ARGUMENT`.

One owned proxy runs with `num_retries: 0`, `disable_cooldowns: true` and one `model_group_retry_policy`
entry per group, so each group's budget is isolated. Most groups hold two deployments of the same
provider, which the router retries without any backoff sleep. Attempts are counted at the peer.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Final

import pytest
from integration.routing.dec1_reliab_support import (
    ANTHROPIC_FILTER_TEXT,
    ANTHROPIC_RATE_TEXT,
    ANTHROPIC_TOO_LONG_TEXT,
    AZURE_GENERIC_500_TEXT,
    AZURE_INTERNAL_TEXT,
    BEDROCK_ACCESS_TEXT,
    BEDROCK_INTERNAL_TEXT,
    BEDROCK_NOT_READY_TEXT,
    BEDROCK_THROTTLE_TEXT,
    BEDROCK_TOKEN_TEXT,
    BEDROCK_TOO_LONG_TEXT,
    BEDROCK_UNAVAILABLE_TEXT,
    BEDROCK_VALIDATION_TEXT,
    GOOGLE_BAD_GATEWAY_TEXT,
    GOOGLE_EXHAUSTED_TEXT,
    GOOGLE_INVALID_TEXT,
    GOOGLE_OVERLOADED_TEXT,
    GOOGLE_PERMISSION_TEXT,
    GOOGLE_QUOTA_TEXT,
    GOOGLE_TOO_LONG_TEXT,
    GOOGLE_UNAVAILABLE_TEXT,
    OPENAI_SERVER_TEXT,
    OPENAI_TOO_LONG_TEXT,
    OPENAI_UNAVAILABLE_TEXT,
    Rig,
    anthropic_error,
    anthropic_overloaded,
    azure_rate_limit,
    bedrock_error,
    chat,
    content_of,
    error_message,
    error_of,
    google_error,
    openai_error,
    openai_rate_limit,
    pair_of,
    reliab_rig,
)
from pydantic import JsonValue

from tests.integration._support.wire import Reply

pytestmark: Final = [pytest.mark.timeout(300), pytest.mark.shares_worker_state]

INTERNAL_TWO: Final[dict[str, JsonValue]] = {"InternalServerErrorRetries": 2, "ServiceUnavailableErrorRetries": 0}
UNAVAILABLE_TWO: Final[dict[str, JsonValue]] = {"ServiceUnavailableErrorRetries": 2, "InternalServerErrorRetries": 0}
DEFAULT_TWO: Final[dict[str, JsonValue]] = {"DefaultRetries": 2, "InternalServerErrorRetries": 0}
RATE_TWO: Final[dict[str, JsonValue]] = {"RateLimitErrorRetries": 2, "DefaultRetries": 0}
BAD_REQUEST_ONE: Final[dict[str, JsonValue]] = {"BadRequestErrorRetries": 1}

# group -> (provider, the label's fixed reply, the group's retry policy or None)
GROUPS: Final[dict[str, tuple[str, Reply, dict[str, JsonValue] | None]]] = {
    "bedrock-500": ("bedrock", bedrock_error(500, "InternalServerException", BEDROCK_INTERNAL_TEXT), UNAVAILABLE_TWO),
    "bedrock-500-internal": (
        "bedrock",
        bedrock_error(500, "InternalServerException", BEDROCK_INTERNAL_TEXT),
        INTERNAL_TWO,
    ),
    "bedrock-503": (
        "bedrock",
        bedrock_error(503, "ServiceUnavailableException", BEDROCK_UNAVAILABLE_TEXT),
        UNAVAILABLE_TWO,
    ),
    "gemini-overloaded": ("gemini", google_error(503, "UNAVAILABLE", GOOGLE_OVERLOADED_TEXT), INTERNAL_TWO),
    "gemini-overloaded-su": ("gemini", google_error(503, "UNAVAILABLE", GOOGLE_OVERLOADED_TEXT), UNAVAILABLE_TWO),
    "gemini-unavailable": ("gemini", google_error(503, "UNAVAILABLE", GOOGLE_UNAVAILABLE_TEXT), INTERNAL_TWO),
    "vertex-overloaded": ("vertex", google_error(503, "UNAVAILABLE", GOOGLE_OVERLOADED_TEXT), INTERNAL_TWO),
    "anthropic-529": ("anthropic", anthropic_overloaded(), INTERNAL_TWO),
    "anthropic-529-su": ("anthropic", anthropic_overloaded(), UNAVAILABLE_TWO),
    "anthropic-503": ("anthropic", anthropic_error(503, "api_error", "Service unavailable"), INTERNAL_TWO),
    "azure-500-generic": ("azure", openai_error(500, AZURE_GENERIC_500_TEXT, kind="server_error"), DEFAULT_TWO),
    "azure-500-internal": ("azure", openai_error(500, AZURE_INTERNAL_TEXT, kind="server_error"), DEFAULT_TWO),
    "openai-500": ("openai", openai_error(500, OPENAI_SERVER_TEXT, kind="server_error"), INTERNAL_TWO),
    "openai-503": ("openai", openai_error(503, OPENAI_UNAVAILABLE_TEXT, kind="server_error"), INTERNAL_TWO),
    "rl-bedrock-throttle": ("bedrock", bedrock_error(429, "ThrottlingException", BEDROCK_THROTTLE_TEXT), RATE_TWO),
    "rl-bedrock-not-ready": ("bedrock", bedrock_error(429, "ModelNotReadyException", BEDROCK_NOT_READY_TEXT), RATE_TWO),
    "rl-bedrock-throttle-on-400": (
        "bedrock",
        bedrock_error(400, "ThrottlingException", f"ThrottlingException: {BEDROCK_THROTTLE_TEXT}"),
        RATE_TWO,
    ),
    "rl-vertex-exhausted": ("vertex", google_error(429, "RESOURCE_EXHAUSTED", GOOGLE_EXHAUSTED_TEXT), RATE_TWO),
    "rl-vertex-quota-on-400": ("vertex", google_error(400, "FAILED_PRECONDITION", GOOGLE_QUOTA_TEXT), RATE_TWO),
    "rl-gemini-exhausted": ("gemini", google_error(429, "RESOURCE_EXHAUSTED", GOOGLE_EXHAUSTED_TEXT), RATE_TWO),
    "rl-anthropic": ("anthropic", anthropic_error(429, "rate_limit_error", ANTHROPIC_RATE_TEXT), RATE_TWO),
    "rl-azure": ("azure", azure_rate_limit({"retry-after": "1", "retry-after-ms": "1000"}), RATE_TWO),
    "rl-openai-quota": ("openai", openai_rate_limit("insufficient_quota"), RATE_TWO),
    "rl-openai-rate": ("openai", openai_rate_limit("rate_limit_exceeded"), RATE_TWO),
    "bedrock-403-default": (
        "bedrock",
        bedrock_error(403, "AccessDeniedException", BEDROCK_ACCESS_TEXT),
        {"DefaultRetries": 1, "AuthenticationErrorRetries": 0},
    ),
    "bedrock-403-plain": ("bedrock", bedrock_error(403, "AccessDeniedException", BEDROCK_ACCESS_TEXT), None),
    "vertex-403-plain": ("vertex", google_error(403, "PERMISSION_DENIED", GOOGLE_PERMISSION_TEXT), None),
    "bedrock-token": (
        "bedrock",
        bedrock_error(403, "UnrecognizedClientException", BEDROCK_TOKEN_TEXT),
        {"AuthenticationErrorRetries": 1, "DefaultRetries": 0},
    ),
    "vertex-400-plain": ("vertex", google_error(400, "INVALID_ARGUMENT", GOOGLE_INVALID_TEXT), None),
    "vertex-400-default": ("vertex", google_error(400, "INVALID_ARGUMENT", GOOGLE_INVALID_TEXT), {"DefaultRetries": 2}),
    "bedrock-validation": (
        "bedrock",
        bedrock_error(400, "ValidationException", BEDROCK_VALIDATION_TEXT),
        BAD_REQUEST_ONE,
    ),
    "cw-bedrock": ("bedrock", bedrock_error(400, "ValidationException", BEDROCK_TOO_LONG_TEXT), BAD_REQUEST_ONE),
    "cw-anthropic": (
        "anthropic",
        anthropic_error(400, "invalid_request_error", ANTHROPIC_TOO_LONG_TEXT),
        BAD_REQUEST_ONE,
    ),
    "cw-gemini": ("gemini", google_error(400, "INVALID_ARGUMENT", GOOGLE_TOO_LONG_TEXT), BAD_REQUEST_ONE),
    "cw-openai": (
        "openai",
        openai_error(400, OPENAI_TOO_LONG_TEXT, code="context_length_exceeded"),
        BAD_REQUEST_ONE,
    ),
    "anthropic-filter": (
        "anthropic",
        anthropic_error(400, "invalid_request_error", ANTHROPIC_FILTER_TEXT),
        {"ContentPolicyViolationErrorRetries": 1, "BadRequestErrorRetries": 0},
    ),
    "gemini-502": (
        "gemini",
        google_error(502, "UNAVAILABLE", GOOGLE_BAD_GATEWAY_TEXT),
        {"DefaultRetries": 1, "InternalServerErrorRetries": 0},
    ),
    "azure-502": (
        "azure",
        openai_error(502, "Bad gateway", kind="server_error"),
        {"DefaultRetries": 1, "InternalServerErrorRetries": 0},
    ),
}

# Lone deployments that fail once and then answer, for the success headers.
RECOVERING: Final = ("recover-anthropic", "recover-gemini", "recover-bedrock", "recover-azure", "recover-openai")


def _models(base: str) -> Sequence[dict[str, JsonValue]]:
    entries: Final[list[dict[str, JsonValue]]] = []  # mutable-ok: built once for the config
    for group, (provider, _, _) in GROUPS.items():
        entries.extend(pair_of(provider, group, group, base))
    for group in RECOVERING:
        provider: str = group.removeprefix("recover-")
        entries.append(pair_of(provider, group, group, base)[0])
    return entries


ROUTER_SETTINGS: Final[dict[str, JsonValue]] = {
    "num_retries": 0,
    "disable_cooldowns": True,
    "model_group_retry_policy": {group: policy for group, (_, _, policy) in GROUPS.items() if policy is not None},
}


@pytest.fixture(scope="module")
def rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Rig]:
    with reliab_rig(tmp_path_factory.mktemp("dec1-reliab-retry-classes"), _models, ROUTER_SETTINGS) as value:
        for group, (_, reply, _) in GROUPS.items():
            value.peer.always(f"{group}-a", reply)
            value.peer.always(f"{group}-b", reply)
        yield value


def test_a_bedrock_500_reads_the_service_unavailable_budget(rig: Rig) -> None:
    """Both Bedrock deployments answer 500 `InternalServerException`; the policy grants
    `ServiceUnavailableErrorRetries: 2` and `InternalServerErrorRetries: 0`.

    Bedrock's 500 is raised as `ServiceUnavailableError` (`exception_mapping_utils.py:1086`), so the peer must
    see 1 + 2 = 3 attempts and the caller HTTP 503 whose message carries `Internal failure`. A port that keyed
    the budget on the raw 500 would read the internal server budget and stop after one attempt.
    """
    response: Final = chat(rig.gateway, "bedrock-500")
    assert response.status_code == 503, response.text
    assert BEDROCK_INTERNAL_TEXT in error_message(response)
    assert rig.both("bedrock-500") == 3


def test_a_bedrock_500_ignores_the_internal_server_error_budget(rig: Rig) -> None:
    """Same Bedrock 500, but the policy grants `InternalServerErrorRetries: 2` and
    `ServiceUnavailableErrorRetries: 0`.

    The internal server budget does not apply to Bedrock's 500, so the peer must see exactly one attempt and
    the caller HTTP 503.
    """
    response: Final = chat(rig.gateway, "bedrock-500-internal")
    assert response.status_code == 503, response.text
    assert rig.both("bedrock-500-internal") == 1


def test_a_bedrock_503_service_unavailable_exception_reads_the_service_unavailable_budget(rig: Rig) -> None:
    """Both Bedrock deployments answer 503 `ServiceUnavailableException` and the policy grants
    `ServiceUnavailableErrorRetries: 2`.

    The peer must see 3 attempts and the caller HTTP 503 carrying `Service is temporarily unavailable.`.
    """
    response: Final = chat(rig.gateway, "bedrock-503")
    assert response.status_code == 503, response.text
    assert BEDROCK_UNAVAILABLE_TEXT in error_message(response)
    assert rig.both("bedrock-503") == 3


def test_googles_503_overload_text_reads_the_internal_server_error_budget(rig: Rig) -> None:
    """Both Gemini deployments answer 503 `UNAVAILABLE` with `The model is overloaded.`; the policy grants
    `InternalServerErrorRetries: 2`, `ServiceUnavailableErrorRetries: 0`.

    The overload text is matched before the status (`exception_mapping_utils.py:1405`) and raised as
    `InternalServerError`, so the peer must see 3 attempts and the caller HTTP 500, not 503.
    """
    response: Final = chat(rig.gateway, "gemini-overloaded")
    assert response.status_code == 500, response.text
    assert "The model is overloaded" in error_message(response)
    assert rig.both("gemini-overloaded") == 3


def test_googles_503_overload_text_ignores_the_service_unavailable_budget(rig: Rig) -> None:
    """Same Gemini overload 503 with `ServiceUnavailableErrorRetries: 2` and `InternalServerErrorRetries: 0`.

    Although Google answered 503, the service unavailable budget does not apply: one attempt, caller 500.
    """
    response: Final = chat(rig.gateway, "gemini-overloaded-su")
    assert response.status_code == 500, response.text
    assert rig.both("gemini-overloaded-su") == 1


def test_a_google_503_with_other_text_is_not_covered_by_the_internal_server_error_budget(rig: Rig) -> None:
    """Both Gemini deployments answer 503 `UNAVAILABLE` with `The service is currently unavailable.`; the
    policy grants `InternalServerErrorRetries: 2` and `ServiceUnavailableErrorRetries: 0`.

    Without the overload text the status branch raises `ServiceUnavailableError`, so there is one attempt and
    the caller gets HTTP 503.
    """
    response: Final = chat(rig.gateway, "gemini-unavailable")
    assert response.status_code == 503, response.text
    assert rig.both("gemini-unavailable") == 1


def test_a_vertex_overload_503_reads_the_internal_server_error_budget_like_gemini(rig: Rig) -> None:
    """Both Vertex deployments answer 503 with the overload text; the policy grants
    `InternalServerErrorRetries: 2`.

    Vertex and Gemini share `_map_vertex_exception`, so Vertex must behave the same: 3 attempts, caller 500.
    """
    response: Final = chat(rig.gateway, "vertex-overloaded")
    assert response.status_code == 500, response.text
    assert rig.both("vertex-overloaded") == 3


def test_an_anthropic_529_reads_the_internal_server_error_budget(rig: Rig) -> None:
    """Both Anthropic deployments answer 529 `overloaded_error`; the policy grants
    `InternalServerErrorRetries: 2` and `ServiceUnavailableErrorRetries: 0`.

    A 529 is raised as `InternalServerError` with `provider_status_code: 529`
    (`exception_mapping_utils.py:637`), so the peer must see 3 attempts. The chat caller still gets HTTP 503
    with `error.code: server_is_overloaded`, the shape OpenAI SDK back-off understands.
    """
    response: Final = chat(rig.gateway, "anthropic-529")
    assert response.status_code == 503, response.text
    assert error_of(response)["code"] == "server_is_overloaded"
    assert rig.both("anthropic-529") == 3


def test_an_anthropic_529_ignores_the_service_unavailable_budget_although_the_caller_sees_503(rig: Rig) -> None:
    """Same Anthropic 529 with `ServiceUnavailableErrorRetries: 2` and `InternalServerErrorRetries: 0`.

    The caller-facing 503 is applied after routing, so the router reads the internal server budget of 0:
    exactly one attempt, caller HTTP 503.
    """
    response: Final = chat(rig.gateway, "anthropic-529-su")
    assert response.status_code == 503, response.text
    assert rig.both("anthropic-529-su") == 1


def test_an_anthropic_503_is_a_service_unavailable_error_not_an_overload(rig: Rig) -> None:
    """Both Anthropic deployments answer 503 `api_error` `Service unavailable` with
    `InternalServerErrorRetries: 2` and `ServiceUnavailableErrorRetries: 0`.

    A 503 without the overload text maps to `ServiceUnavailableError` (`exception_mapping_utils.py:704`), so
    the peer sees one attempt and the caller HTTP 503.
    """
    response: Final = chat(rig.gateway, "anthropic-503")
    assert response.status_code == 503, response.text
    assert rig.both("anthropic-503") == 1


def test_an_azure_500_without_the_internal_server_error_text_reads_default_retries(rig: Rig) -> None:
    """Both Azure deployments answer 500 `The service encountered a transient error.`; the policy grants
    `DefaultRetries: 2` and `InternalServerErrorRetries: 0`.

    `_map_azure_exception` has no 500 status branch, so the error is a bare `APIError` with status 500 that no
    named budget covers: the peer must see 3 attempts and the caller HTTP 500.
    """
    response: Final = chat(rig.gateway, "azure-500-generic")
    assert response.status_code == 500, response.text
    assert rig.both("azure-500-generic") == 3


def test_an_azure_500_with_the_internal_server_error_text_reads_the_internal_server_budget(rig: Rig) -> None:
    """Both Azure deployments answer 500 `Internal server error` with the same policy (`DefaultRetries: 2`,
    `InternalServerErrorRetries: 0`).

    The text branch raises `InternalServerError` (`exception_mapping_utils.py:2108`), whose own budget is 0, so
    the peer sees one attempt. The two Azure 500s differ only by message text.
    """
    response: Final = chat(rig.gateway, "azure-500-internal")
    assert response.status_code == 500, response.text
    assert rig.both("azure-500-internal") == 1


def test_an_openai_500_reads_the_internal_server_error_budget(rig: Rig) -> None:
    """Both OpenAI deployments answer 500 `server_error`; the policy grants `InternalServerErrorRetries: 2`.

    OpenAI's mapping has a 500 branch, unlike Azure's, so the peer sees 3 attempts and the caller HTTP 500.
    """
    response: Final = chat(rig.gateway, "openai-500")
    assert response.status_code == 500, response.text
    assert rig.both("openai-500") == 3


def test_an_openai_503_is_not_covered_by_the_internal_server_error_budget(rig: Rig) -> None:
    """Both OpenAI deployments answer 503 `The engine is currently overloaded`; the policy grants
    `InternalServerErrorRetries: 2` and `ServiceUnavailableErrorRetries: 0`.

    OpenAI's 503 maps by status to `ServiceUnavailableError` even with an overload message, so one attempt
    and caller HTTP 503. Contrast with Google, whose overload text turns a 503 into a 500.
    """
    response: Final = chat(rig.gateway, "openai-503")
    assert response.status_code == 503, response.text
    assert rig.both("openai-503") == 1


@pytest.mark.parametrize(
    "group",
    [
        "rl-bedrock-throttle",
        "rl-bedrock-not-ready",
        "rl-bedrock-throttle-on-400",
        "rl-vertex-exhausted",
        "rl-vertex-quota-on-400",
        "rl-gemini-exhausted",
        "rl-anthropic",
        "rl-azure",
        "rl-openai-quota",
        "rl-openai-rate",
    ],
)
def test_every_providers_throttle_reads_the_rate_limit_budget_and_ends_in_429(rig: Rig, group: str) -> None:
    """Both deployments of the group answer the provider's own throttle, and the policy grants
    `RateLimitErrorRetries: 2` with `DefaultRetries: 0`.

    Every one of these is a `RateLimitError`: Bedrock `ThrottlingException` and `ModelNotReadyException`
    (even a `ThrottlingException` text on a 400, matched before the status), Google `RESOURCE_EXHAUSTED` and a
    400 whose text is `Quota exceeded for`, Anthropic and Azure 429, and both OpenAI 429 codes. OpenAI's
    `insufficient_quota` is not told apart from `rate_limit_exceeded`, so a billing failure is retried like a
    per-minute limit. The peer must see 3 attempts and the caller HTTP 429.
    """
    response: Final = chat(rig.gateway, group)
    assert response.status_code == 429, response.text
    assert rig.both(group) == 3


def test_a_bedrock_access_denied_reads_default_retries_because_permission_denied_has_no_budget(rig: Rig) -> None:
    """Both Bedrock deployments answer 403 `AccessDeniedException`; the policy grants `DefaultRetries: 1`
    and `AuthenticationErrorRetries: 0`.

    `PermissionDeniedError` has no field of its own in `RetryPolicy` (`retry.rs:48`), so `DefaultRetries`
    applies: 2 attempts and caller HTTP 403.
    """
    response: Final = chat(rig.gateway, "bedrock-403-default")
    assert response.status_code == 403, response.text
    assert rig.both("bedrock-403-default") == 2


def test_a_bedrock_access_denied_is_retried_on_the_sibling_without_any_policy(rig: Rig) -> None:
    """Both Bedrock deployments answer 403 `AccessDeniedException`; no policy, the request sends
    `num_retries: 2`.

    `should_retry` exempts 401 and 403 from the non-retryable statuses when the group has more than one
    deployment (`retry.rs:85`), so the peer sees 3 attempts and the caller HTTP 403.
    """
    response: Final = chat(rig.gateway, "bedrock-403-plain", num_retries=2)
    assert response.status_code == 403, response.text
    assert rig.both("bedrock-403-plain") == 3


def test_a_vertex_permission_denied_is_a_400_and_is_never_retried(rig: Rig) -> None:
    """Both Vertex deployments answer 403 `PERMISSION_DENIED` (the body carries `"code": 403`); no policy,
    `num_retries: 2` on the request.

    `_map_vertex_exception` turns a 403 whose text names 403 into `BadRequestError` (`exception_mapping_utils.py:1331`),
    so unlike Bedrock's 403 it is not retried: one attempt, and the caller gets HTTP 400.
    """
    response: Final = chat(rig.gateway, "vertex-403-plain", num_retries=2)
    assert response.status_code == 400, response.text
    assert rig.both("vertex-403-plain") == 1


def test_a_bedrock_invalid_security_token_reads_the_authentication_budget(rig: Rig) -> None:
    """Both Bedrock deployments answer 403 `UnrecognizedClientException` with `The security token included
    in the request is invalid.`; the policy grants `AuthenticationErrorRetries: 1`, `DefaultRetries: 0`.

    The text maps to `AuthenticationError` (401) despite the 403 status, so the peer sees 2 attempts and the
    caller HTTP 401.
    """
    response: Final = chat(rig.gateway, "bedrock-token")
    assert response.status_code == 401, response.text
    assert rig.both("bedrock-token") == 2


def test_a_google_invalid_argument_is_not_retried_without_a_policy(rig: Rig) -> None:
    """Both Vertex deployments answer 400 `INVALID_ARGUMENT`; no policy, `num_retries: 2` on the request.

    A 400 is never retried by `should_retry`: one attempt, caller HTTP 400 carrying Google's message.
    """
    response: Final = chat(rig.gateway, "vertex-400-plain", num_retries=2)
    assert response.status_code == 400, response.text
    assert "Invalid value at" in error_message(response)
    assert rig.both("vertex-400-plain") == 1


def test_default_retries_in_a_policy_makes_the_router_retry_a_google_400(rig: Rig) -> None:
    """Both Vertex deployments answer 400 `INVALID_ARGUMENT`; the group's policy is only `DefaultRetries: 2`.

    Once a policy field applies, `should_retry_this_error` is skipped (`call.rs:1015`), so even a 400 is
    retried: the peer sees 3 attempts and the caller HTTP 400. A port that kept the status check in front of
    the policy would make one attempt.
    """
    response: Final = chat(rig.gateway, "vertex-400-default")
    assert response.status_code == 400, response.text
    assert rig.both("vertex-400-default") == 3


def test_a_bedrock_validation_exception_reads_the_bad_request_budget(rig: Rig) -> None:
    """Both Bedrock deployments answer 400 `ValidationException` (`Malformed input request`); the policy
    grants `BadRequestErrorRetries: 1`.

    The peer sees 2 attempts and the caller HTTP 400 carrying the AWS text.
    """
    response: Final = chat(rig.gateway, "bedrock-validation")
    assert response.status_code == 400, response.text
    assert "Malformed input request" in error_message(response)
    assert rig.both("bedrock-validation") == 2


@pytest.mark.parametrize("group", ["cw-bedrock", "cw-anthropic", "cw-gemini", "cw-openai"])
def test_a_context_window_error_reads_the_bad_request_budget_when_no_context_window_fallback_exists(
    rig: Rig, group: str
) -> None:
    """Both deployments answer the provider's context overflow (Bedrock `Input is too long`, Anthropic
    `prompt is too long`, Google `exceeds the maximum number of tokens allowed`, OpenAI
    `context_length_exceeded`); the policy grants `BadRequestErrorRetries: 1` and there are no
    `context_window_fallbacks`.

    `ContextWindowExceededError` has no budget of its own and is a `BadRequestError`, so the peer sees 2
    attempts and the caller HTTP 400 whose message names `ContextWindowExceededError` or the provider text.
    """
    response: Final = chat(rig.gateway, group)
    assert response.status_code == 400, response.text
    assert rig.both(group) == 2


def test_an_anthropic_content_filter_refusal_reads_the_content_policy_budget(rig: Rig) -> None:
    """Both Anthropic deployments answer 400 `Output blocked by content filtering policy`; the policy grants
    `ContentPolicyViolationErrorRetries: 1` and `BadRequestErrorRetries: 0`.

    The text maps to `ContentPolicyViolationError`, the most specific class, so its own budget wins over the
    bad request one: 2 attempts and caller HTTP 400.
    """
    response: Final = chat(rig.gateway, "anthropic-filter")
    assert response.status_code == 400, response.text
    assert rig.both("anthropic-filter") == 2


def test_a_google_502_is_a_connection_error_that_reads_default_retries(rig: Rig) -> None:
    """Both Gemini deployments answer 502; the policy grants `DefaultRetries: 1` and
    `InternalServerErrorRetries: 0`.

    `_map_vertex_exception` raises `APIConnectionError` for a 502 (`exception_mapping_utils.py:1487`), whose
    status is 500: the peer sees 2 attempts and the caller gets HTTP 500, not 502.
    """
    response: Final = chat(rig.gateway, "gemini-502")
    assert response.status_code == 500, response.text
    assert rig.both("gemini-502") == 2


def test_an_azure_502_is_a_bad_gateway_error_that_reads_default_retries_and_stays_502(rig: Rig) -> None:
    """Both Azure deployments answer 502 with the same policy as the Google 502 group.

    Azure maps 502 to `BadGatewayError`, which has no budget of its own: 2 attempts, and the caller gets
    HTTP 502. The same upstream status reaches the caller as 500 from Google and 502 from Azure.
    """
    response: Final = chat(rig.gateway, "azure-502")
    assert response.status_code == 502, response.text
    assert rig.both("azure-502") == 2


@pytest.mark.parametrize(
    ("group", "failure"),
    [
        ("recover-anthropic", anthropic_overloaded()),
        ("recover-gemini", google_error(503, "UNAVAILABLE", GOOGLE_OVERLOADED_TEXT)),
        ("recover-bedrock", bedrock_error(429, "ThrottlingException", BEDROCK_THROTTLE_TEXT)),
        ("recover-azure", azure_rate_limit()),
        ("recover-openai", openai_error(500, OPENAI_SERVER_TEXT, kind="server_error")),
    ],
)
def test_a_provider_failure_that_clears_on_retry_returns_200_with_the_retry_headers(
    rig: Rig, group: str, failure: Reply
) -> None:
    """The group has one deployment that answers the provider's transient failure once and then a 200; the
    request sends `num_retries: 1`.

    The caller must get HTTP 200 with `served <label>`, `x-litellm-attempted-retries: 1`,
    `x-litellm-max-retries: 1` and `x-litellm-model-id` naming the deployment (`dec1-reliab-<label>`). The
    peer sees exactly 2 attempts.
    """
    label: Final = f"{group}-a"
    rig.peer.plan(label, failure)
    response: Final = chat(rig.gateway, group, num_retries=1)
    assert response.status_code == 200, response.text
    assert content_of(response) == f"served {label}"
    assert response.headers["x-litellm-attempted-retries"] == "1"
    assert response.headers["x-litellm-max-retries"] == "1"
    assert response.headers["x-litellm-model-id"] == f"dec1-reliab-{label}"
    assert rig.peer.count(label) == 2
