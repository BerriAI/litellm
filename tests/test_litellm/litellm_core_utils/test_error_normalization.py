import time

import httpx
import pytest

import litellm
from litellm.exceptions import MidStreamFallbackError
from litellm.litellm_core_utils.error_normalization import normalize_error
from litellm.litellm_core_utils.litellm_logging import StandardLoggingPayloadSetup
from litellm.proxy._types import ProxyErrorTypes, ProxyException
from litellm.types.router import RouterErrors

_RESPONSE = httpx.Response(status_code=500, request=httpx.Request("POST", "https://example.invalid"))


def _proxy_exc(message: str, error_type: str, code: int) -> ProxyException:
    return ProxyException(message=message, type=error_type, param=None, code=code)


@pytest.mark.parametrize(
    ("messages", "expected"),
    [
        (
            (
                _proxy_exc("Rate limit exceeded for team X. Reset at 10:01", "rate_limit_error", 429),
                _proxy_exc("Rate limit exceeded for team Y. Reset at 10:02", "rate_limit_error", 429),
            ),
            "429_RATE_LIMIT_EXCEEDED",
        ),
        (
            (
                litellm.BudgetExceededError(current_cost=3501.85, max_budget=3500),
                _proxy_exc(
                    "User=abc, Current cost=1000.03, Max budget=1000", ProxyErrorTypes.budget_exceeded.value, 400
                ),
                litellm.RateLimitError(
                    "budget",
                    llm_provider="openai",
                    model="gpt",
                    rate_limit_type=litellm.exceptions.RateLimitType.BUDGET,
                ),
            ),
            "429_BUDGET_EXCEEDED",
        ),
        (
            (
                _proxy_exc("Token Expired", ProxyErrorTypes.expired_key.value, 401),
                _proxy_exc("Malformed API Key", ProxyErrorTypes.auth_error.value, 401),
                litellm.AuthenticationError("Signature verification failed", llm_provider="azure", model="gpt"),
            ),
            "401_AUTHENTICATION_FAILED",
        ),
        (
            (
                _proxy_exc("No team has access to gpt-5.5-mini", ProxyErrorTypes.team_model_access_denied.value, 401),
                _proxy_exc("key not allowed to access claude", ProxyErrorTypes.key_model_access_denied.value, 401),
                ValueError(
                    "Not allowed to access model due to tags configuration. Passed model=gpt-5.5 and tags=['team-a']"
                ),
            ),
            "403_MODEL_ACCESS_DENIED",
        ),
        (
            (
                _proxy_exc("Missing required parameter: messages", ProxyErrorTypes.bad_request_error.value, 400),
                litellm.BadRequestError("Missing required parameter: input", llm_provider="openai", model="gpt"),
            ),
            "400_MISSING_REQUIRED_PARAMETER",
        ),
        (
            (
                litellm.ContextWindowExceededError(
                    "1002823 tokens > 1000000 maximum", model="g", llm_provider="vertex"
                ),
                litellm.BadRequestError("Input is too long for requested model", llm_provider="anthropic", model="c"),
            ),
            "400_CONTEXT_WINDOW_EXCEEDED",
        ),
        (
            (
                litellm.NotFoundError("Response id xxx not found", llm_provider="openai", model="gpt"),
                _proxy_exc("No vector store found with id abc", ProxyErrorTypes.not_found_error.value, 404),
            ),
            "404_RESOURCE_NOT_FOUND",
        ),
        (
            (
                litellm.APIConnectionError("Connection error", llm_provider="openai", model="gpt"),
                litellm.InternalServerError("TransferEncodingError", llm_provider="openai", model="gpt"),
                litellm.APIError(500, "Response payload is not completed", llm_provider="openai", model="gpt"),
                httpx.RemoteProtocolError(
                    "peer closed connection without sending complete message body (incomplete chunked read)"
                ),
            ),
            "500_PROVIDER_CONNECTION_ERROR",
        ),
        (
            (
                litellm.ServiceUnavailableError("server_is_overloaded", llm_provider="anthropic", model="c"),
                litellm.InternalServerError(
                    "Bedrock is unable to process your request", llm_provider="bedrock", model="c"
                ),
                litellm.APIError(529, "Overloaded", llm_provider="anthropic", model="c"),
            ),
            "503_PROVIDER_OVERLOADED",
        ),
        (
            (
                litellm.InternalServerError(
                    "The server had an error while processing your request", llm_provider="openai", model="gpt"
                ),
                litellm.APIError(500, "server_error", llm_provider="openai", model="gpt"),
            ),
            "500_PROVIDER_INTERNAL_ERROR",
        ),
        (
            (
                _proxy_exc("No fallback model group found for gpt-5.6", "internal_server_error", 500),
                _proxy_exc("No fallback model group found for claude-46-sonnet", "internal_server_error", 500),
            ),
            "500_ROUTER_NO_FALLBACK",
        ),
        (
            (
                _proxy_exc("Error doing the fallback: RateLimitError", "internal_server_error", 500),
                MidStreamFallbackError(
                    "stream died", model="gpt", llm_provider="openai", original_exception=ValueError("boom")
                ),
            ),
            "500_ROUTER_FALLBACK_FAILURE",
        ),
        (
            (
                TypeError("cannot pickle '_thread.RLock' object"),
                RuntimeError("dictionary changed size during iteration"),
                TypeError("'NoneType' object is not iterable"),
            ),
            "500_INTERNAL_STATE_ERROR",
        ),
        (
            (
                litellm.Timeout("Timeout on reading data from socket", model="gpt", llm_provider="openai"),
                litellm.APIError(504, "Request timed out", llm_provider="openai", model="gpt"),
            ),
            "408_UPSTREAM_TIMEOUT",
        ),
        (
            (
                _proxy_exc("500: Upstream passthrough request failed", "internal_server_error", 500),
                _proxy_exc("503: Upstream passthrough request failed", "internal_server_error", 503),
            ),
            "500_UPSTREAM_PASSTHROUGH",
        ),
        (
            (
                _proxy_exc("OCR is not supported for provider openai", "internal_server_error", 500),
                NotImplementedError("rerank"),
            ),
            "500_UNSUPPORTED_OPERATION",
        ),
    ],
)
def test_variants_of_one_failure_share_a_normalized_error(messages: tuple[Exception, ...], expected: str) -> None:
    normalized = {StandardLoggingPayloadSetup.get_error_information(exc)["normalized_error"] for exc in messages}
    assert normalized == {expected}


def test_normalize_error_passthrough_prefix_wins_over_upstream_body_text() -> None:
    from fastapi import HTTPException

    for detail in (
        'Upstream passthrough request failed with status 400: {"error": {"message": "no deployments available for this model"}}',
        'Upstream passthrough request failed with status 400: {"error": {"message": "max budget reached"}}',
    ):
        exc = HTTPException(status_code=400, detail=detail)
        message = f"400: {detail}"
        assert normalize_error(exc, "400", message) == "500_UPSTREAM_PASSTHROUGH", message


def test_normalize_error_passthrough_prefix_wins_over_quota_and_budget_wording() -> None:
    from fastapi import HTTPException

    detail = (
        'Upstream passthrough request failed with status 429: {"error": {"message": '
        '"You exceeded your current quota, please check your plan and billing details. Budget spent"}}'
    )
    exc = HTTPException(status_code=429, detail=detail)
    assert normalize_error(exc, "429", f"429: {detail}") == "500_UPSTREAM_PASSTHROUGH"


def test_router_no_healthy_deployment_wording_clusters_as_no_healthy_deployments() -> None:
    for message in (RouterErrors.no_healthy_deployments.value, "No healthy deployments found."):
        exc = litellm.BadRequestError(message, llm_provider="openai", model="gpt-4o")
        assert normalize_error(exc, "400", message) == "429_NO_HEALTHY_DEPLOYMENTS", message


def test_provider_budget_routing_wording_clusters_as_budget_exceeded() -> None:
    message = RouterErrors.no_deployments_with_provider_budget_routing.value
    exc = litellm.BadRequestError(message, llm_provider="openai", model="gpt-4o")
    assert normalize_error(exc, "400", message) == "429_BUDGET_EXCEEDED"


def test_router_fallback_wording_does_not_hide_the_wrapped_exception_class() -> None:
    provider_message = "litellm.AuthenticationError: OpenAIException - Incorrect API key provided"
    exc = litellm.AuthenticationError(
        provider_message + "\nNo fallback model group found for lookup_groups=['x']",
        llm_provider="openai",
        model="gpt",
    )
    assert normalize_error(exc, "401", str(exc)) == "401_AUTHENTICATION_FAILED"
    wrapped = litellm.AuthenticationError(
        "Error doing the fallback: " + provider_message, llm_provider="openai", model="gpt"
    )
    assert normalize_error(wrapped, "401", str(wrapped)) == "401_AUTHENTICATION_FAILED"


def test_parameter_length_error_is_not_a_context_window_error() -> None:
    exc = litellm.BadRequestError("string too long: 'user' max 64 chars", llm_provider="openai", model="gpt")
    assert StandardLoggingPayloadSetup.get_error_information(exc)["normalized_error"] == "400_INVALID_REQUEST"


def test_no_exception_has_no_normalized_error() -> None:
    assert StandardLoggingPayloadSetup.get_error_information(None)["normalized_error"] is None


def test_unknown_exception_falls_back_to_status_then_unclassified() -> None:
    assert normalize_error(Exception("x"), "429", "x") == "429_RATE_LIMIT_EXCEEDED"
    assert normalize_error(Exception("x"), "", "x") == "UNCLASSIFIED"


def test_budget_exceeded_error_with_custom_wording_is_still_a_budget_error() -> None:
    exc = litellm.BudgetExceededError(current_cost=2.0, max_budget=1.0, message="Spending cap reached for key")
    assert StandardLoggingPayloadSetup.get_error_information(exc)["normalized_error"] == "429_BUDGET_EXCEEDED"


def test_every_model_access_denied_proxy_type_shares_one_cluster() -> None:
    access_denied_types = tuple(t for t in ProxyErrorTypes if t.value.endswith("_model_access_denied"))
    assert len(access_denied_types) >= 6, access_denied_types
    codes = {normalize_error(_proxy_exc("denied", t.value, 403), "403", "denied") for t in access_denied_types}
    assert codes == {"403_MODEL_ACCESS_DENIED"}, codes


def test_non_string_type_attribute_falls_through_to_status() -> None:
    class _OddType(Exception):
        type = {"kind": "odd"}

    assert normalize_error(_OddType("odd"), "500", "odd") == "500_PROVIDER_INTERNAL_ERROR"


def test_normalized_error_never_embeds_dynamic_parts() -> None:
    exc = _proxy_exc(
        "No team has access to anthropic.claude-sonnet-4-5", ProxyErrorTypes.team_model_access_denied.value, 401
    )
    info = StandardLoggingPayloadSetup.get_error_information(exc)
    assert info["error_message"] == "No team has access to anthropic.claude-sonnet-4-5"
    assert "claude" not in (info["normalized_error"] or "")


def test_repeated_exceeded_in_a_288kb_message_classifies_in_linear_time() -> None:
    model = ("exceeded " * 32_000)[:288_000]
    message = (
        f"/chat/completions: Invalid model name passed in model={model}. Call `/v1/models` to view available models"
    )
    exc = litellm.BadRequestError(message=message, model="unknown-model", llm_provider="openai")
    started = time.perf_counter()
    code = normalize_error(exc, "400", message)
    elapsed = time.perf_counter() - started
    assert code == "400_INVALID_REQUEST", code
    assert elapsed < 1.0, f"normalize_error took {elapsed:.2f}s on a 288 KB message"


@pytest.mark.parametrize(
    "message",
    [
        "ExceededBudget: User=abc over budget. Spend=12.5, Budget=10.0",
        "Exceeded budget for provider openai: 105.2 >= 100.0",
        "LiteLLM Team: team-1, exceeded budget for model=gpt-4o-mini",
        "ExceededBudget: Key over 1d budget. Spend=3.0, Budget=2.0",
        "Budget has been exceeded! Key=sk-... Current cost: 11.0, Max budget: 10.0",
        "EXCEEDED " + "x" * 65 + " BuDgEt",
    ],
)
def test_real_budget_wordings_still_cluster_as_budget_exceeded(message: str) -> None:
    assert normalize_error(Exception(message), "400", message) == "429_BUDGET_EXCEEDED"


@pytest.mark.parametrize("message", ["budget then exceeded", "exceeded the limit\nbudget unaffected", "exceededbudge"])
def test_exceeded_without_a_following_budget_on_the_same_line_is_not_budget(message: str) -> None:
    assert normalize_error(Exception(message), "400", message) == "400_INVALID_REQUEST"
