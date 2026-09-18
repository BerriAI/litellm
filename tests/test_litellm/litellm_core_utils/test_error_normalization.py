import httpx
import pytest

import litellm
from litellm.exceptions import MidStreamFallbackError
from litellm.litellm_core_utils.error_normalization import normalize_error
from litellm.litellm_core_utils.litellm_logging import StandardLoggingPayloadSetup
from litellm.proxy._types import ProxyErrorTypes, ProxyException

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


def test_no_exception_has_no_normalized_error() -> None:
    assert StandardLoggingPayloadSetup.get_error_information(None)["normalized_error"] is None


def test_unknown_exception_falls_back_to_status_then_unclassified() -> None:
    assert normalize_error(Exception("x"), "429", "x") == "429_RATE_LIMIT_EXCEEDED"
    assert normalize_error(Exception("x"), "", "x") == "UNCLASSIFIED"


def test_normalized_error_never_embeds_dynamic_parts() -> None:
    exc = _proxy_exc(
        "No team has access to anthropic.claude-sonnet-4-5", ProxyErrorTypes.team_model_access_denied.value, 401
    )
    info = StandardLoggingPayloadSetup.get_error_information(exc)
    assert info["error_message"] == "No team has access to anthropic.claude-sonnet-4-5"
    assert "claude" not in (info["normalized_error"] or "")
