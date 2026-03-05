"""
Test that all custom exception classes can be pickled and unpickled.

This is critical for compatibility with concurrent.futures.ProcessPoolExecutor,
which pickles exceptions raised in worker processes to send them back to the
main process.

Fixes https://github.com/BerriAI/litellm/issues/22812
"""

import pickle

import httpx
import pytest

from litellm.exceptions import (
    APIConnectionError,
    APIError,
    APIResponseValidationError,
    AuthenticationError,
    BadGatewayError,
    BadRequestError,
    BudgetExceededError,
    ContentPolicyViolationError,
    ContextWindowExceededError,
    InternalServerError,
    JSONSchemaValidationError,
    NotFoundError,
    OpenAIError,
    PermissionDeniedError,
    RateLimitError,
    RejectedRequestError,
    ServiceUnavailableError,
    Timeout,
    UnprocessableEntityError,
    UnsupportedParamsError,
)


class TestExceptionPickle:
    """Verify that all litellm exception classes survive pickle round-trips."""

    def test_rate_limit_error_pickle(self):
        """Reproduce the exact scenario from issue #22812."""
        err = RateLimitError(
            message="rate limited",
            llm_provider="anthropic",
            model="claude-3-sonnet",
            response=httpx.Response(
                429, request=httpx.Request("POST", "http://x")
            ),
        )
        restored = pickle.loads(pickle.dumps(err))
        assert "rate limited" in restored.message
        assert restored.llm_provider == "anthropic"
        assert restored.model == "claude-3-sonnet"
        assert restored.status_code == 429

    def test_authentication_error_pickle(self):
        err = AuthenticationError(
            message="invalid key", llm_provider="openai", model="gpt-4"
        )
        restored = pickle.loads(pickle.dumps(err))
        assert "invalid key" in restored.message
        assert restored.llm_provider == "openai"
        assert restored.model == "gpt-4"

    def test_bad_request_error_pickle(self):
        err = BadRequestError(
            message="bad input", model="gpt-4", llm_provider="openai"
        )
        restored = pickle.loads(pickle.dumps(err))
        assert "bad input" in restored.message

    def test_not_found_error_pickle(self):
        err = NotFoundError(
            message="model not found", model="gpt-8", llm_provider="openai"
        )
        restored = pickle.loads(pickle.dumps(err))
        assert "model not found" in restored.message

    def test_timeout_pickle(self):
        err = Timeout(
            message="timed out", model="gpt-4", llm_provider="openai"
        )
        restored = pickle.loads(pickle.dumps(err))
        assert "timed out" in restored.message

    def test_permission_denied_error_pickle(self):
        err = PermissionDeniedError(
            message="denied", llm_provider="openai", model="gpt-4"
        )
        restored = pickle.loads(pickle.dumps(err))
        assert "denied" in restored.message

    def test_service_unavailable_error_pickle(self):
        err = ServiceUnavailableError(
            message="unavailable", llm_provider="openai", model="gpt-4"
        )
        restored = pickle.loads(pickle.dumps(err))
        assert "unavailable" in restored.message

    def test_bad_gateway_error_pickle(self):
        err = BadGatewayError(
            message="bad gateway", llm_provider="openai", model="gpt-4"
        )
        restored = pickle.loads(pickle.dumps(err))
        assert "bad gateway" in restored.message

    def test_internal_server_error_pickle(self):
        err = InternalServerError(
            message="server error", llm_provider="openai", model="gpt-4"
        )
        restored = pickle.loads(pickle.dumps(err))
        assert "server error" in restored.message

    def test_api_error_pickle(self):
        err = APIError(
            status_code=500,
            message="api error",
            llm_provider="openai",
            model="gpt-4",
        )
        restored = pickle.loads(pickle.dumps(err))
        assert "api error" in restored.message
        assert restored.status_code == 500

    def test_api_connection_error_pickle(self):
        err = APIConnectionError(
            message="conn error", llm_provider="openai", model="gpt-4"
        )
        restored = pickle.loads(pickle.dumps(err))
        assert "conn error" in restored.message

    def test_api_response_validation_error_pickle(self):
        err = APIResponseValidationError(
            message="validation error", llm_provider="openai", model="gpt-4"
        )
        restored = pickle.loads(pickle.dumps(err))
        assert "validation error" in restored.message

    def test_context_window_exceeded_error_pickle(self):
        err = ContextWindowExceededError(
            message="too long", model="gpt-4", llm_provider="openai"
        )
        restored = pickle.loads(pickle.dumps(err))
        assert "too long" in restored.message

    def test_content_policy_violation_error_pickle(self):
        err = ContentPolicyViolationError(
            message="policy violation", model="gpt-4", llm_provider="openai"
        )
        restored = pickle.loads(pickle.dumps(err))
        assert "policy violation" in restored.message

    def test_unprocessable_entity_error_pickle(self):
        err = UnprocessableEntityError(
            message="unprocessable", model="gpt-4", llm_provider="openai"
        )
        restored = pickle.loads(pickle.dumps(err))
        assert "unprocessable" in restored.message

    def test_openai_error_pickle(self):
        err = OpenAIError()
        restored = pickle.loads(pickle.dumps(err))
        assert restored.llm_provider == "openai"

    def test_unsupported_params_error_pickle(self):
        err = UnsupportedParamsError(message="unsupported param")
        restored = pickle.loads(pickle.dumps(err))
        assert "unsupported param" in restored.message

    def test_json_schema_validation_error_pickle(self):
        err = JSONSchemaValidationError(
            model="gpt-4",
            llm_provider="openai",
            raw_response="bad response",
            schema="{}",
        )
        restored = pickle.loads(pickle.dumps(err))
        assert restored.raw_response == "bad response"
        assert restored.schema == "{}"

    def test_budget_exceeded_error_pickle(self):
        err = BudgetExceededError(current_cost=10.0, max_budget=5.0)
        restored = pickle.loads(pickle.dumps(err))
        assert restored.current_cost == 10.0
        assert restored.max_budget == 5.0

    def test_rejected_request_error_pickle(self):
        err = RejectedRequestError(
            message="rejected",
            model="gpt-4",
            llm_provider="openai",
            request_data={"prompt": "test"},
        )
        restored = pickle.loads(pickle.dumps(err))
        assert "rejected" in restored.message

    def test_metadata_preserved_after_pickle(self):
        """Verify that optional metadata (debug info, retries) survives pickling."""
        err = RateLimitError(
            message="rate limited",
            llm_provider="anthropic",
            model="claude-3-sonnet",
            litellm_debug_info="some debug info",
            max_retries=3,
            num_retries=1,
        )
        restored = pickle.loads(pickle.dumps(err))
        assert restored.litellm_debug_info == "some debug info"
        assert restored.max_retries == 3
        assert restored.num_retries == 1

    def test_message_not_double_prefixed(self):
        """Ensure the message prefix is not duplicated after pickling."""
        err = RateLimitError(
            message="rate limited",
            llm_provider="anthropic",
            model="claude-3-sonnet",
        )
        restored = pickle.loads(pickle.dumps(err))
        # The message should contain exactly one "litellm.RateLimitError:" prefix
        assert restored.message.count("litellm.RateLimitError:") == 1

    def test_isinstance_preserved_after_pickle(self):
        """Verify that isinstance checks still work after pickling."""
        err = RateLimitError(
            message="rate limited",
            llm_provider="anthropic",
            model="claude-3-sonnet",
        )
        restored = pickle.loads(pickle.dumps(err))
        assert isinstance(restored, RateLimitError)
