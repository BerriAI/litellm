"""
Tests that all LiteLLM exception classes survive a pickle round-trip.

Relevant issue: https://github.com/BerriAI/litellm/issues/24136

Without the fix, exceptions fail with TypeError when pickle tries to
reconstruct them via cls(*self.args), because the required positional
arguments (message, llm_provider, model, …) are not stored in self.args.
Additionally, httpx.Response / httpx.Request attributes are not picklable
by default.
"""

import pickle

import httpx
import pytest

import litellm.exceptions as exc


def _roundtrip(obj):
    """Pickle and unpickle an object, returning the reconstructed copy."""
    return pickle.loads(pickle.dumps(obj))


def _make_response(status: int = 400) -> httpx.Response:
    return httpx.Response(
        status_code=status,
        request=httpx.Request(method="GET", url="https://litellm.ai"),
    )


# ---------------------------------------------------------------------------
# Fixtures — one instance per exception class
# ---------------------------------------------------------------------------

CASES = [
    exc.AuthenticationError(
        message="bad key",
        llm_provider="openai",
        model="gpt-4",
    ),
    exc.NotFoundError(
        message="model not found",
        model="gpt-99",
        llm_provider="openai",
    ),
    exc.BadRequestError(
        message="invalid param",
        model="gpt-4",
        llm_provider="openai",
    ),
    exc.ImageFetchError(
        message="cannot fetch image",
        model="gpt-4",
        llm_provider="openai",
    ),
    exc.UnprocessableEntityError(
        message="unprocessable",
        model="gpt-4",
        llm_provider="openai",
        response=_make_response(422),
    ),
    exc.Timeout(
        message="timed out",
        model="gpt-4",
        llm_provider="openai",
    ),
    exc.PermissionDeniedError(
        message="denied",
        llm_provider="openai",
        model="gpt-4",
        response=_make_response(403),
    ),
    exc.RateLimitError(
        message="rate limited",
        llm_provider="openai",
        model="gpt-4",
    ),
    exc.ContextWindowExceededError(
        message="context too long",
        model="gpt-4",
        llm_provider="openai",
    ),
    exc.RejectedRequestError(
        message="guardrail rejected",
        model="gpt-4",
        llm_provider="openai",
        request_data={"messages": []},
    ),
    exc.ContentPolicyViolationError(
        message="content violation",
        model="gpt-4",
        llm_provider="openai",
    ),
    exc.ServiceUnavailableError(
        message="service down",
        llm_provider="openai",
        model="gpt-4",
    ),
    exc.BadGatewayError(
        message="bad gateway",
        llm_provider="openai",
        model="gpt-4",
    ),
    exc.InternalServerError(
        message="internal error",
        llm_provider="openai",
        model="gpt-4",
    ),
    exc.APIError(
        status_code=500,
        message="api error",
        llm_provider="openai",
        model="gpt-4",
    ),
    exc.APIConnectionError(
        message="connection failed",
        llm_provider="openai",
        model="gpt-4",
    ),
    exc.APIResponseValidationError(
        message="bad response",
        llm_provider="openai",
        model="gpt-4",
    ),
    exc.JSONSchemaValidationError(
        model="gpt-4",
        llm_provider="openai",
        raw_response='{"bad": true}',
        schema='{"type": "object"}',
    ),
    exc.OpenAIError(),
    exc.UnsupportedParamsError(
        message="unsupported param",
        llm_provider="openai",
        model="gpt-4",
    ),
    exc.BudgetExceededError(current_cost=1.5, max_budget=1.0),
    exc.InvalidRequestError(
        message="invalid request",
        model="gpt-4",
        llm_provider="openai",
    ),
    exc.MockException(
        status_code=500,
        message="mock error",
        llm_provider="openai",
        model="gpt-4",
    ),
    exc.LiteLLMUnknownProvider(model="unknown/model"),
    exc.GuardrailRaisedException(guardrail_name="my-guard", message="blocked"),
    exc.BlockedPiiEntityError(entity_type="email", guardrail_name="pii-guard"),
    exc.MidStreamFallbackError(
        message="mid-stream fail",
        model="gpt-4",
        llm_provider="openai",
    ),
    exc.GuardrailInterventionNormalStringError(message="intervention"),
]


@pytest.mark.parametrize("original", CASES, ids=lambda e: type(e).__name__)
def test_pickle_roundtrip(original):
    """Each exception must survive pickle.dumps → pickle.loads without error."""
    restored = _roundtrip(original)
    assert type(restored) is type(original)


@pytest.mark.parametrize("original", CASES, ids=lambda e: type(e).__name__)
def test_pickle_preserves_message(original):
    """The message attribute must be identical after the round-trip."""
    if not hasattr(original, "message"):
        pytest.skip("exception has no message attribute")
    restored = _roundtrip(original)
    assert restored.message == original.message


@pytest.mark.parametrize("original", CASES, ids=lambda e: type(e).__name__)
def test_pickle_preserves_llm_provider(original):
    """llm_provider must be preserved when present."""
    if not hasattr(original, "llm_provider"):
        pytest.skip("exception has no llm_provider attribute")
    restored = _roundtrip(original)
    assert restored.llm_provider == original.llm_provider


@pytest.mark.parametrize("original", CASES, ids=lambda e: type(e).__name__)
def test_pickle_preserves_model(original):
    """model must be preserved when present."""
    if not hasattr(original, "model"):
        pytest.skip("exception has no model attribute")
    restored = _roundtrip(original)
    assert restored.model == original.model


@pytest.mark.parametrize("original", CASES, ids=lambda e: type(e).__name__)
def test_pickle_isinstance_checks_still_work(original):
    """After round-trip, isinstance checks against the original class must pass."""
    restored = _roundtrip(original)
    assert isinstance(restored, type(original))
    assert isinstance(restored, Exception)


def test_pickle_rejected_request_preserves_request_data():
    original = exc.RejectedRequestError(
        message="blocked",
        model="gpt-4",
        llm_provider="openai",
        request_data={"messages": [{"role": "user", "content": "hello"}]},
    )
    restored = _roundtrip(original)
    assert restored.request_data == original.request_data


def test_pickle_budget_exceeded_preserves_costs():
    original = exc.BudgetExceededError(current_cost=2.5, max_budget=1.0)
    restored = _roundtrip(original)
    assert restored.current_cost == original.current_cost
    assert restored.max_budget == original.max_budget


def test_pickle_json_schema_error_preserves_raw_response():
    original = exc.JSONSchemaValidationError(
        model="gpt-4",
        llm_provider="openai",
        raw_response='{"unexpected": "field"}',
        schema='{"required": ["name"]}',
    )
    restored = _roundtrip(original)
    assert restored.raw_response == original.raw_response
    assert restored.schema == original.schema


def test_pickle_guardrail_raised_preserves_guardrail_name():
    original = exc.GuardrailRaisedException(
        guardrail_name="content-filter", message="profanity detected"
    )
    restored = _roundtrip(original)
    assert restored.guardrail_name == original.guardrail_name


def test_pickle_blocked_pii_preserves_entity_type():
    original = exc.BlockedPiiEntityError(
        entity_type="credit_card", guardrail_name="pii-guard"
    )
    restored = _roundtrip(original)
    assert restored.entity_type == original.entity_type
    assert restored.guardrail_name == original.guardrail_name


def test_pickle_midstream_preserves_generated_content():
    original = exc.MidStreamFallbackError(
        message="stream interrupted",
        model="gpt-4",
        llm_provider="openai",
        generated_content="partial response text",
        is_pre_first_chunk=False,
    )
    restored = _roundtrip(original)
    assert restored.generated_content == original.generated_content
    assert restored.is_pre_first_chunk == original.is_pre_first_chunk


def test_pickle_with_retries_info():
    original = exc.RateLimitError(
        message="too many requests",
        llm_provider="openai",
        model="gpt-4",
        max_retries=3,
        num_retries=3,
    )
    restored = _roundtrip(original)
    assert restored.max_retries == 3
    assert restored.num_retries == 3


def test_pickle_exception_is_raiseable():
    """Unpickled exceptions must still be raise-able and catchable."""
    original = exc.AuthenticationError(
        message="invalid api key", llm_provider="openai", model="gpt-4"
    )
    restored = _roundtrip(original)
    with pytest.raises(exc.AuthenticationError):
        raise restored


def test_pickle_multiple_roundtrips():
    """Exceptions must survive multiple sequential pickle round-trips."""
    original = exc.InternalServerError(
        message="server error", llm_provider="anthropic", model="claude-3"
    )
    result = original
    for _ in range(3):
        result = _roundtrip(result)
    assert result.message == original.message
    assert result.llm_provider == original.llm_provider
