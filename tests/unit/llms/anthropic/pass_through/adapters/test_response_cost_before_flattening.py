"""Regression tests for the x-litellm-response-cost image-token race (#44743).

The Anthropic adapter flattens the OpenAI-shaped usage into the Anthropic
response TypedDict, dropping ``completion_tokens_details.image_tokens``. The
proxy recomputes the response-cost header from that translated response when
the async success handler has not stored the cost yet, so on some calls image
output tokens were priced at the text rate. The adapter now computes and
stores the cost from the untouched ModelResponse before the flattening.
"""

import time
from typing import cast

import pytest

from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLogging
from litellm.llms.anthropic.pass_through.adapters.handler import (
    ANTHROPIC_ADAPTER,
    _store_response_cost_before_usage_flattening,
)
from litellm.types.utils import (
    CompletionTokensDetailsWrapper,
    ModelResponse,
    Usage,
)


def _image_model_response() -> ModelResponse:
    return ModelResponse(
        id="resp_1",
        model="gemini-3.1-flash-image",
        choices=[],
        usage=Usage(
            prompt_tokens=7,
            completion_tokens=1120,
            completion_tokens_details=CompletionTokensDetailsWrapper(image_tokens=1120),
        ),
    )


@pytest.fixture
def logging_obj() -> LiteLLMLogging:
    obj = LiteLLMLogging(
        model="vertex_ai/gemini-3.1-flash-image",
        messages=[{"role": "user", "content": "a red square"}],
        stream=False,
        call_type="anthropic_messages",
        start_time=time.time(),
        litellm_call_id="44743",
        function_id="44743",
    )
    return obj


def test_stores_cost_from_pre_flattening_response(logging_obj: LiteLLMLogging) -> None:
    """The stored cost is computed from a response that still carries image_tokens."""
    seen_results = []

    def fake_calculator(self, result, **kwargs):
        seen_results.append(result)
        return 0.0672035

    original = LiteLLMLogging._response_cost_calculator
    LiteLLMLogging._response_cost_calculator = fake_calculator
    try:
        _store_response_cost_before_usage_flattening(_image_model_response(), {"litellm_logging_obj": logging_obj})
    finally:
        LiteLLMLogging._response_cost_calculator = original

    assert logging_obj.model_call_details["response_cost"] == 0.0672035
    assert len(seen_results) == 1
    usage = seen_results[0].usage
    assert usage.completion_tokens_details.image_tokens == 1120


def test_no_logging_obj_is_a_noop() -> None:
    _store_response_cost_before_usage_flattening(_image_model_response(), {})


def test_calculator_failure_is_swallowed(logging_obj: LiteLLMLogging) -> None:
    def raising_calculator(self, result, **kwargs):
        raise RuntimeError("no pricing for model")

    original = LiteLLMLogging._response_cost_calculator
    LiteLLMLogging._response_cost_calculator = raising_calculator
    try:
        _store_response_cost_before_usage_flattening(_image_model_response(), {"litellm_logging_obj": logging_obj})
    finally:
        LiteLLMLogging._response_cost_calculator = original

    assert "response_cost" not in logging_obj.model_call_details


def test_translated_response_drops_image_token_details(logging_obj: LiteLLMLogging) -> None:
    """Pins the root cause: after the Anthropic translation the usage no longer
    distinguishes image tokens, so a recompute from it prices them at the text
    rate — which is why the cost must be captured before translation."""
    translated = ANTHROPIC_ADAPTER.translate_completion_output_params(cast(ModelResponse, _image_model_response()))
    assert translated is not None
    usage = translated["usage"]
    assert usage["output_tokens"] == 1120
    assert "completion_tokens_details" not in usage
    assert not hasattr(usage, "completion_tokens_details")
