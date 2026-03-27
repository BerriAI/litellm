import os

import litellm
from litellm.litellm_core_utils.llm_cost_calc.utils import generic_cost_per_token
from litellm.types.utils import (
    CompletionTokensDetailsWrapper,
    PromptTokensDetailsWrapper,
    Usage,
)


def test_gemini_partial_prompt_token_breakdown_bills_unaccounted_remainder():
    """
    Reproduce #24375: when prompt_tokens_details only covers a subset of the
    prompt tokens, the remainder should still be billed as text tokens.
    """
    original_local_cost_map = os.environ.get("LITELLM_LOCAL_MODEL_COST_MAP")
    original_model_cost = litellm.model_cost

    try:
        os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
        litellm.model_cost = litellm.get_model_cost_map(url="")

        model_info = litellm.get_model_info(
            model="gemini/gemini-2.5-flash", custom_llm_provider="gemini"
        )
        output_cost_per_reasoning_token = model_info.get(
            "output_cost_per_reasoning_token", model_info["output_cost_per_token"]
        )

        usage = Usage(
            prompt_tokens=783,
            completion_tokens=96,
            total_tokens=879,
            prompt_tokens_details=PromptTokensDetailsWrapper(text_tokens=9),
            completion_tokens_details=CompletionTokensDetailsWrapper(
                reasoning_tokens=92,
                text_tokens=4,
            ),
        )

        input_cost, output_cost = generic_cost_per_token(
            model="gemini/gemini-2.5-flash",
            usage=usage,
            custom_llm_provider="gemini",
        )

        expected_input_cost = usage.prompt_tokens * model_info["input_cost_per_token"]
        expected_output_cost = (4 * model_info["output_cost_per_token"]) + (
            92 * output_cost_per_reasoning_token
        )

        assert abs(input_cost - expected_input_cost) < 1e-12
        assert abs(output_cost - expected_output_cost) < 1e-12
    finally:
        if original_local_cost_map is None:
            os.environ.pop("LITELLM_LOCAL_MODEL_COST_MAP", None)
        else:
            os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = original_local_cost_map
        litellm.model_cost = original_model_cost


def test_partial_completion_token_breakdown_bills_unaccounted_remainder_as_text():
    """
    If completion_tokens_details omits part of the total completion tokens, the
    remainder should be billed as text tokens instead of being dropped.
    """
    original_local_cost_map = os.environ.get("LITELLM_LOCAL_MODEL_COST_MAP")
    original_model_cost = litellm.model_cost

    try:
        os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
        litellm.model_cost = litellm.get_model_cost_map(url="")

        model_info = litellm.get_model_info(
            model="gemini/gemini-2.5-flash", custom_llm_provider="gemini"
        )
        output_cost_per_reasoning_token = model_info.get(
            "output_cost_per_reasoning_token", model_info["output_cost_per_token"]
        )

        usage = Usage(
            prompt_tokens=10,
            completion_tokens=120,
            total_tokens=130,
            prompt_tokens_details=PromptTokensDetailsWrapper(text_tokens=10),
            completion_tokens_details=CompletionTokensDetailsWrapper(
                reasoning_tokens=92,
                text_tokens=4,
            ),
        )

        _, output_cost = generic_cost_per_token(
            model="gemini/gemini-2.5-flash",
            usage=usage,
            custom_llm_provider="gemini",
        )

        expected_output_cost = (28 * model_info["output_cost_per_token"]) + (
            92 * output_cost_per_reasoning_token
        )

        assert abs(output_cost - expected_output_cost) < 1e-12
    finally:
        if original_local_cost_map is None:
            os.environ.pop("LITELLM_LOCAL_MODEL_COST_MAP", None)
        else:
            os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = original_local_cost_map
        litellm.model_cost = original_model_cost
