from unittest.mock import patch
import os
import sys
from litellm.types.utils import (
    CompletionTokensDetailsWrapper,
    PromptTokensDetailsWrapper
)

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path

from litellm.litellm_core_utils.llm_cost_calc.utils import (
    generic_cost_per_token
)
from litellm.types.utils import Usage

def test_completion_breakdown_ignores_inflated_text_tokens():
    """
    Some providers set text_tokens to the full completion count while also reporting
    reasoning/audio/image. Billing must use remainder text:
    max(0, completion_tokens - reasoning - audio - image).
    """
    mock_model_info = {
        "input_cost_per_token": 1e-6,
        "output_cost_per_token": 2e-6,
        "output_cost_per_audio_token": 3e-6,
        "output_cost_per_reasoning_token": 5e-6,
        "output_cost_per_image_token": 7e-6,
    }
    usage = Usage(
        prompt_tokens=400,
        completion_tokens=1000,
        total_tokens=1400,
        completion_tokens_details=CompletionTokensDetailsWrapper(
            accepted_prediction_tokens=None,
            audio_tokens=50,
            reasoning_tokens=100,
            rejected_prediction_tokens=None,
            text_tokens=1000,
            image_tokens=200,
        ),
        prompt_tokens_details=PromptTokensDetailsWrapper(
            audio_tokens=None, cached_tokens=None, text_tokens=400, image_tokens=None
        ),
    )
    with patch(
        "litellm.litellm_core_utils.llm_cost_calc.utils.get_model_info",
        return_value=mock_model_info,
    ):
        prompt_cost, completion_cost = generic_cost_per_token(
            model="test-model", usage=usage, custom_llm_provider="anthropic"
        )
    # Expected prompt: 400 * 1e-6 = 0.0004
    # Expected completion: text 650 (1000-100-50-200) * 2e-6 + 50*3e-6 + 100*5e-6 + 200*7e-6 = 0.00335
    expected_prompt_cost = 400 * 1e-6
    expected_completion_cost = (
        (1000 - 100 - 50 - 200) * 2e-6 + 50 * 3e-6 + 100 * 5e-6 + 200 * 7e-6
    )
    assert round(prompt_cost, 12) == round(expected_prompt_cost, 12)
    assert round(completion_cost, 12) == round(expected_completion_cost, 12)