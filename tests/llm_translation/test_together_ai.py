"""
TogetherAI supported-params goldens. The live BaseLLMChatTest subclass moved
to tests/harness_suites/chat_live_longtail/.
"""

import os
import sys

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path

import litellm
import pytest


class TestTogetherAIParams:
    @pytest.mark.parametrize(
        "model, expected_bool",
        [
            ("meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo", True),
            ("nvidia/Llama-3.1-Nemotron-70B-Instruct-HF", False),
        ],
    )
    def test_get_supported_response_format_together_ai(
        self, model: str, expected_bool: bool
    ) -> None:
        os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
        litellm.model_cost = litellm.get_model_cost_map(url="")
        optional_params = litellm.get_supported_openai_params(
            model, custom_llm_provider="together_ai"
        )
        # Mapped provider
        assert isinstance(optional_params, list)

        if expected_bool:
            assert "response_format" in optional_params
            assert "tools" in optional_params
        else:
            assert "response_format" not in optional_params
            assert "tools" not in optional_params
