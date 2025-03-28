import os
import sys
from unittest.mock import MagicMock


sys.path.insert(
    0, os.path.abspath("../../../../..")
)  # Adds the parent directory to the system path

from litellm.llms.mistral.mistral_chat_transformation import MistralConfig


class TestMistralTransform:
    def setup_method(self):
        self.config = MistralConfig()
        self.model = "mistral-small-latest"
        self.logging_obj = MagicMock()

    def test_map_mistral_params(self):
        """Test that parameters are correctly mapped"""
        test_params = {"temperature": 0.7, "max_tokens": 200, "max_completion_tokens": 256}

        result = self.config.map_openai_params(
            non_default_params=test_params,
            optional_params={},
            model=self.model,
            drop_params=False,
        )

        # The function should properly map max_completion_tokens to max_tokens and override max_tokens
        assert result == {"temperature": 0.7, "max_tokens": 256}

    def test_mistral_max_tokens_backward_compat(self):
        """Test that parameters are correctly mapped"""
        test_params = {"temperature": 0.7, "max_tokens": 200,}

        result = self.config.map_openai_params(
            non_default_params=test_params,
            optional_params={},
            model=self.model,
            drop_params=False,
        )

        # The function should properly map max_tokens if max_completion_tokens is not provided
        assert result == {"temperature": 0.7, "max_tokens": 200}
