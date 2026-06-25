import os
import sys
from unittest.mock import MagicMock

sys.path.insert(
    0, os.path.abspath("../../../../..")
)  # Adds the parent directory to the system path

import litellm
from litellm.llms.cohere.chat.transformation import CohereChatConfig
from litellm.llms.cohere.chat.v2_transformation import CohereV2ChatConfig


class TestCohereTransform:
    def setup_method(self):
        self.config = CohereChatConfig()
        self.model = "command-r-plus-latest"
        self.logging_obj = MagicMock()

    def test_map_cohere_params(self):
        """Test that parameters are correctly mapped"""
        test_params = {
            "temperature": 0.7,
            "max_tokens": 200,
            "max_completion_tokens": 256,
        }

        result = self.config.map_openai_params(
            non_default_params=test_params,
            optional_params={},
            model=self.model,
            drop_params=False,
        )

        # The function should properly map max_completion_tokens to max_tokens and override max_tokens
        assert result == {"temperature": 0.7, "max_tokens": 256}

    def test_cohere_max_tokens_backward_compat(self):
        """Test that parameters are correctly mapped"""
        test_params = {
            "temperature": 0.7,
            "max_tokens": 200,
        }

        result = self.config.map_openai_params(
            non_default_params=test_params,
            optional_params={},
            model=self.model,
            drop_params=False,
        )

        # The function should properly map max_tokens if max_completion_tokens is not provided
        assert result == {"temperature": 0.7, "max_tokens": 200}


class TestCohereV2Transform:
    def setup_method(self):
        self.config = CohereV2ChatConfig()
        self.model = "command-r"

    def test_v2_supports_max_completion_tokens(self):
        """max_completion_tokens must be advertised so get_optional_params does not reject it"""
        assert "max_completion_tokens" in self.config.get_supported_openai_params(
            self.model
        )

    def test_v2_max_tokens_only_still_maps(self):
        """max_tokens alone maps to cohere max_tokens when max_completion_tokens is absent"""
        result = self.config.map_openai_params(
            non_default_params={"temperature": 0.7, "max_tokens": 200},
            optional_params={},
            model=self.model,
            drop_params=False,
        )

        assert result == {"temperature": 0.7, "max_tokens": 200}

    def test_v2_map_max_completion_tokens_overrides_max_tokens(self):
        """max_completion_tokens maps to cohere max_tokens and overrides max_tokens, matching v1"""
        result = self.config.map_openai_params(
            non_default_params={
                "temperature": 0.7,
                "max_tokens": 200,
                "max_completion_tokens": 256,
            },
            optional_params={},
            model=self.model,
            drop_params=False,
        )

        assert result == {"temperature": 0.7, "max_tokens": 256}

    def test_v2_max_completion_tokens_precedence_is_order_independent(self):
        """max_completion_tokens wins over max_tokens regardless of dict ordering"""
        max_tokens_first = self.config.map_openai_params(
            non_default_params={"max_tokens": 200, "max_completion_tokens": 256},
            optional_params={},
            model=self.model,
            drop_params=False,
        )
        max_completion_first = self.config.map_openai_params(
            non_default_params={"max_completion_tokens": 256, "max_tokens": 200},
            optional_params={},
            model=self.model,
            drop_params=False,
        )

        assert max_tokens_first == {"max_tokens": 256}
        assert max_completion_first == {"max_tokens": 256}

    def test_v2_default_route_accepts_max_completion_tokens(self):
        """The default cohere_chat route resolves to v2; max_completion_tokens must not raise"""
        optional_params = litellm.get_optional_params(
            model=self.model,
            custom_llm_provider="cohere_chat",
            max_completion_tokens=256,
        )

        assert optional_params["max_tokens"] == 256
