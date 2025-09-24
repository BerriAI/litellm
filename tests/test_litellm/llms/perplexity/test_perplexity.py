import os
import sys

sys.path.insert(0, os.path.abspath("../../.."))

import pytest


class TestPerplexityWebSearch:
    """Test suite for Perplexity web search functionality."""

    @pytest.mark.parametrize(
        "model",
        ["perplexity/sonar", "perplexity/sonar-pro"]
    )
    def test_web_search_options_in_supported_params(self, model):
        """
        Test that web_search_options is in the list of supported parameters for Perplexity sonar models
        """
        from litellm.llms.perplexity.chat.transformation import PerplexityChatConfig
        
        config = PerplexityChatConfig()
        supported_params = config.get_supported_openai_params(model=model)
        
        assert "web_search_options" in supported_params, f"web_search_options should be supported for {model}"
