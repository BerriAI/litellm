import os
import sys

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../.."))
)

from litellm.llms.azure.chat.gpt_transformation import AzureOpenAIConfig


class TestAzureOpenAIConfig:
    def test_is_response_format_supported_model(self):
        config = AzureOpenAIConfig()
        assert config._is_response_format_supported_model("gpt-4.1")
        assert config._is_response_format_supported_model("gpt-4-1")
