from base_llm_unit_tests import BaseLLMChatTest
import pytest
import os

import litellm
from litellm.types.llms.bedrock import BedrockInvokeNovaRequest

_LITELLM_LOGO_IMAGE_URL = (
    "https://cdn.jsdelivr.net/gh/BerriAI/litellm@d769e81c90d453240c61fc572cdb27fae06a89d0/"
    "ui/litellm-dashboard/public/assets/logos/litellm_logo.jpg"
)
_AWSMP_LOGO_IMAGE_URL = (
    "https://awsmp-logos.s3.amazonaws.com/seller-xw5kijmvmzasy/"
    "c233c9ade2ccb5491072ae232c814942.png"
)

@pytest.mark.flaky(retries=3, delay=5)
class TestBedrockInvokeClaudeJson(BaseLLMChatTest):
    def get_base_completion_call_args(self) -> dict:
        litellm.turn_on_debug()
        return {
            "model": "bedrock/invoke/us.anthropic.claude-haiku-4-5-20251001-v1:0",
        }

    test_pdf_handling = None

class TestBedrockInvokeNovaJson(BaseLLMChatTest):

    def get_base_completion_call_args(self) -> dict:
        return {
            "model": "bedrock/invoke/us.amazon.nova-micro-v1:0",
        }

    @pytest.fixture(autouse=True)
    def skip_non_json_tests(self, request):
        if not "json" in request.function.__name__.lower():
            pytest.skip(
                f"Skipping non-JSON test: {request.function.__name__} does not contain 'json'"
            )

