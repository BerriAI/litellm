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

    @pytest.mark.parametrize(
        "image_url, detail",
        [
            (_LITELLM_LOGO_IMAGE_URL, None),
            (_LITELLM_LOGO_IMAGE_URL, "low"),
            (_LITELLM_LOGO_IMAGE_URL, "high"),
            (_AWSMP_LOGO_IMAGE_URL, "low"),
            (_AWSMP_LOGO_IMAGE_URL, "high"),
        ],
    )
    @pytest.mark.flaky(retries=4, delay=2)
    def test_image_url(self, image_url, detail):
        super().test_image_url(detail=detail, image_url=image_url)
    test_content_list_handling = None
    test_image_url_string = None
    test_pdf_handling = None


class TestBedrockInvokeNovaJson(BaseLLMChatTest):
    test_json_response_format = None

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

    def test_json_response_pydantic_obj(self):
        if os.environ.get("LITELLM_RUN_LIVE_BEDROCK_NOVA_JSON_TESTS") != "1":
            pytest.skip("Live Bedrock Nova response-schema E2E tests are opt-in")
        if os.environ.get("CASSETTE_REDIS_URL"):
            pytest.skip(
                "Live Bedrock Nova response-schema E2E tests cannot run under VCR replay"
            )
        super().test_json_response_pydantic_obj()
