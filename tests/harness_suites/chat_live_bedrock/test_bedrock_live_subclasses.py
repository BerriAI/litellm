"""Live Bedrock BaseLLMChatTest subclasses migrated from
tests/llm_translation by the CI keep/drop audit (8c, chat_live_bedrock).
Request-body goldens these classes used to carry stayed behind in
tests/llm_translation/test_bedrock_gpt_oss.py and test_bedrock_moonshot.py.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath("../.."))
import litellm
from base_llm_unit_tests import BaseLLMChatTest

class TestBedrockInvokeClaudeJson(BaseLLMChatTest):
    def get_base_completion_call_args(self) -> dict:
        litellm._turn_on_debug()
        return {
            "model": "bedrock/invoke/us.anthropic.claude-haiku-4-5-20251001-v1:0",
        }


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

    def test_json_response_pydantic_obj(self):
        if os.environ.get("LITELLM_RUN_LIVE_BEDROCK_NOVA_JSON_TESTS") != "1":
            pytest.skip("Live Bedrock Nova response-schema E2E tests are opt-in")
        if os.environ.get("CASSETTE_REDIS_URL"):
            pytest.skip(
                "Live Bedrock Nova response-schema E2E tests cannot run under VCR replay"
            )
        super().test_json_response_pydantic_obj()


class TestBedrockGPTOSS(BaseLLMChatTest):
    def get_base_completion_call_args(self) -> dict:
        return {
            "model": "bedrock/converse/openai.gpt-oss-20b-1:0",
        }

    def test_function_calling_with_tool_response(self):
        """Bedrock GPT-OSS intermittently emits truncated toolUse.input deltas on
        the live endpoint, which makes the inherited live integration test flaky.
        The accumulation side is covered deterministically by
        tests/test_litellm/llms/bedrock/chat/test_invoke_handler.py::test_transform_tool_calls_index;
        the GPT-OSS-specific request-body transformation is covered by
        TestBedrockGPTOSSGoldens in tests/llm_translation/test_bedrock_gpt_oss.py.
        """
        pass

    def test_prompt_caching(self):
        """
        Remove override once we have access to Bedrock prompt caching
        """
        pass

    async def test_completion_cost(self):
        """
        Bedrock GPT-OSS models are flaky and occasionally report 0 token counts in api response
        """
        pass


class TestBedrockMoonshotInvoke(BaseLLMChatTest):
    """Live suite for Bedrock Moonshot (Kimi K2) via the invoke route."""

    def get_base_completion_call_args(self) -> dict:
        litellm._turn_on_debug()
        return {
            "model": "bedrock/invoke/moonshot.kimi-k2-thinking",
        }
