from base_llm_unit_tests import BaseLLMChatTest
import json
import pytest
import sys
import os
from unittest.mock import patch, Mock, MagicMock

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import litellm
from litellm.llms.bedrock.chat.converse_transformation import AmazonConverseConfig
from litellm.llms.custom_httpx.http_handler import HTTPHandler


class TestBedrockGPTOSS(BaseLLMChatTest):
    def get_base_completion_call_args(self) -> dict:
        return {
            "model": "bedrock/converse/openai.gpt-oss-20b-1:0",
        }

    def test_tool_call_no_arguments(self, tool_call_no_arguments):
        """Test that tool calls with no arguments is translated correctly. Relevant issue: https://github.com/BerriAI/litellm/issues/6833"""
        pass

    def test_function_calling_with_tool_response(self):
        """Bedrock GPT-OSS intermittently emits truncated toolUse.input deltas on
        the live endpoint, which makes the inherited live integration test flaky.
        The accumulation side is covered deterministically by
        tests/test_litellm/llms/bedrock/chat/test_invoke_handler.py::test_transform_tool_calls_index;
        the GPT-OSS-specific request-body transformation is covered by
        test_function_calling_request_body_gpt_oss below.
        """
        pass

    def test_function_calling_request_body_gpt_oss(self):
        """Verify the Bedrock Converse request body is well-formed for GPT-OSS when the
        caller supplies a tool schema with OpenAI-style metadata ($id, $schema,
        additionalProperties, strict). Bedrock only accepts a trimmed JSON Schema in
        toolSpec.inputSchema.json, so the extra fields must be stripped and the
        required shape preserved.
        """
        client = HTTPHandler()

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the weather in a city",
                    "parameters": {
                        "$id": "https://some/internal/name",
                        "$schema": "https://json-schema.org/draft-07/schema",
                        "type": "object",
                        "properties": {
                            "city": {
                                "type": "string",
                                "description": "The city to get the weather for",
                            }
                        },
                        "required": ["city"],
                        "additionalProperties": False,
                    },
                    "strict": True,
                },
            }
        ]

        with patch.object(client, "post", new=Mock()) as mock_post:
            try:
                litellm.completion(
                    model="bedrock/converse/openai.gpt-oss-20b-1:0",
                    messages=[
                        {"role": "user", "content": "How is the weather in Mumbai?"}
                    ],
                    tools=tools,
                    aws_region_name="us-west-2",
                    client=client,
                )
            except Exception:
                # We only care about the outgoing request; the mocked post returns
                # a Mock that can't be parsed as a real Converse response.
                pass

        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args.kwargs

        assert call_kwargs["url"].endswith(
            "/model/openai.gpt-oss-20b-1%3A0/converse"
        ), call_kwargs["url"]

        request_body = json.loads(call_kwargs["data"])

        assert "toolConfig" in request_body
        tool_specs = request_body["toolConfig"]["tools"]
        assert len(tool_specs) == 1
        tool_spec = tool_specs[0]["toolSpec"]
        assert tool_spec["name"] == "get_weather"
        assert tool_spec["description"] == "Get the weather in a city"

        input_schema = tool_spec["inputSchema"]["json"]
        assert input_schema["type"] == "object"
        assert input_schema["required"] == ["city"]
        assert input_schema["properties"]["city"]["type"] == "string"

        # Bedrock's toolSpec.inputSchema.json only accepts type/properties/required;
        # the OpenAI-style metadata must not leak through.
        for stripped_field in ("$id", "$schema", "additionalProperties", "strict"):
            assert (
                stripped_field not in input_schema
            ), f"{stripped_field} should be stripped before hitting Bedrock"

        assert request_body["messages"][0]["role"] == "user"
        assert (
            request_body["messages"][0]["content"][0]["text"]
            == "How is the weather in Mumbai?"
        )

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

    @pytest.mark.parametrize(
        "model",
        [
            "bedrock/openai.gpt-oss-20b-1:0",
            "bedrock/openai.gpt-oss-120b-1:0",
        ],
    )
    def test_reasoning_effort_transformation_gpt_oss(self, model):
        """Test that reasoning_effort is handled correctly for GPT-OSS models."""
        config = AmazonConverseConfig()

        # Test GPT-OSS model - should keep reasoning_effort as-is
        non_default_params = {"reasoning_effort": "low"}
        optional_params = {}

        result = config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=model,
            drop_params=False,
        )

        # GPT-OSS should have reasoning_effort in result, not thinking
        assert "reasoning_effort" in result
        assert result["reasoning_effort"] == "low"
        assert "thinking" not in result
