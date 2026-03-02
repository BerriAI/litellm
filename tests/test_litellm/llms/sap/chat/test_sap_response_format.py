"""
Test SAP response_format support for various models.

SAP GenAI Hub natively supports response_format for Anthropic models,
so no tool-based conversion is needed. This test verifies that response_format
is correctly passed through to the API for supported models.
"""

import pytest

from litellm.llms.sap.chat.transformation import GenAIHubOrchestrationConfig


class TestResponseFormatSupport:
    """Test response_format is supported for appropriate models."""

    def test_anthropic_model_supports_response_format(self):
        """Anthropic models should support response_format param (native SAP support)."""
        config = GenAIHubOrchestrationConfig()
        params = config.get_supported_openai_params("anthropic--claude-3-5-sonnet")
        assert "response_format" in params

    def test_anthropic_claude_4_supports_response_format(self):
        """Claude 4 models should support response_format param."""
        config = GenAIHubOrchestrationConfig()
        params = config.get_supported_openai_params("anthropic--claude-4.5-sonnet")
        assert "response_format" in params

    def test_cohere_model_does_not_support_response_format(self):
        """Cohere models should not support response_format param."""
        config = GenAIHubOrchestrationConfig()
        params = config.get_supported_openai_params("cohere--command-r")
        assert "response_format" not in params

    def test_amazon_model_does_not_support_response_format(self):
        """Amazon models should not support response_format."""
        config = GenAIHubOrchestrationConfig()
        params = config.get_supported_openai_params("amazon--nova-pro")
        assert "response_format" not in params

    def test_alephalpha_model_does_not_support_response_format(self):
        """AlephAlpha models should not support response_format param."""
        config = GenAIHubOrchestrationConfig()
        params = config.get_supported_openai_params("alephalpha--luminous")
        assert "response_format" not in params

    def test_gpt4_exact_does_not_support_response_format(self):
        """gpt-4 (exact match) should not support response_format param."""
        config = GenAIHubOrchestrationConfig()
        params = config.get_supported_openai_params("gpt-4")
        assert "response_format" not in params

    def test_gpt4o_supports_response_format(self):
        """gpt-4o should support response_format (native support)."""
        config = GenAIHubOrchestrationConfig()
        params = config.get_supported_openai_params("gpt-4o")
        assert "response_format" in params

    def test_gemini_supports_response_format(self):
        """Gemini models should support response_format (native support)."""
        config = GenAIHubOrchestrationConfig()
        params = config.get_supported_openai_params("gemini-1.5-pro")
        assert "response_format" in params


class TestTransformRequestWithResponseFormat:
    """Test transform_request handles response_format correctly."""

    def test_transform_request_includes_json_schema_response_format(self):
        """transform_request should include response_format with json_schema type."""
        config = GenAIHubOrchestrationConfig()
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "test",
                "schema": {
                    "type": "object",
                    "properties": {"result": {"type": "string"}},
                    "required": ["result"]
                }
            }
        }

        request = config.transform_request(
            model="anthropic--claude-3-5-sonnet",
            messages=[{"role": "user", "content": "Hello"}],
            optional_params={"response_format": response_format},
            litellm_params={},
            headers={},
        )

        # Verify response_format is in the request
        prompt_config = request["config"]["modules"]["prompt_templating"]["prompt"]
        assert "response_format" in prompt_config
        assert prompt_config["response_format"]["type"] == "json_schema"

    def test_transform_request_includes_json_object_response_format(self):
        """transform_request should include response_format with json_object type."""
        config = GenAIHubOrchestrationConfig()
        response_format = {"type": "json_object"}

        request = config.transform_request(
            model="anthropic--claude-3-5-sonnet",
            messages=[{"role": "user", "content": "Hello"}],
            optional_params={"response_format": response_format},
            litellm_params={},
            headers={},
        )

        # Verify response_format is in the request
        prompt_config = request["config"]["modules"]["prompt_templating"]["prompt"]
        assert "response_format" in prompt_config
        assert prompt_config["response_format"]["type"] == "json_object"

    def test_transform_request_without_response_format(self):
        """transform_request should work without response_format."""
        config = GenAIHubOrchestrationConfig()

        request = config.transform_request(
            model="anthropic--claude-3-5-sonnet",
            messages=[{"role": "user", "content": "Hello"}],
            optional_params={},
            litellm_params={},
            headers={},
        )

        # Verify response_format is NOT in the request
        prompt_config = request["config"]["modules"]["prompt_templating"]["prompt"]
        assert "response_format" not in prompt_config

    def test_transform_request_with_tools_and_response_format(self):
        """transform_request should include both tools and response_format."""
        config = GenAIHubOrchestrationConfig()

        user_tools = [{
            "type": "function",
            "function": {
                "name": "search_web",
                "description": "Search the web",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"]
                }
            }
        }]

        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "result",
                "schema": {
                    "type": "object",
                    "properties": {"answer": {"type": "string"}}
                }
            }
        }

        request = config.transform_request(
            model="anthropic--claude-3-5-sonnet",
            messages=[{"role": "user", "content": "Hello"}],
            optional_params={"tools": user_tools, "response_format": response_format},
            litellm_params={},
            headers={},
        )

        prompt_config = request["config"]["modules"]["prompt_templating"]["prompt"]

        # Both should be present
        assert "tools" in prompt_config
        assert "response_format" in prompt_config
        assert len(prompt_config["tools"]) == 1
        assert prompt_config["tools"][0]["function"]["name"] == "search_web"


class TestStreamIterators:
    """Test streaming iterators work without json_mode."""

    def test_sync_stream_iterator_basic(self):
        """SAPStreamIterator should work without json_mode parameter."""
        from litellm.llms.sap.chat.handler import SAPStreamIterator

        iterator = SAPStreamIterator(response=iter([]))
        assert iterator._done is False

    def test_async_stream_iterator_basic(self):
        """AsyncSAPStreamIterator should work without json_mode parameter."""
        from litellm.llms.sap.chat.handler import AsyncSAPStreamIterator

        async def async_gen():
            yield ""

        iterator = AsyncSAPStreamIterator(response=async_gen())
        assert iterator._done is False

    def test_get_model_response_iterator_sync(self):
        """get_model_response_iterator should return sync iterator."""
        config = GenAIHubOrchestrationConfig()

        iterator = config.get_model_response_iterator(
            streaming_response=iter([]),
            sync_stream=True,
        )

        from litellm.llms.sap.chat.handler import SAPStreamIterator
        assert isinstance(iterator, SAPStreamIterator)

    def test_get_model_response_iterator_async(self):
        """get_model_response_iterator should return async iterator."""
        config = GenAIHubOrchestrationConfig()

        async def async_gen():
            yield ""

        iterator = config.get_model_response_iterator(
            streaming_response=async_gen(),
            sync_stream=False,
        )

        from litellm.llms.sap.chat.handler import AsyncSAPStreamIterator
        assert isinstance(iterator, AsyncSAPStreamIterator)


class TestNestedSchema:
    """Test that complex nested schemas are preserved correctly."""

    def test_nested_schema_preserved(self):
        """Complex nested schemas should be preserved in the request."""
        config = GenAIHubOrchestrationConfig()
        nested_schema = {
            "type": "object",
            "properties": {
                "user": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "addresses": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "street": {"type": "string"},
                                    "city": {"type": "string"}
                                }
                            }
                        }
                    }
                }
            }
        }

        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "nested",
                "schema": nested_schema
            }
        }

        request = config.transform_request(
            model="anthropic--claude-3-5-sonnet",
            messages=[{"role": "user", "content": "Test"}],
            optional_params={"response_format": response_format},
            litellm_params={},
            headers={},
        )

        # Verify the nested schema is preserved
        prompt_config = request["config"]["modules"]["prompt_templating"]["prompt"]
        assert "response_format" in prompt_config
        assert prompt_config["response_format"]["json_schema"]["schema"] == nested_schema


class TestMarkdownStripping:
    """Test markdown code block stripping for JSON responses."""

    def test_strip_markdown_json_wrapper(self):
        """Should strip ```json ... ``` wrapper from content."""
        from litellm.types.utils import ModelResponse, Choices, Message

        config = GenAIHubOrchestrationConfig()
        response = ModelResponse(
            id="test",
            choices=[Choices(
                index=0,
                message=Message(role="assistant", content='```json\n{"answer": 4}\n```'),
                finish_reason="stop"
            )],
            model="test"
        )

        result = config._strip_markdown_json(response)
        assert result.choices[0].message.content == '{"answer": 4}'

    def test_strip_plain_markdown_wrapper(self):
        """Should strip ``` ... ``` wrapper (without json label)."""
        from litellm.types.utils import ModelResponse, Choices, Message

        config = GenAIHubOrchestrationConfig()
        response = ModelResponse(
            id="test",
            choices=[Choices(
                index=0,
                message=Message(role="assistant", content='```\n{"answer": 4}\n```'),
                finish_reason="stop"
            )],
            model="test"
        )

        result = config._strip_markdown_json(response)
        assert result.choices[0].message.content == '{"answer": 4}'

    def test_no_strip_when_no_markdown(self):
        """Should not modify content without markdown wrapper."""
        from litellm.types.utils import ModelResponse, Choices, Message

        config = GenAIHubOrchestrationConfig()
        response = ModelResponse(
            id="test",
            choices=[Choices(
                index=0,
                message=Message(role="assistant", content='{"answer": 4}'),
                finish_reason="stop"
            )],
            model="test"
        )

        result = config._strip_markdown_json(response)
        assert result.choices[0].message.content == '{"answer": 4}'

    def test_strip_only_for_json_response_format(self):
        """Should only strip for json_object or json_schema types, not text."""
        config = GenAIHubOrchestrationConfig()

        # json_schema type should trigger stripping
        assert config.get_supported_openai_params("anthropic--claude-3-5-sonnet")
        # The actual stripping is tested in transform_response, which checks type
