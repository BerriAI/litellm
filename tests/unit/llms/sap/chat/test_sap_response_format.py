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
                    "required": ["result"],
                },
            },
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

        user_tools = [
            {
                "type": "function",
                "function": {
                    "name": "search_web",
                    "description": "Search the web",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"],
                    },
                },
            }
        ]

        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "result",
                "schema": {
                    "type": "object",
                    "properties": {"answer": {"type": "string"}},
                },
            },
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
                                    "city": {"type": "string"},
                                },
                            },
                        },
                    },
                }
            },
        }

        response_format = {
            "type": "json_schema",
            "json_schema": {"name": "nested", "schema": nested_schema},
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
        assert (
            prompt_config["response_format"]["json_schema"]["schema"] == nested_schema
        )


class TestTransformResponseWithResponseFormat:
    """Test transform_response handles response_format correctly."""

    def test_transform_response_strips_markdown_for_json_schema(self):
        """transform_response should strip markdown when response_format.type=json_schema."""
        from unittest.mock import MagicMock
        from litellm.types.utils import ModelResponse, Choices, Message

        config = GenAIHubOrchestrationConfig()

        # Create mock raw_response
        raw_response = MagicMock()
        raw_response.json.return_value = {
            "final_result": {
                "id": "test-id",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": '```json\n{"result": "success"}\n```',
                        },
                        "finish_reason": "stop",
                    }
                ],
                "model": "anthropic--claude-3-5-sonnet",
            }
        }
        raw_response.text = '{"final_result": {...}}'

        # Create mock logging_obj
        logging_obj = MagicMock()

        response_format = {
            "type": "json_schema",
            "json_schema": {"name": "test", "schema": {"type": "object"}},
        }

        result = config.transform_response(
            model="anthropic--claude-3-5-sonnet",
            raw_response=raw_response,
            model_response=ModelResponse(id="test", model="test"),
            logging_obj=logging_obj,
            request_data={},
            messages=[{"role": "user", "content": "test"}],
            optional_params={"response_format": response_format},
            litellm_params={},
            encoding=None,
        )

        assert result.choices[0].message.content == '{"result": "success"}'

    def test_transform_response_strips_markdown_for_json_object(self):
        """transform_response should strip markdown when response_format.type=json_object."""
        from unittest.mock import MagicMock
        from litellm.types.utils import ModelResponse

        config = GenAIHubOrchestrationConfig()

        raw_response = MagicMock()
        raw_response.json.return_value = {
            "final_result": {
                "id": "test-id",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": '```json\n{"answer": 42}\n```',
                        },
                        "finish_reason": "stop",
                    }
                ],
                "model": "anthropic--claude-3-5-sonnet",
            }
        }
        raw_response.text = '{"final_result": {...}}'

        logging_obj = MagicMock()

        result = config.transform_response(
            model="anthropic--claude-3-5-sonnet",
            raw_response=raw_response,
            model_response=ModelResponse(id="test", model="test"),
            logging_obj=logging_obj,
            request_data={},
            messages=[{"role": "user", "content": "test"}],
            optional_params={"response_format": {"type": "json_object"}},
            litellm_params={},
            encoding=None,
        )

        assert result.choices[0].message.content == '{"answer": 42}'

    def test_transform_response_no_strip_for_text_type(self):
        """transform_response should NOT strip markdown when response_format.type=text."""
        from unittest.mock import MagicMock
        from litellm.types.utils import ModelResponse

        config = GenAIHubOrchestrationConfig()

        raw_response = MagicMock()
        raw_response.json.return_value = {
            "final_result": {
                "id": "test-id",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": '```json\n{"data": "keep me"}\n```',
                        },
                        "finish_reason": "stop",
                    }
                ],
                "model": "anthropic--claude-3-5-sonnet",
            }
        }
        raw_response.text = '{"final_result": {...}}'

        logging_obj = MagicMock()

        result = config.transform_response(
            model="anthropic--claude-3-5-sonnet",
            raw_response=raw_response,
            model_response=ModelResponse(id="test", model="test"),
            logging_obj=logging_obj,
            request_data={},
            messages=[{"role": "user", "content": "test"}],
            optional_params={"response_format": {"type": "text"}},
            litellm_params={},
            encoding=None,
        )

        # Content should remain unchanged for text type
        assert result.choices[0].message.content == '```json\n{"data": "keep me"}\n```'

    def test_transform_response_no_strip_without_response_format(self):
        """transform_response should NOT strip markdown when no response_format provided."""
        from unittest.mock import MagicMock
        from litellm.types.utils import ModelResponse

        config = GenAIHubOrchestrationConfig()

        raw_response = MagicMock()
        raw_response.json.return_value = {
            "final_result": {
                "id": "test-id",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": '```json\n{"preserve": true}\n```',
                        },
                        "finish_reason": "stop",
                    }
                ],
                "model": "anthropic--claude-3-5-sonnet",
            }
        }
        raw_response.text = '{"final_result": {...}}'

        logging_obj = MagicMock()

        result = config.transform_response(
            model="anthropic--claude-3-5-sonnet",
            raw_response=raw_response,
            model_response=ModelResponse(id="test", model="test"),
            logging_obj=logging_obj,
            request_data={},
            messages=[{"role": "user", "content": "test"}],
            optional_params={},  # No response_format
            litellm_params={},
            encoding=None,
        )

        # Content should remain unchanged when no response_format
        assert result.choices[0].message.content == '```json\n{"preserve": true}\n```'


class TestMarkdownStripping:
    """Test markdown code block stripping for JSON responses."""

    def test_strip_markdown_json_wrapper(self):
        """Should strip ```json ... ``` wrapper from content."""
        from litellm.types.utils import ModelResponse, Choices, Message

        config = GenAIHubOrchestrationConfig()
        response = ModelResponse(
            id="test",
            choices=[
                Choices(
                    index=0,
                    message=Message(
                        role="assistant", content='```json\n{"answer": 4}\n```'
                    ),
                    finish_reason="stop",
                )
            ],
            model="test",
        )

        result = config._strip_markdown_json(response)
        assert result.choices[0].message.content == '{"answer": 4}'

    def test_strip_plain_markdown_wrapper(self):
        """Should strip ``` ... ``` wrapper (without json label)."""
        from litellm.types.utils import ModelResponse, Choices, Message

        config = GenAIHubOrchestrationConfig()
        response = ModelResponse(
            id="test",
            choices=[
                Choices(
                    index=0,
                    message=Message(
                        role="assistant", content='```\n{"answer": 4}\n```'
                    ),
                    finish_reason="stop",
                )
            ],
            model="test",
        )

        result = config._strip_markdown_json(response)
        assert result.choices[0].message.content == '{"answer": 4}'

    def test_no_strip_when_no_markdown(self):
        """Should not modify content without markdown wrapper."""
        from litellm.types.utils import ModelResponse, Choices, Message

        config = GenAIHubOrchestrationConfig()
        response = ModelResponse(
            id="test",
            choices=[
                Choices(
                    index=0,
                    message=Message(role="assistant", content='{"answer": 4}'),
                    finish_reason="stop",
                )
            ],
            model="test",
        )

        result = config._strip_markdown_json(response)
        assert result.choices[0].message.content == '{"answer": 4}'

    def test_strip_only_for_json_response_format(self):
        """Should only strip for json_object or json_schema types, not text."""
        config = GenAIHubOrchestrationConfig()

        # json_schema type should trigger stripping
        assert config.get_supported_openai_params("anthropic--claude-3-5-sonnet")
        # The actual stripping is tested in transform_response, which checks type

    def test_strip_multiple_choices(self):
        """Should strip markdown from all choices, not just the first."""
        from litellm.types.utils import ModelResponse, Choices, Message

        config = GenAIHubOrchestrationConfig()
        response = ModelResponse(
            id="test",
            choices=[
                Choices(
                    index=0,
                    message=Message(
                        role="assistant", content='```json\n{"choice": 0}\n```'
                    ),
                    finish_reason="stop",
                ),
                Choices(
                    index=1,
                    message=Message(
                        role="assistant", content='```json\n{"choice": 1}\n```'
                    ),
                    finish_reason="stop",
                ),
                Choices(
                    index=2,
                    message=Message(
                        role="assistant", content='```\n{"choice": 2}\n```'
                    ),
                    finish_reason="stop",
                ),
            ],
            model="test",
        )

        result = config._strip_markdown_json(response)
        assert result.choices[0].message.content == '{"choice": 0}'
        assert result.choices[1].message.content == '{"choice": 1}'
        assert result.choices[2].message.content == '{"choice": 2}'

    def test_strip_with_whitespace_variations(self):
        """Should handle various whitespace patterns around JSON."""
        from litellm.types.utils import ModelResponse, Choices, Message

        config = GenAIHubOrchestrationConfig()

        # Test with extra spaces and different newline styles
        test_cases = [
            ('```json\n{"a":1}\n```', '{"a":1}'),  # Standard
            ('```json\n  {"a":1}  \n```', '{"a":1}'),  # Extra spaces inside
            (
                '  ```json\n{"a":1}\n```  ',
                '{"a":1}',
            ),  # Extra spaces outside (stripped by .strip())
            ('```json\n\n{"a":1}\n\n```', '{"a":1}'),  # Extra newlines
        ]

        for input_content, expected in test_cases:
            response = ModelResponse(
                id="test",
                choices=[
                    Choices(
                        index=0,
                        message=Message(role="assistant", content=input_content),
                        finish_reason="stop",
                    )
                ],
                model="test",
            )

            result = config._strip_markdown_json(response)
            assert (
                result.choices[0].message.content == expected
            ), f"Failed for input: {repr(input_content)}"

    def test_no_strip_partial_markdown(self):
        """Should not corrupt content with incomplete markdown (only opening ```)."""
        from litellm.types.utils import ModelResponse, Choices, Message

        config = GenAIHubOrchestrationConfig()

        # Only opening backticks - should be preserved
        response = ModelResponse(
            id="test",
            choices=[
                Choices(
                    index=0,
                    message=Message(
                        role="assistant", content='```json\n{"incomplete": true}'
                    ),
                    finish_reason="stop",
                )
            ],
            model="test",
        )

        result = config._strip_markdown_json(response)
        # Should remain unchanged since there's no closing ```
        assert result.choices[0].message.content == '```json\n{"incomplete": true}'

    def test_preserve_markdown_in_json_value(self):
        """Should preserve markdown code blocks inside JSON string values."""
        from litellm.types.utils import ModelResponse, Choices, Message

        config = GenAIHubOrchestrationConfig()

        # JSON with markdown inside a string value - only outer wrapper should be stripped
        content_with_nested = '```json\n{"code": "```python\\nprint(1)\\n```"}\n```'
        response = ModelResponse(
            id="test",
            choices=[
                Choices(
                    index=0,
                    message=Message(role="assistant", content=content_with_nested),
                    finish_reason="stop",
                )
            ],
            model="test",
        )

        result = config._strip_markdown_json(response)
        # Only the outer wrapper should be stripped, inner markdown preserved
        assert (
            result.choices[0].message.content
            == '{"code": "```python\\nprint(1)\\n```"}'
        )


class TestResponseFormatErrorHandling:
    """Test error handling in response_format processing."""

    def test_empty_content_handling(self):
        """_strip_markdown_json should handle None/empty content gracefully."""
        from litellm.types.utils import ModelResponse, Choices, Message

        config = GenAIHubOrchestrationConfig()

        # Test with None content
        response_none = ModelResponse(
            id="test",
            choices=[
                Choices(
                    index=0,
                    message=Message(role="assistant", content=None),
                    finish_reason="stop",
                )
            ],
            model="test",
        )

        result = config._strip_markdown_json(response_none)
        assert result.choices[0].message.content is None

        # Test with empty string content
        response_empty = ModelResponse(
            id="test",
            choices=[
                Choices(
                    index=0,
                    message=Message(role="assistant", content=""),
                    finish_reason="stop",
                )
            ],
            model="test",
        )

        result = config._strip_markdown_json(response_empty)
        assert result.choices[0].message.content == ""

    def test_response_format_with_no_choices(self):
        """_strip_markdown_json should handle response with empty choices."""
        from litellm.types.utils import ModelResponse

        config = GenAIHubOrchestrationConfig()

        # Empty choices list
        response = ModelResponse(id="test", choices=[], model="test")

        # Should not raise an error
        result = config._strip_markdown_json(response)
        assert result.choices == []

    def test_response_format_with_message_no_content(self):
        """_strip_markdown_json should handle choice with message but no content."""
        from litellm.types.utils import ModelResponse, Choices, Message

        config = GenAIHubOrchestrationConfig()

        # Choice with message but content is None
        response = ModelResponse(
            id="test",
            choices=[
                Choices(
                    index=0,
                    message=Message(role="assistant", content=None),
                    finish_reason="stop",
                )
            ],
            model="test",
        )

        # Should not raise an error and content should remain None
        result = config._strip_markdown_json(response)
        assert result.choices[0].message.content is None


class TestStrictParameterFiltering:
    """Test that strict parameter is filtered from model_params.

    LangChain agents pass strict=true at the top level of optional_params,
    but SAP AI Core Orchestration API does not accept it as a model parameter
    for GPT models. The strict parameter should only exist inside
    response_format.json_schema, not as a top-level model param.
    """

    def test_strict_param_filtered_from_model_params(self):
        """strict should be filtered out and not appear in model.params."""
        config = GenAIHubOrchestrationConfig()

        request = config.transform_request(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hello"}],
            optional_params={"strict": True, "temperature": 0.7},
            litellm_params={},
            headers={},
        )

        # strict should NOT be in model.params
        model_params = request["config"]["modules"]["prompt_templating"]["model"][
            "params"
        ]
        assert "strict" not in model_params
        # Other params should still be there
        assert model_params.get("temperature") == 0.7

    def test_strict_preserved_inside_response_format_json_schema(self):
        """strict inside response_format.json_schema should be preserved."""
        config = GenAIHubOrchestrationConfig()

        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "test_schema",
                "strict": True,  # This is the correct location for strict
                "schema": {
                    "type": "object",
                    "properties": {"result": {"type": "string"}},
                    "required": ["result"],
                },
            },
        }

        request = config.transform_request(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hello"}],
            optional_params={"response_format": response_format},
            litellm_params={},
            headers={},
        )

        # strict should be preserved inside json_schema
        prompt_config = request["config"]["modules"]["prompt_templating"]["prompt"]
        assert prompt_config["response_format"]["json_schema"]["strict"] is True

    def test_langchain_style_strict_filtered_with_response_format(self):
        """LangChain sends strict at top level AND inside json_schema - only top level filtered."""
        config = GenAIHubOrchestrationConfig()

        # LangChain sends both top-level strict AND inside json_schema
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "agent_response",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {"answer": {"type": "string"}},
                },
            },
        }

        request = config.transform_request(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hello"}],
            optional_params={
                "strict": True,  # Top-level strict from LangChain - should be filtered
                "response_format": response_format,
                "temperature": 0.5,
            },
            litellm_params={},
            headers={},
        )

        # Top-level strict should NOT be in model.params
        model_params = request["config"]["modules"]["prompt_templating"]["model"][
            "params"
        ]
        assert "strict" not in model_params
        assert model_params.get("temperature") == 0.5

        # strict inside json_schema should be preserved
        prompt_config = request["config"]["modules"]["prompt_templating"]["prompt"]
        assert prompt_config["response_format"]["json_schema"]["strict"] is True

    def test_strict_preserved_for_anthropic_models(self):
        """strict should be preserved for Anthropic models (SAP API accepts it)."""
        config = GenAIHubOrchestrationConfig()

        request = config.transform_request(
            model="anthropic--claude-3-5-sonnet",
            messages=[{"role": "user", "content": "Hello"}],
            optional_params={"strict": True, "max_tokens": 1000},
            litellm_params={},
            headers={},
        )

        model_params = request["config"]["modules"]["prompt_templating"]["model"][
            "params"
        ]
        # Anthropic models CAN have strict in model.params (SAP API accepts it)
        assert model_params.get("strict") is True
        assert model_params.get("max_tokens") == 1000


class TestModelVariantSupport:
    """Test response_format support for various model variants."""

    def test_gpt4_turbo_supports_response_format(self):
        """gpt-4-turbo should support response_format (native support)."""
        config = GenAIHubOrchestrationConfig()
        params = config.get_supported_openai_params("gpt-4-turbo")
        assert "response_format" in params

    def test_mistral_model_support(self):
        """Mistral models should support response_format (native support)."""
        config = GenAIHubOrchestrationConfig()
        params = config.get_supported_openai_params("mistral-large")
        assert "response_format" in params


class TestMarkdownStrippingModelGating:
    """Test that markdown stripping is only applied to Anthropic models.

    The markdown stripping behavior is specific to Anthropic models on SAP GenAI Hub.
    GPT/Gemini models don't exhibit this behavior, so stripping should be gated
    to avoid accidentally modifying valid responses.
    """

    def test_gpt_model_no_markdown_strip_json_schema(self):
        """GPT models should NOT have markdown stripped for json_schema response_format."""
        from unittest.mock import MagicMock
        from litellm.types.utils import ModelResponse

        config = GenAIHubOrchestrationConfig()

        # GPT response with markdown-wrapped JSON
        raw_response = MagicMock()
        raw_response.json.return_value = {
            "final_result": {
                "id": "test-id",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": '```json\n{"result": "success"}\n```',
                        },
                        "finish_reason": "stop",
                    }
                ],
                "model": "gpt-4o",
            }
        }
        raw_response.text = '{"final_result": {...}}'

        logging_obj = MagicMock()

        response_format = {
            "type": "json_schema",
            "json_schema": {"name": "test", "schema": {"type": "object"}},
        }

        result = config.transform_response(
            model="gpt-4o",  # GPT model - should NOT strip
            raw_response=raw_response,
            model_response=ModelResponse(id="test", model="test"),
            logging_obj=logging_obj,
            request_data={},
            messages=[{"role": "user", "content": "test"}],
            optional_params={"response_format": response_format},
            litellm_params={},
            encoding=None,
        )

        # Markdown should NOT be stripped for GPT models
        assert (
            result.choices[0].message.content == '```json\n{"result": "success"}\n```'
        )

    def test_gpt_model_no_markdown_strip_json_object(self):
        """GPT models should NOT have markdown stripped for json_object response_format."""
        from unittest.mock import MagicMock
        from litellm.types.utils import ModelResponse

        config = GenAIHubOrchestrationConfig()

        raw_response = MagicMock()
        raw_response.json.return_value = {
            "final_result": {
                "id": "test-id",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": '```json\n{"answer": 42}\n```',
                        },
                        "finish_reason": "stop",
                    }
                ],
                "model": "gpt-4o",
            }
        }
        raw_response.text = '{"final_result": {...}}'

        logging_obj = MagicMock()

        result = config.transform_response(
            model="gpt-4o",  # GPT model - should NOT strip
            raw_response=raw_response,
            model_response=ModelResponse(id="test", model="test"),
            logging_obj=logging_obj,
            request_data={},
            messages=[{"role": "user", "content": "test"}],
            optional_params={"response_format": {"type": "json_object"}},
            litellm_params={},
            encoding=None,
        )

        # Markdown should NOT be stripped for GPT models
        assert result.choices[0].message.content == '```json\n{"answer": 42}\n```'

    def test_gemini_model_no_markdown_strip(self):
        """Gemini models should NOT have markdown stripped."""
        from unittest.mock import MagicMock
        from litellm.types.utils import ModelResponse

        config = GenAIHubOrchestrationConfig()

        raw_response = MagicMock()
        raw_response.json.return_value = {
            "final_result": {
                "id": "test-id",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": '```json\n{"data": "gemini"}\n```',
                        },
                        "finish_reason": "stop",
                    }
                ],
                "model": "gemini-1.5-pro",
            }
        }
        raw_response.text = '{"final_result": {...}}'

        logging_obj = MagicMock()

        result = config.transform_response(
            model="gemini-1.5-pro",  # Gemini model - should NOT strip
            raw_response=raw_response,
            model_response=ModelResponse(id="test", model="test"),
            logging_obj=logging_obj,
            request_data={},
            messages=[{"role": "user", "content": "test"}],
            optional_params={"response_format": {"type": "json_object"}},
            litellm_params={},
            encoding=None,
        )

        # Markdown should NOT be stripped for Gemini models
        assert result.choices[0].message.content == '```json\n{"data": "gemini"}\n```'

    def test_mistral_model_no_markdown_strip(self):
        """Mistral models should NOT have markdown stripped."""
        from unittest.mock import MagicMock
        from litellm.types.utils import ModelResponse

        config = GenAIHubOrchestrationConfig()

        raw_response = MagicMock()
        raw_response.json.return_value = {
            "final_result": {
                "id": "test-id",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": '```json\n{"model": "mistral"}\n```',
                        },
                        "finish_reason": "stop",
                    }
                ],
                "model": "mistral-large",
            }
        }
        raw_response.text = '{"final_result": {...}}'

        logging_obj = MagicMock()

        result = config.transform_response(
            model="mistral-large",  # Mistral model - should NOT strip
            raw_response=raw_response,
            model_response=ModelResponse(id="test", model="test"),
            logging_obj=logging_obj,
            request_data={},
            messages=[{"role": "user", "content": "test"}],
            optional_params={
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {"name": "test", "schema": {}},
                }
            },
            litellm_params={},
            encoding=None,
        )

        # Markdown should NOT be stripped for Mistral models
        assert result.choices[0].message.content == '```json\n{"model": "mistral"}\n```'

    def test_anthropic_model_still_strips_markdown(self):
        """Anthropic models should still have markdown stripped (existing behavior)."""
        from unittest.mock import MagicMock
        from litellm.types.utils import ModelResponse

        config = GenAIHubOrchestrationConfig()

        raw_response = MagicMock()
        raw_response.json.return_value = {
            "final_result": {
                "id": "test-id",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": '```json\n{"result": "anthropic"}\n```',
                        },
                        "finish_reason": "stop",
                    }
                ],
                "model": "anthropic--claude-3-5-sonnet",
            }
        }
        raw_response.text = '{"final_result": {...}}'

        logging_obj = MagicMock()

        result = config.transform_response(
            model="anthropic--claude-3-5-sonnet",  # Anthropic - SHOULD strip
            raw_response=raw_response,
            model_response=ModelResponse(id="test", model="test"),
            logging_obj=logging_obj,
            request_data={},
            messages=[{"role": "user", "content": "test"}],
            optional_params={
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {"name": "test", "schema": {}},
                }
            },
            litellm_params={},
            encoding=None,
        )

        # Markdown SHOULD be stripped for Anthropic models
        assert result.choices[0].message.content == '{"result": "anthropic"}'

    def test_anthropic_claude_4_strips_markdown(self):
        """Claude 4 models should have markdown stripped."""
        from unittest.mock import MagicMock
        from litellm.types.utils import ModelResponse

        config = GenAIHubOrchestrationConfig()

        raw_response = MagicMock()
        raw_response.json.return_value = {
            "final_result": {
                "id": "test-id",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": '```json\n{"model": "claude-4"}\n```',
                        },
                        "finish_reason": "stop",
                    }
                ],
                "model": "anthropic--claude-4.5-sonnet",
            }
        }
        raw_response.text = '{"final_result": {...}}'

        logging_obj = MagicMock()

        result = config.transform_response(
            model="anthropic--claude-4.5-sonnet",  # Claude 4 Anthropic - SHOULD strip
            raw_response=raw_response,
            model_response=ModelResponse(id="test", model="test"),
            logging_obj=logging_obj,
            request_data={},
            messages=[{"role": "user", "content": "test"}],
            optional_params={"response_format": {"type": "json_object"}},
            litellm_params={},
            encoding=None,
        )

        # Markdown SHOULD be stripped for Anthropic models
        assert result.choices[0].message.content == '{"model": "claude-4"}'
