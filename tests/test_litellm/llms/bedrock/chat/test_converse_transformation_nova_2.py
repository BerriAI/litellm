"""
Unit tests for Amazon Nova 2 reasoning configuration transformation.

Tests the _transform_reasoning_effort_to_reasoning_config method in AmazonConverseConfig.
"""

import pytest
import sys
import os

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path

import litellm
from litellm.llms.bedrock.chat.converse_transformation import AmazonConverseConfig


class TestNova15ReasoningTransformation:
    """Test suite for Nova 2 reasoning effort transformation."""

    def test_reasoning_effort_low_transformation(self):
        """Test that reasoning_effort='low' is transformed to correct reasoningConfig structure."""
        config = AmazonConverseConfig()

        result = config._transform_reasoning_effort_to_reasoning_config("low")

        # Verify the structure
        assert "reasoningConfig" in result
        assert result["reasoningConfig"]["type"] == "enabled"
        assert result["reasoningConfig"]["maxReasoningEffort"] == "low"

    def test_reasoning_effort_high_transformation(self):
        """Test that reasoning_effort='high' is transformed to correct reasoningConfig structure."""
        config = AmazonConverseConfig()

        result = config._transform_reasoning_effort_to_reasoning_config("high")

        # Verify the structure
        assert "reasoningConfig" in result
        assert result["reasoningConfig"]["type"] == "enabled"
        assert result["reasoningConfig"]["maxReasoningEffort"] == "high"

    def test_invalid_reasoning_effort_value(self):
        """Test that invalid reasoning_effort values raise BadRequestError."""
        config = AmazonConverseConfig()

        # Test with invalid value "invalid"
        with pytest.raises(litellm.exceptions.BadRequestError) as exc_info:
            config._transform_reasoning_effort_to_reasoning_config("invalid")

        # Verify error message contains the invalid value and valid values
        error_message = str(exc_info.value)
        assert "invalid" in error_message
        assert "low" in error_message
        assert "high" in error_message
        assert "Nova 2" in error_message

    def test_invalid_reasoning_effort_empty_string(self):
        """Test that empty string raises BadRequestError."""
        config = AmazonConverseConfig()

        with pytest.raises(litellm.exceptions.BadRequestError) as exc_info:
            config._transform_reasoning_effort_to_reasoning_config("")

        # Verify error message
        error_message = str(exc_info.value)
        assert "low" in error_message
        assert "high" in error_message

    def test_invalid_reasoning_effort_wrong_case(self):
        """Test that case-sensitive values are rejected (e.g., 'Low' instead of 'low')."""
        config = AmazonConverseConfig()

        with pytest.raises(litellm.exceptions.BadRequestError):
            config._transform_reasoning_effort_to_reasoning_config("Low")

        with pytest.raises(litellm.exceptions.BadRequestError):
            config._transform_reasoning_effort_to_reasoning_config("HIGH")


class TestNova2ParameterMapping:
    """Test suite for Nova 2 parameter mapping integration."""

    def test_nova_2_reasoning_effort_low_mapping(self):
        """Test that reasoning_effort='low' is correctly mapped to reasoningConfig for Nova 2."""
        config = AmazonConverseConfig()

        model = "amazon.nova-2-lite-v1:0"
        non_default_params = {"reasoning_effort": "low"}
        optional_params = {}

        result = config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=model,
            drop_params=False,
        )

        # Verify reasoningConfig is in result
        assert "reasoningConfig" in result
        assert result["reasoningConfig"]["type"] == "enabled"
        assert result["reasoningConfig"]["maxReasoningEffort"] == "low"

        # Verify thinking is NOT in result
        assert "thinking" not in result

        # Verify reasoning_effort is NOT kept as-is (should be transformed)
        assert "reasoning_effort" not in result

    def test_nova_2_reasoning_effort_high_mapping(self):
        """Test that reasoning_effort='high' is correctly mapped to reasoningConfig for Nova 2."""
        config = AmazonConverseConfig()

        model = "amazon.nova-2-lite-v1:0"
        non_default_params = {"reasoning_effort": "high"}
        optional_params = {}

        result = config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=model,
            drop_params=False,
        )

        # Verify reasoningConfig is in result
        assert "reasoningConfig" in result
        assert result["reasoningConfig"]["type"] == "enabled"
        assert result["reasoningConfig"]["maxReasoningEffort"] == "high"

        # Verify thinking is NOT in result
        assert "thinking" not in result

        # Verify reasoning_effort is NOT kept as-is (should be transformed)
        assert "reasoning_effort" not in result

    def test_nova_2_without_reasoning_effort(self):
        """Test that Nova 2 without reasoning_effort has no reasoningConfig in result."""
        config = AmazonConverseConfig()

        model = "amazon.nova-2-lite-v1:0"
        non_default_params = {"temperature": 0.7}
        optional_params = {}

        result = config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=model,
            drop_params=False,
        )

        # Verify reasoningConfig is NOT in result
        assert "reasoningConfig" not in result

        # Verify thinking is NOT in result
        assert "thinking" not in result

        # Verify reasoning_effort is NOT in result
        assert "reasoning_effort" not in result

    def test_nova_2_regional_variant_us(self):
        """Test that US regional variant of Nova 2 works correctly."""
        config = AmazonConverseConfig()

        model = "us.amazon.nova-2-lite-v1:0"
        non_default_params = {"reasoning_effort": "high"}
        optional_params = {}

        result = config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=model,
            drop_params=False,
        )

        # Verify reasoningConfig is in result
        assert "reasoningConfig" in result
        assert result["reasoningConfig"]["type"] == "enabled"
        assert result["reasoningConfig"]["maxReasoningEffort"] == "high"

    def test_nova_2_regional_variant_eu(self):
        """Test that EU regional variant of Nova 2 works correctly."""
        config = AmazonConverseConfig()

        model = "eu.amazon.nova-2-lite-v1:0"
        non_default_params = {"reasoning_effort": "low"}
        optional_params = {}

        result = config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=model,
            drop_params=False,
        )

        # Verify reasoningConfig is in result
        assert "reasoningConfig" in result
        assert result["reasoningConfig"]["type"] == "enabled"
        assert result["reasoningConfig"]["maxReasoningEffort"] == "low"

    def test_nova_2_regional_variant_apac(self):
        """Test that APAC regional variant of Nova 2 works correctly."""
        config = AmazonConverseConfig()

        model = "apac.amazon.nova-2-lite-v1:0"
        non_default_params = {"reasoning_effort": "high"}
        optional_params = {}

        result = config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=model,
            drop_params=False,
        )

        # Verify reasoningConfig is in result
        assert "reasoningConfig" in result
        assert result["reasoningConfig"]["type"] == "enabled"
        assert result["reasoningConfig"]["maxReasoningEffort"] == "high"

    def test_nova_2_with_other_params(self):
        """Test that Nova 2 reasoning works alongside other parameters."""
        config = AmazonConverseConfig()

        model = "amazon.nova-2-lite-v1:0"
        non_default_params = {
            "reasoning_effort": "high",
            "temperature": 0.8,
            "max_tokens": 1000,
            "top_p": 0.9,
        }
        optional_params = {}

        result = config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=model,
            drop_params=False,
        )

        # Verify reasoningConfig is in result
        assert "reasoningConfig" in result
        assert result["reasoningConfig"]["type"] == "enabled"
        assert result["reasoningConfig"]["maxReasoningEffort"] == "high"

        # Verify other params are also present
        assert result["temperature"] == 0.8
        assert result["maxTokens"] == 1000
        assert result["topP"] == 0.9


class TestNova15SupportedParameters:
    """Test suite for Nova 2 supported parameters."""

    def test_nova_2_supports_reasoning_effort(self):
        """Test that Nova 2 model reports reasoning_effort in supported params."""
        config = AmazonConverseConfig()

        model = "amazon.nova-2-lite-v1:0"
        supported_params = config.get_supported_openai_params(model)

        # Verify reasoning_effort is in supported params
        assert "reasoning_effort" in supported_params

        # Verify thinking is NOT in supported params (Nova 2 uses reasoningConfig, not thinking)
        assert "thinking" not in supported_params

    def test_nova_2_regional_variant_us_supported_params(self):
        """Test that US regional variant returns same supported params."""
        config = AmazonConverseConfig()

        model = "us.amazon.nova-2-lite-v1:0"
        supported_params = config.get_supported_openai_params(model)

        # Verify reasoning_effort is in supported params
        assert "reasoning_effort" in supported_params

        # Verify thinking is NOT in supported params
        assert "thinking" not in supported_params

    def test_nova_2_regional_variant_eu_supported_params(self):
        """Test that EU regional variant returns same supported params."""
        config = AmazonConverseConfig()

        model = "eu.amazon.nova-2-lite-v1:0"
        supported_params = config.get_supported_openai_params(model)

        # Verify reasoning_effort is in supported params
        assert "reasoning_effort" in supported_params

        # Verify thinking is NOT in supported params
        assert "thinking" not in supported_params

    def test_nova_2_regional_variant_apac_supported_params(self):
        """Test that APAC regional variant returns same supported params."""
        config = AmazonConverseConfig()

        model = "apac.amazon.nova-2-lite-v1:0"
        supported_params = config.get_supported_openai_params(model)

        # Verify reasoning_effort is in supported params
        assert "reasoning_effort" in supported_params

        # Verify thinking is NOT in supported params
        assert "thinking" not in supported_params

    def test_nova_2_has_standard_params(self):
        """Test that Nova 2 still has all standard supported params."""
        config = AmazonConverseConfig()

        model = "amazon.nova-2-lite-v1:0"
        supported_params = config.get_supported_openai_params(model)

        # Verify standard params are present
        assert "max_tokens" in supported_params
        assert "max_completion_tokens" in supported_params
        assert "stream" in supported_params
        assert "stream_options" in supported_params
        assert "stop" in supported_params
        assert "temperature" in supported_params
        assert "top_p" in supported_params
        assert "tools" in supported_params
        assert "response_format" in supported_params


class TestNova15ResponseParsing:
    """Test suite for Nova 2 response parsing."""

    def test_transform_reasoning_content_single_block(self):
        """Test that reasoning content is extracted correctly from a single block."""
        config = AmazonConverseConfig()

        reasoning_blocks = [
            {"reasoningText": {"text": "Let me think through this step by step..."}}
        ]

        result = config._transform_reasoning_content(reasoning_blocks)

        assert result == "Let me think through this step by step..."

    def test_transform_reasoning_content_multiple_blocks(self):
        """Test that reasoning content is concatenated from multiple blocks."""
        config = AmazonConverseConfig()

        reasoning_blocks = [
            {"reasoningText": {"text": "First, I need to analyze the problem. "}},
            {"reasoningText": {"text": "Then, I'll consider the solution."}},
        ]

        result = config._transform_reasoning_content(reasoning_blocks)

        assert (
            result
            == "First, I need to analyze the problem. Then, I'll consider the solution."
        )

    def test_transform_reasoning_content_empty_blocks(self):
        """Test that empty reasoning blocks return empty string."""
        config = AmazonConverseConfig()

        reasoning_blocks = []

        result = config._transform_reasoning_content(reasoning_blocks)

        assert result == ""

    def test_transform_thinking_blocks_with_text(self):
        """Test that thinking blocks are populated correctly with text."""
        config = AmazonConverseConfig()

        reasoning_blocks = [{"reasoningText": {"text": "My reasoning process..."}}]

        result = config._transform_thinking_blocks(reasoning_blocks)

        assert len(result) == 1
        assert result[0]["type"] == "thinking"
        assert result[0]["thinking"] == "My reasoning process..."
        assert "signature" not in result[0]

    def test_transform_thinking_blocks_with_signature(self):
        """Test that signature field is preserved when present."""
        config = AmazonConverseConfig()

        reasoning_blocks = [
            {
                "reasoningText": {
                    "text": "My reasoning...",
                    "signature": "signature-hash-12345",
                }
            }
        ]

        result = config._transform_thinking_blocks(reasoning_blocks)

        assert len(result) == 1
        assert result[0]["type"] == "thinking"
        assert result[0]["thinking"] == "My reasoning..."
        assert result[0]["signature"] == "signature-hash-12345"

    def test_transform_thinking_blocks_with_redacted_content(self):
        """Test that redacted content blocks are handled correctly."""
        config = AmazonConverseConfig()

        reasoning_blocks = [
            {"reasoningText": {"text": "First part of reasoning..."}},
            {"redactedContent": {}},
            {"reasoningText": {"text": "Second part after redaction..."}},
        ]

        result = config._transform_thinking_blocks(reasoning_blocks)

        assert len(result) == 3
        assert result[0]["type"] == "thinking"
        assert result[0]["thinking"] == "First part of reasoning..."
        assert result[1]["type"] == "redacted_thinking"
        assert result[2]["type"] == "thinking"
        assert result[2]["thinking"] == "Second part after redaction..."

    def test_transform_thinking_blocks_multiple_blocks(self):
        """Test that multiple thinking blocks are all transformed."""
        config = AmazonConverseConfig()

        reasoning_blocks = [
            {"reasoningText": {"text": "Step 1: Analyze the problem"}},
            {
                "reasoningText": {
                    "text": "Step 2: Consider solutions",
                    "signature": "sig-abc",
                }
            },
            {"reasoningText": {"text": "Step 3: Choose best approach"}},
        ]

        result = config._transform_thinking_blocks(reasoning_blocks)

        assert len(result) == 3
        assert all(block["type"] == "thinking" for block in result)
        assert result[0]["thinking"] == "Step 1: Analyze the problem"
        assert result[1]["thinking"] == "Step 2: Consider solutions"
        assert result[1]["signature"] == "sig-abc"
        assert result[2]["thinking"] == "Step 3: Choose best approach"

    def test_transform_thinking_blocks_empty_list(self):
        """Test that empty thinking blocks list returns empty list."""
        config = AmazonConverseConfig()

        reasoning_blocks = []

        result = config._transform_thinking_blocks(reasoning_blocks)

        assert result == []

    def test_response_parsing_integration(self):
        """Test that response parsing works end-to-end with Nova 2 structure."""
        config = AmazonConverseConfig()

        # Simulate a Nova 2 response with reasoning content
        reasoning_blocks = [
            {
                "reasoningText": {
                    "text": "Let me analyze this carefully. ",
                    "signature": "test-signature",
                }
            },
            {"reasoningText": {"text": "Based on my analysis, the answer is clear."}},
        ]

        # Test reasoning content extraction
        reasoning_content = config._transform_reasoning_content(reasoning_blocks)
        assert (
            reasoning_content
            == "Let me analyze this carefully. Based on my analysis, the answer is clear."
        )

        # Test thinking blocks transformation
        thinking_blocks = config._transform_thinking_blocks(reasoning_blocks)
        assert len(thinking_blocks) == 2
        assert thinking_blocks[0]["thinking"] == "Let me analyze this carefully. "
        assert thinking_blocks[0]["signature"] == "test-signature"
        assert (
            thinking_blocks[1]["thinking"]
            == "Based on my analysis, the answer is clear."
        )


class TestNova15StreamingResponseParsing:
    """Test suite for Nova 2 streaming response parsing."""

    def test_streaming_reasoning_content_start_event(self):
        """Test that streaming start event with reasoningContent is handled correctly."""
        from litellm.llms.bedrock.chat.invoke_handler import AWSEventStreamDecoder

        handler = AWSEventStreamDecoder(model="amazon.nova-2-lite-v1:0")

        # Simulate a start event with redacted reasoning content
        chunk_data = {
            "start": {"reasoningContent": {"redactedContent": {}}},
            "contentBlockIndex": 0,
        }

        result = handler.converse_chunk_parser(chunk_data)

        # Verify thinking blocks are populated
        assert result.choices[0].delta.thinking_blocks is not None
        assert len(result.choices[0].delta.thinking_blocks) == 1
        assert result.choices[0].delta.thinking_blocks[0]["type"] == "redacted_thinking"

    def test_streaming_reasoning_content_delta_text(self):
        """Test that streaming delta event with reasoning text is handled correctly."""
        from litellm.llms.bedrock.chat.invoke_handler import AWSEventStreamDecoder

        handler = AWSEventStreamDecoder(model="amazon.nova-2-lite-v1:0")

        # Simulate a delta event with reasoning text
        chunk_data = {
            "delta": {"reasoningContent": {"text": "Let me think about this..."}},
            "contentBlockIndex": 0,
        }

        result = handler.converse_chunk_parser(chunk_data)

        # Verify reasoning content is extracted
        assert result.choices[0].delta.reasoning_content == "Let me think about this..."

        # Verify thinking blocks are populated
        assert result.choices[0].delta.thinking_blocks is not None
        assert len(result.choices[0].delta.thinking_blocks) == 1
        assert result.choices[0].delta.thinking_blocks[0]["type"] == "thinking"
        assert (
            result.choices[0].delta.thinking_blocks[0]["thinking"]
            == "Let me think about this..."
        )

    def test_streaming_reasoning_content_delta_signature(self):
        """Test that streaming delta event with signature is handled correctly."""
        from litellm.llms.bedrock.chat.invoke_handler import AWSEventStreamDecoder

        handler = AWSEventStreamDecoder(model="amazon.nova-2-lite-v1:0")

        # Simulate a delta event with signature
        chunk_data = {
            "delta": {"reasoningContent": {"signature": "signature-hash-xyz"}},
            "contentBlockIndex": 0,
        }

        result = handler.converse_chunk_parser(chunk_data)

        # Verify reasoning content is set to empty string for consistency
        assert result.choices[0].delta.reasoning_content == ""

        # Verify thinking blocks are populated with signature
        assert result.choices[0].delta.thinking_blocks is not None
        assert len(result.choices[0].delta.thinking_blocks) == 1
        assert result.choices[0].delta.thinking_blocks[0]["type"] == "thinking"
        assert (
            result.choices[0].delta.thinking_blocks[0]["signature"]
            == "signature-hash-xyz"
        )
        assert result.choices[0].delta.thinking_blocks[0]["thinking"] == ""

    def test_streaming_reasoning_content_multiple_deltas(self):
        """Test that multiple reasoning content deltas are accumulated correctly."""
        from litellm.llms.bedrock.chat.invoke_handler import AWSEventStreamDecoder

        handler = AWSEventStreamDecoder(model="amazon.nova-2-lite-v1:0")

        # Simulate multiple delta events
        chunks = [
            {
                "delta": {"reasoningContent": {"text": "First, "}},
                "contentBlockIndex": 0,
            },
            {
                "delta": {"reasoningContent": {"text": "I need to analyze "}},
                "contentBlockIndex": 0,
            },
            {
                "delta": {"reasoningContent": {"text": "the problem."}},
                "contentBlockIndex": 0,
            },
        ]

        results = []
        for chunk_data in chunks:
            result = handler.converse_chunk_parser(chunk_data)
            results.append(result)

        # Verify each delta has the correct reasoning content
        assert results[0].choices[0].delta.reasoning_content == "First, "
        assert results[1].choices[0].delta.reasoning_content == "I need to analyze "
        assert results[2].choices[0].delta.reasoning_content == "the problem."

        # Verify thinking blocks are populated for each delta
        for result in results:
            assert result.choices[0].delta.thinking_blocks is not None
            assert len(result.choices[0].delta.thinking_blocks) == 1
            assert result.choices[0].delta.thinking_blocks[0]["type"] == "thinking"

    def test_streaming_reasoning_then_text_content(self):
        """Test that reasoning content followed by text content is handled correctly."""
        from litellm.llms.bedrock.chat.invoke_handler import AWSEventStreamDecoder

        handler = AWSEventStreamDecoder(model="amazon.nova-2-lite-v1:0")

        # Simulate reasoning content followed by text content
        chunks = [
            {
                "delta": {"reasoningContent": {"text": "Let me think..."}},
                "contentBlockIndex": 0,
            },
            {"delta": {"text": "Based on my reasoning, "}, "contentBlockIndex": 1},
            {"delta": {"text": "the answer is 42."}, "contentBlockIndex": 1},
        ]

        results = []
        for chunk_data in chunks:
            result = handler.converse_chunk_parser(chunk_data)
            results.append(result)

        # Verify first chunk has reasoning content
        assert results[0].choices[0].delta.reasoning_content == "Let me think..."
        assert results[0].choices[0].delta.thinking_blocks is not None

        # Verify subsequent chunks have text content
        assert results[1].choices[0].delta.content == "Based on my reasoning, "
        assert results[2].choices[0].delta.content == "the answer is 42."

    def test_streaming_redacted_content_delta(self):
        """Test that streaming delta with redacted content is handled correctly."""
        from litellm.llms.bedrock.chat.invoke_handler import AWSEventStreamDecoder

        handler = AWSEventStreamDecoder(model="amazon.nova-2-lite-v1:0")

        # Simulate a delta event with redacted content
        chunk_data = {
            "delta": {"reasoningContent": {"redactedContent": {}}},
            "contentBlockIndex": 0,
        }

        result = handler.converse_chunk_parser(chunk_data)

        # Verify reasoning content is set to empty string for consistency
        assert result.choices[0].delta.reasoning_content == ""

        # Verify thinking blocks contain redacted block
        assert result.choices[0].delta.thinking_blocks is not None
        assert len(result.choices[0].delta.thinking_blocks) == 1
        assert result.choices[0].delta.thinking_blocks[0]["type"] == "redacted_thinking"

    def test_streaming_provider_specific_fields(self):
        """Test that provider_specific_fields are populated in streaming responses."""
        from litellm.llms.bedrock.chat.invoke_handler import AWSEventStreamDecoder

        handler = AWSEventStreamDecoder(model="amazon.nova-2-lite-v1:0")

        # Simulate a delta event with reasoning content
        chunk_data = {
            "delta": {"reasoningContent": {"text": "Reasoning text"}},
            "contentBlockIndex": 0,
        }

        result = handler.converse_chunk_parser(chunk_data)

        # Verify provider_specific_fields are populated
        assert result.choices[0].delta.provider_specific_fields is not None
        assert "reasoningContent" in result.choices[0].delta.provider_specific_fields
        assert (
            result.choices[0].delta.provider_specific_fields["reasoningContent"]["text"]
            == "Reasoning text"
        )

    def test_streaming_mixed_content_blocks(self):
        """Test streaming with mixed content blocks (reasoning, text, tool calls)."""
        from litellm.llms.bedrock.chat.invoke_handler import AWSEventStreamDecoder

        handler = AWSEventStreamDecoder(model="amazon.nova-2-lite-v1:0")

        # Simulate a complex streaming scenario
        chunks = [
            # Start with reasoning
            {
                "delta": {
                    "reasoningContent": {
                        "text": "I need to call a tool to get information."
                    }
                },
                "contentBlockIndex": 0,
            },
            # Tool use start
            {
                "start": {"toolUse": {"toolUseId": "tool-123", "name": "get_weather"}},
                "contentBlockIndex": 1,
            },
            # Tool use delta
            {
                "delta": {"toolUse": {"input": '{"location": "NYC"}'}},
                "contentBlockIndex": 1,
            },
            # Text response
            {"delta": {"text": "The weather is sunny."}, "contentBlockIndex": 2},
        ]

        results = []
        for chunk_data in chunks:
            result = handler.converse_chunk_parser(chunk_data)
            results.append(result)

        # Verify reasoning content in first chunk
        assert (
            results[0].choices[0].delta.reasoning_content
            == "I need to call a tool to get information."
        )

        # Verify tool call in second and third chunks
        assert results[1].choices[0].delta.tool_calls is not None
        assert (
            results[1].choices[0].delta.tool_calls[0]["function"]["name"]
            == "get_weather"
        )
        assert results[2].choices[0].delta.tool_calls is not None

        # Verify text content in fourth chunk
        assert results[3].choices[0].delta.content == "The weather is sunny."

    def test_extract_reasoning_content_str_with_text(self):
        """Test extract_reasoning_content_str method with text."""
        from litellm.llms.bedrock.chat.invoke_handler import AWSEventStreamDecoder

        handler = AWSEventStreamDecoder(model="amazon.nova-2-lite-v1:0")

        reasoning_block = {"text": "This is reasoning text"}

        result = handler.extract_reasoning_content_str(reasoning_block)

        assert result == "This is reasoning text"

    def test_extract_reasoning_content_str_without_text(self):
        """Test extract_reasoning_content_str method without text (e.g., signature only)."""
        from litellm.llms.bedrock.chat.invoke_handler import AWSEventStreamDecoder

        handler = AWSEventStreamDecoder(model="amazon.nova-2-lite-v1:0")

        reasoning_block = {"signature": "sig-123"}

        result = handler.extract_reasoning_content_str(reasoning_block)

        assert result is None

    def test_translate_thinking_blocks_streaming_text(self):
        """Test translate_thinking_blocks method with text."""
        from litellm.llms.bedrock.chat.invoke_handler import AWSEventStreamDecoder

        handler = AWSEventStreamDecoder(model="amazon.nova-2-lite-v1:0")

        thinking_block = {"text": "Thinking content"}

        result = handler.translate_thinking_blocks(thinking_block)

        assert result is not None
        assert len(result) == 1
        assert result[0]["type"] == "thinking"
        assert result[0]["thinking"] == "Thinking content"

    def test_translate_thinking_blocks_streaming_signature(self):
        """Test translate_thinking_blocks method with signature."""
        from litellm.llms.bedrock.chat.invoke_handler import AWSEventStreamDecoder

        handler = AWSEventStreamDecoder(model="amazon.nova-2-lite-v1:0")

        thinking_block = {"signature": "sig-abc"}

        result = handler.translate_thinking_blocks(thinking_block)

        assert result is not None
        assert len(result) == 1
        assert result[0]["type"] == "thinking"
        assert result[0]["signature"] == "sig-abc"
        assert (
            result[0]["thinking"] == ""
        )  # Empty string for consistency with Anthropic

    def test_translate_thinking_blocks_streaming_redacted(self):
        """Test translate_thinking_blocks method with redacted content."""
        from litellm.llms.bedrock.chat.invoke_handler import AWSEventStreamDecoder

        handler = AWSEventStreamDecoder(model="amazon.nova-2-lite-v1:0")

        thinking_block = {"redactedContent": {}}

        result = handler.translate_thinking_blocks(thinking_block)

        assert result is not None
        assert len(result) == 1
        assert result[0]["type"] == "redacted_thinking"
