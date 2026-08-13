"""
Unit tests for Amazon Nova 2 reasoning configuration transformation.

Tests request transformation, response parsing, multi-turn message translation,
and model detection for Nova 2 Lite and Nova 2 Pro via the Bedrock Converse API.

Reference: https://docs.aws.amazon.com/nova/latest/nova2-userguide/using-converse-api.html
"""

import pytest
import sys
import os

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path

import httpx
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


class TestNova2ResponseParsing:
    """Test that reasoningContent blocks are parsed into reasoning_content strings."""

    def test_should_extract_single_reasoning_block(self):
        config = AmazonConverseConfig()
        result = config._transform_reasoning_content(
            [{"reasoningText": {"text": "Let me think through this step by step..."}}]
        )
        assert result == "Let me think through this step by step..."

    def test_should_concatenate_multiple_reasoning_blocks(self):
        config = AmazonConverseConfig()
        result = config._transform_reasoning_content(
            [
                {"reasoningText": {"text": "First, I need to analyze the problem. "}},
                {"reasoningText": {"text": "Then, I'll consider the solution."}},
            ]
        )
        assert (
            result
            == "First, I need to analyze the problem. Then, I'll consider the solution."
        )

    def test_should_return_empty_string_for_empty_blocks(self):
        config = AmazonConverseConfig()
        assert config._transform_reasoning_content([]) == ""


class TestNova2StreamingResponseParsing:
    """Test that streaming reasoningContent deltas produce reasoning_content on the delta."""

    def test_should_extract_reasoning_content_from_delta(self):
        from litellm.llms.bedrock.chat.invoke_handler import AWSEventStreamDecoder

        handler = AWSEventStreamDecoder(model="amazon.nova-2-lite-v1:0")
        chunk_data = {
            "delta": {"reasoningContent": {"text": "Let me think about this..."}},
            "contentBlockIndex": 0,
        }
        result = handler.converse_chunk_parser(chunk_data)
        assert result.choices[0].delta.reasoning_content == "Let me think about this..."

    def test_should_accumulate_multiple_reasoning_deltas(self):
        from litellm.llms.bedrock.chat.invoke_handler import AWSEventStreamDecoder

        handler = AWSEventStreamDecoder(model="amazon.nova-2-lite-v1:0")
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
        results = [handler.converse_chunk_parser(c) for c in chunks]
        assert results[0].choices[0].delta.reasoning_content == "First, "
        assert results[1].choices[0].delta.reasoning_content == "I need to analyze "
        assert results[2].choices[0].delta.reasoning_content == "the problem."

    def test_should_stream_reasoning_then_text(self):
        from litellm.llms.bedrock.chat.invoke_handler import AWSEventStreamDecoder

        handler = AWSEventStreamDecoder(model="amazon.nova-2-lite-v1:0")
        chunks = [
            {
                "delta": {"reasoningContent": {"text": "Let me think..."}},
                "contentBlockIndex": 0,
            },
            {"delta": {"text": "Based on my reasoning, "}, "contentBlockIndex": 1},
            {"delta": {"text": "the answer is 42."}, "contentBlockIndex": 1},
        ]
        results = [handler.converse_chunk_parser(c) for c in chunks]
        assert results[0].choices[0].delta.reasoning_content == "Let me think..."
        assert results[1].choices[0].delta.content == "Based on my reasoning, "
        assert results[2].choices[0].delta.content == "the answer is 42."

    def test_should_populate_provider_specific_fields(self):
        from litellm.llms.bedrock.chat.invoke_handler import AWSEventStreamDecoder

        handler = AWSEventStreamDecoder(model="amazon.nova-2-lite-v1:0")
        chunk_data = {
            "delta": {"reasoningContent": {"text": "Reasoning text"}},
            "contentBlockIndex": 0,
        }
        result = handler.converse_chunk_parser(chunk_data)
        psf = result.choices[0].delta.provider_specific_fields
        assert psf is not None
        assert psf["reasoningContent"]["text"] == "Reasoning text"

    def test_should_stream_reasoning_with_tool_calls(self):
        from litellm.llms.bedrock.chat.invoke_handler import AWSEventStreamDecoder

        handler = AWSEventStreamDecoder(model="amazon.nova-2-lite-v1:0")
        chunks = [
            {
                "delta": {"reasoningContent": {"text": "I need to call a tool."}},
                "contentBlockIndex": 0,
            },
            {
                "start": {"toolUse": {"toolUseId": "tool-123", "name": "get_weather"}},
                "contentBlockIndex": 1,
            },
            {
                "delta": {"toolUse": {"input": '{"location": "NYC"}'}},
                "contentBlockIndex": 1,
            },
            {"delta": {"text": "The weather is sunny."}, "contentBlockIndex": 2},
        ]
        results = [handler.converse_chunk_parser(c) for c in chunks]
        assert results[0].choices[0].delta.reasoning_content == "I need to call a tool."
        assert (
            results[1].choices[0].delta.tool_calls[0]["function"]["name"]
            == "get_weather"
        )
        assert results[3].choices[0].delta.content == "The weather is sunny."


# ---------------------------------------------------------------------------
# Model detection — _is_nova_2_model covers both Lite and Pro
# ---------------------------------------------------------------------------

NOVA_2_LITE = "amazon.nova-2-lite-v1:0"
NOVA_2_PRO = "us.amazon.nova-2-pro-preview-20251202-v1:0"


class TestNova2ModelDetection:
    """Verify _is_nova_2_model identifies all Nova 2 variants (lite, pro, regional, routed)."""

    @pytest.mark.parametrize(
        "model",
        [
            "amazon.nova-2-lite-v1:0",
            "amazon.nova-2-pro-preview-20251202-v1:0",
            "us.amazon.nova-2-lite-v1:0",
            "us.amazon.nova-2-pro-preview-20251202-v1:0",
            "eu.amazon.nova-2-lite-v1:0",
            "apac.amazon.nova-2-pro-preview-20251202-v1:0",
            "bedrock/converse/amazon.nova-2-lite-v1:0",
            "bedrock/converse/us.amazon.nova-2-pro-preview-20251202-v1:0",
            "bedrock/amazon.nova-2-lite-v1:0",
            "converse/us.amazon.nova-2-lite-v1:0",
            "converse/amazon.nova-2-pro-preview-20251202-v1:0",
        ],
    )
    def test_should_recognize_nova_2_models(self, model):
        assert AmazonConverseConfig()._is_nova_2_model(model) is True

    @pytest.mark.parametrize(
        "model",
        [
            "amazon.nova-pro-v1:0",
            "amazon.nova-lite-v1:0",
            "amazon.nova-pro-1-5-v1:0",
            "anthropic.claude-3-sonnet-20240229-v1:0",
            "us.amazon.nova-pro-v1:0",
        ],
    )
    def test_should_not_match_non_nova_2_models(self, model):
        assert AmazonConverseConfig()._is_nova_2_model(model) is False


# ---------------------------------------------------------------------------
# End-to-end request body — reasoningConfig in additionalModelRequestFields
# ---------------------------------------------------------------------------


class TestNova2EndToEndRequest:
    """Verify transform_request places reasoningConfig correctly for both model variants."""

    def _build_request(self, model, effort, **extra):
        config = AmazonConverseConfig()
        optional_params = config.map_openai_params(
            non_default_params={"reasoning_effort": effort, **extra},
            optional_params={},
            model=model,
            drop_params=False,
        )
        return config.transform_request(
            model=model,
            messages=[{"role": "user", "content": "What is 2+2?"}],
            optional_params=optional_params,
            litellm_params={},
            headers={},
        )

    @pytest.mark.parametrize("model", [NOVA_2_LITE, NOVA_2_PRO])
    def test_should_place_reasoning_config_in_additional_model_request_fields(
        self, model
    ):
        body = self._build_request(model, "high")
        additional = body.get("additionalModelRequestFields", {})
        assert additional["reasoningConfig"] == {
            "type": "enabled",
            "maxReasoningEffort": "high",
        }
        assert "reasoningConfig" not in body  # not top-level
        assert "thinking" not in body  # not Anthropic-style

    @pytest.mark.parametrize("model", [NOVA_2_LITE, NOVA_2_PRO])
    def test_should_coexist_with_inference_params(self, model):
        body = self._build_request(model, "high", temperature=0.5, max_tokens=512)
        assert (
            body["additionalModelRequestFields"]["reasoningConfig"]["type"] == "enabled"
        )
        inf = body.get("inferenceConfig", {})
        assert inf.get("temperature") == 0.5
        assert inf.get("maxTokens") == 512


# ---------------------------------------------------------------------------
# End-to-end response — reasoningContent parsed to reasoning_content string
# ---------------------------------------------------------------------------


class TestNova2EndToEndResponse:
    """Verify transform_response produces reasoning_content from reasoningContent blocks."""

    def _transform(self, content_blocks, model=NOVA_2_LITE):
        config = AmazonConverseConfig()
        body = {
            "output": {"message": {"role": "assistant", "content": content_blocks}},
            "usage": {"inputTokens": 10, "outputTokens": 50, "totalTokens": 60},
            "stopReason": "end_turn",
            "metrics": {"latencyMs": 100},
        }
        resp = httpx.Response(
            200, json=body, request=httpx.Request("POST", "https://bedrock")
        )
        return config.transform_response(
            model=model,
            raw_response=resp,
            model_response=litellm.ModelResponse(),
            logging_obj=None,
            request_data={},
            messages=[],
            optional_params={},
            litellm_params={},
            encoding=None,
            api_key=None,
            json_mode=None,
        )

    def test_should_extract_reasoning_content_as_string(self):
        result = self._transform(
            [
                {"reasoningContent": {"reasoningText": {"text": "Step 1. "}}},
                {"reasoningContent": {"reasoningText": {"text": "Step 2."}}},
                {"text": "The answer is 4."},
            ]
        )
        msg = result.choices[0].message
        assert msg.content == "The answer is 4."
        assert msg.reasoning_content == "Step 1. Step 2."

    def test_should_include_raw_blocks_in_provider_specific_fields(self):
        result = self._transform(
            [
                {"reasoningContent": {"reasoningText": {"text": "thinking..."}}},
                {"text": "done"},
            ]
        )
        psf = result.choices[0].message.get("provider_specific_fields", {})
        assert "reasoningContentBlocks" in psf

    def test_should_omit_reasoning_content_when_absent(self):
        result = self._transform([{"text": "Plain answer."}])
        assert not getattr(result.choices[0].message, "reasoning_content", None)


# ---------------------------------------------------------------------------
# Multi-turn — reasoning_content round-trips back to Bedrock format
# ---------------------------------------------------------------------------


class TestNova2MultiTurnMessageTranslation:
    """Verify that assistant messages carrying reasoning from a previous turn are
    correctly translated to Bedrock content blocks via _bedrock_converse_messages_pt."""

    def _to_bedrock(self, messages, model=NOVA_2_LITE):
        from litellm.litellm_core_utils.prompt_templates.factory import (
            _bedrock_converse_messages_pt,
        )

        return _bedrock_converse_messages_pt(
            messages=messages,
            model=model,
            llm_provider="bedrock_converse",
        )

    def test_should_inline_unsigned_thinking_blocks_as_text(self):
        """Without a signature, reasoning text becomes a plain text block."""
        bedrock_msgs = self._to_bedrock(
            [
                {"role": "user", "content": "What is 2+2?"},
                {
                    "role": "assistant",
                    "content": "4.",
                    "thinking_blocks": [
                        {"type": "thinking", "thinking": "Simple addition"},
                    ],
                },
                {"role": "user", "content": "Sure?"},
            ]
        )
        assistant = next(m for m in bedrock_msgs if m["role"] == "assistant")
        texts = [b["text"] for b in assistant["content"] if "text" in b]
        assert "Simple addition" in texts
        assert "4." in texts

    def test_should_keep_signed_thinking_blocks_as_reasoning_content(self):
        """With a signature, reasoning is preserved as a reasoningContent block."""
        bedrock_msgs = self._to_bedrock(
            [
                {"role": "user", "content": "What is 2+2?"},
                {
                    "role": "assistant",
                    "content": "4.",
                    "thinking_blocks": [
                        {"type": "thinking", "thinking": "math", "signature": "sig-1"},
                    ],
                },
                {"role": "user", "content": "Sure?"},
            ]
        )
        assistant = next(m for m in bedrock_msgs if m["role"] == "assistant")
        rc_blocks = [b for b in assistant["content"] if "reasoningContent" in b]
        assert len(rc_blocks) >= 1
        assert rc_blocks[0]["reasoningContent"]["reasoningText"]["text"] == "math"
        assert rc_blocks[0]["reasoningContent"]["reasoningText"]["signature"] == "sig-1"

    def test_should_translate_inline_content_list_thinking_type(self):
        """content=[{type:'thinking',...},{type:'text',...}] should also round-trip."""
        bedrock_msgs = self._to_bedrock(
            [
                {"role": "user", "content": "Hi"},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": "hmm", "signature": "sig-2"},
                        {"type": "text", "text": "Hello!"},
                    ],
                },
                {"role": "user", "content": "Bye"},
            ]
        )
        assistant = next(m for m in bedrock_msgs if m["role"] == "assistant")
        rc_blocks = [b for b in assistant["content"] if "reasoningContent" in b]
        text_blocks = [
            b
            for b in assistant["content"]
            if "text" in b and "reasoningContent" not in b
        ]
        assert len(rc_blocks) >= 1
        assert any("Hello!" in b["text"] for b in text_blocks)
