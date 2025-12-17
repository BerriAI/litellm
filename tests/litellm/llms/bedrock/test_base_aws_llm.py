"""
Tests for BaseAWSLLM class, specifically model ID extraction for imported models.

Fixes: https://github.com/BerriAI/litellm/issues/17763
"""

import pytest
from litellm.llms.bedrock.base_aws_llm import BaseAWSLLM


class TestBedrockModelIdExtraction:
    """Test model ID extraction and encoding for imported Bedrock models."""

    def test_qwen3_imported_model_strips_prefix(self):
        """
        Test that qwen3/ prefix is stripped from imported model ARN.

        Fixes: https://github.com/BerriAI/litellm/issues/17763
        """
        model = "qwen3/arn:aws:bedrock:eu-central-1:123456789012:imported-model/test-model-123"
        provider = BaseAWSLLM.get_bedrock_invoke_provider(model)

        assert provider == "qwen3"

        model_id = BaseAWSLLM.get_bedrock_model_id(
            model=model,
            provider=provider,
            optional_params={}
        )

        # The ARN should be URL encoded
        assert "arn" in model_id
        assert "imported-model" in model_id
        # Ensure the qwen3/ prefix was stripped (this was the bug)
        assert "qwen3%2F" not in model_id
        assert "qwen3/" not in model_id

    def test_qwen2_imported_model_strips_prefix(self):
        """Test that qwen2/ prefix is stripped from imported model ARN."""
        model = "qwen2/arn:aws:bedrock:us-west-2:123456789012:imported-model/test-model-456"
        provider = BaseAWSLLM.get_bedrock_invoke_provider(model)

        assert provider == "qwen2"

        model_id = BaseAWSLLM.get_bedrock_model_id(
            model=model,
            provider=provider,
            optional_params={}
        )

        # The ARN should be URL encoded
        assert "arn" in model_id
        assert "imported-model" in model_id
        # Ensure the qwen2/ prefix was stripped
        assert "qwen2%2F" not in model_id
        assert "qwen2/" not in model_id

    def test_openai_imported_model_strips_prefix(self):
        """Test that openai/ prefix is stripped from imported model ARN."""
        model = "openai/arn:aws:bedrock:us-east-1:123456789012:imported-model/test-model-789"
        provider = BaseAWSLLM.get_bedrock_invoke_provider(model)

        assert provider == "openai"

        model_id = BaseAWSLLM.get_bedrock_model_id(
            model=model,
            provider=provider,
            optional_params={}
        )

        # The ARN should be URL encoded
        assert "arn" in model_id
        assert "imported-model" in model_id
        # Ensure the openai/ prefix was stripped
        assert "openai%2F" not in model_id
        assert "openai/" not in model_id

    def test_llama_imported_model_strips_prefix(self):
        """Test that llama/ prefix is stripped from imported model ARN."""
        model = "llama/arn:aws:bedrock:us-east-1:123456789012:imported-model/test-model-abc"
        provider = BaseAWSLLM.get_bedrock_invoke_provider(model)

        assert provider == "llama"

        model_id = BaseAWSLLM.get_bedrock_model_id(
            model=model,
            provider=provider,
            optional_params={}
        )

        # The ARN should be URL encoded
        assert "arn" in model_id
        assert "imported-model" in model_id
        # Ensure the llama/ prefix was stripped
        assert "llama%2F" not in model_id
        assert "llama/" not in model_id

    def test_native_model_unchanged(self):
        """Test that native Bedrock model IDs are not modified."""
        model = "qwen.qwen3-30b-v1:0"
        provider = BaseAWSLLM.get_bedrock_invoke_provider(model)

        model_id = BaseAWSLLM.get_bedrock_model_id(
            model=model,
            provider=provider,
            optional_params={}
        )

        # Native model should remain unchanged (just URL encoded)
        assert "qwen" in model_id
        assert "30b" in model_id
