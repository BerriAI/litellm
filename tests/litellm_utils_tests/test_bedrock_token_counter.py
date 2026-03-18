"""
Bedrock Token Counter Tests.

Tests for the Bedrock token counter implementation using the base test suite.

Note: Not all Bedrock models support token counting. The CountTokens API
is only available for specific models. If the model doesn't support token
counting, the test will be skipped.
"""

import os
import sys
from typing import Any, Dict, List

import pytest

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path

from litellm.llms.base_llm.base_utils import BaseTokenCounter
from litellm.llms.bedrock.count_tokens.bedrock_token_counter import BedrockTokenCounter
from tests.litellm_utils_tests.base_token_counter_test import BaseTokenCounterTest


class TestBedrockTokenCounter(BaseTokenCounterTest):
    """Test suite for Bedrock token counter.
    
    Note: Bedrock CountTokens API support varies by model. Some models
    (like older Claude versions) may not support token counting.
    Use amazon.nova-* models for reliable token counting support.
    """

    def get_token_counter(self) -> BaseTokenCounter:
        return BedrockTokenCounter()

    def get_test_model(self) -> str:
        # Use Amazon Nova model which supports token counting
        # Alternatively, use environment variable to override
        return os.getenv("BEDROCK_TEST_MODEL", "amazon.nova-lite-v1:0")

    def get_test_messages(self) -> List[Dict[str, Any]]:
        return [
            {"role": "user", "content": "Hello, how are you today?"}
        ]

    def get_deployment_config(self) -> Dict[str, Any]:
        # Bedrock uses AWS credentials from environment
        # Check for AWS credentials
        aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
        aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        aws_region = os.getenv("AWS_REGION_NAME", "us-east-1")
        
        if not aws_access_key or not aws_secret_key:
            pytest.skip("AWS credentials not set (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)")
            
        return {
            "litellm_params": {
                "aws_access_key_id": aws_access_key,
                "aws_secret_access_key": aws_secret_key,
                "aws_region_name": aws_region,
            }
        }

    def get_custom_llm_provider(self) -> str:
        return "bedrock"

    @pytest.mark.asyncio
    async def test_count_tokens_basic(self):
        """
        Test basic token counting functionality.
        
        Override to handle models that don't support token counting.
        """
        from litellm.types.utils import TokenCountResponse

        token_counter = self.get_token_counter()
        model = self.get_test_model()
        messages = self.get_test_messages()
        deployment = self.get_deployment_config()

        result = await token_counter.count_tokens(
            model_to_use=model,
            messages=messages,
            contents=None,
            deployment=deployment,
            request_model=model,
        )

        print(f"Token count result: {result}")

        assert result is not None, "Token counter should return a result"
        assert isinstance(result, TokenCountResponse), "Result should be TokenCountResponse"
        
        # Check if the model doesn't support token counting
        if result.error and "doesn't support counting tokens" in str(result.error_message):
            pytest.skip(f"Model {model} doesn't support token counting: {result.error_message}")
        
        assert result.total_tokens > 0, f"Token count should be > 0, got {result.total_tokens}"
        assert result.tokenizer_type is not None, "tokenizer_type should be set"
        assert result.error is not True, f"Token counting should not error: {result.error_message}"
