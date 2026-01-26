import json
import os
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(
    0, os.path.abspath("../../../..")
)  # Adds the parent directory to the system path


from litellm.llms.bedrock.common_utils import BedrockModelInfo, is_bedrock_opus_4_5


def test_deepseek_cris():
    """
    Test that DeepSeek models with cross-region inference prefix use converse route
    """
    bedrock_model_info = BedrockModelInfo
    bedrock_route = bedrock_model_info.get_bedrock_route(
        model="bedrock/us.deepseek.r1-v1:0"
    )
    assert bedrock_route == "converse"


def test_govcloud_cross_region_inference_prefix():
    """
    Test that GovCloud models with cross-region inference prefix (us-gov.) are parsed correctly
    """
    bedrock_model_info = BedrockModelInfo
    
    # Test us-gov prefix is stripped correctly for Claude models
    base_model = bedrock_model_info.get_base_model(
        model="bedrock/us-gov.anthropic.claude-3-5-sonnet-20240620-v1:0"
    )
    assert base_model == "anthropic.claude-3-5-sonnet-20240620-v1:0"
    
    # Test us-gov prefix is stripped correctly for different Claude versions
    base_model = bedrock_model_info.get_base_model(
        model="bedrock/us-gov.anthropic.claude-sonnet-4-5-20250929-v1:0"
    )
    assert base_model == "anthropic.claude-sonnet-4-5-20250929-v1:0"
    
    # Test us-gov prefix is stripped correctly for Haiku models
    base_model = bedrock_model_info.get_base_model(
        model="bedrock/us-gov.anthropic.claude-3-haiku-20240307-v1:0"
    )
    assert base_model == "anthropic.claude-3-haiku-20240307-v1:0"
    
    # Test us-gov prefix is stripped correctly for Meta models
    base_model = bedrock_model_info.get_base_model(
        model="bedrock/us-gov.meta.llama3-8b-instruct-v1:0"
    )
    assert base_model == "meta.llama3-8b-instruct-v1:0"


def test_is_bedrock_opus_4_5():
    """
    Test the is_bedrock_opus_4_5 helper function.
    
    This helper is used to determine if Bedrock-specific features like input_examples
    are supported, as they are only available on Claude Opus 4.5.
    """
    # Test Opus 4.5 models (should return True)
    assert is_bedrock_opus_4_5("us.anthropic.claude-opus-4-5-20251101-v1:0") is True
    assert is_bedrock_opus_4_5("anthropic.claude-opus-4-5-20251101-v1:0") is True
    assert is_bedrock_opus_4_5("us.anthropic.claude-opus_4_5-20251101-v1:0") is True
    assert is_bedrock_opus_4_5("CLAUDE-OPUS-4-5") is True  # Case insensitive
    assert is_bedrock_opus_4_5("claude_opus_4_5") is True
    
    # Test non-Opus 4.5 models (should return False)
    assert is_bedrock_opus_4_5("us.anthropic.claude-sonnet-4-5-20250929-v1:0") is False
    assert is_bedrock_opus_4_5("anthropic.claude-3-5-sonnet-20240620-v1:0") is False
    assert is_bedrock_opus_4_5("anthropic.claude-opus-4-0-20250514-v1:0") is False
    assert is_bedrock_opus_4_5("anthropic.claude-opus-4-1-20250514-v1:0") is False
    assert is_bedrock_opus_4_5("anthropic.claude-3-haiku-20240307-v1:0") is False
    assert is_bedrock_opus_4_5("meta.llama3-8b-instruct-v1:0") is False
    
    # Test with bedrock/ prefix
    assert is_bedrock_opus_4_5("bedrock/us.anthropic.claude-opus-4-5-20251101-v1:0") is True
    assert is_bedrock_opus_4_5("bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0") is False
    
    # Test with invoke/ prefix
    assert is_bedrock_opus_4_5("bedrock/invoke/us.anthropic.claude-opus-4-5-20251101-v1:0") is True
    assert is_bedrock_opus_4_5("bedrock/invoke/us.anthropic.claude-sonnet-4-5-20250929-v1:0") is False


