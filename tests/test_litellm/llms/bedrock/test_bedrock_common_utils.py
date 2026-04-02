import json
import os
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(
    0, os.path.abspath("../../../..")
)  # Adds the parent directory to the system path


from litellm.llms.bedrock.common_utils import (
    BedrockModelInfo,
    apply_embedded_bedrock_region_from_model_path,
)


def test_apply_embedded_bedrock_region_strips_prefix_and_sets_region():
    optional_params: dict = {}
    out = apply_embedded_bedrock_region_from_model_path(
        "bedrock/us-west-2/mistral.mistral-7b-instruct-v0:2", optional_params
    )
    assert out == "mistral.mistral-7b-instruct-v0:2"
    assert optional_params.get("aws_region_name") == "us-west-2"


def test_apply_embedded_bedrock_region_respects_explicit_aws_region_name():
    optional_params = {"aws_region_name": "us-east-1"}
    out = apply_embedded_bedrock_region_from_model_path(
        "bedrock/us-west-2/mistral.mistral-7b-instruct-v0:2", optional_params
    )
    assert out == "mistral.mistral-7b-instruct-v0:2"
    assert optional_params.get("aws_region_name") == "us-east-1"


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
        model="bedrock/us-gov.anthropic.claude-haiku-4-5-20251001-v1:0"
    )
    assert base_model == "anthropic.claude-haiku-4-5-20251001-v1:0"
    
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


