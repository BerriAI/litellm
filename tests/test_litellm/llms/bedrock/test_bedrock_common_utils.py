import json
import os
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(
    0, os.path.abspath("../../../..")
)  # Adds the parent directory to the system path


from litellm.llms.bedrock.common_utils import BedrockModelInfo


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


def test_context_window_suffix_stripped_for_cost_lookup():
    """
    Test that [1m], [200k] etc. context window suffixes are stripped from
    Bedrock model names before cost lookup.

    Models configured like `bedrock/us.anthropic.claude-opus-4-6-v1[1m]`
    should resolve to the base model name so pricing can be found.
    """
    from litellm.llms.bedrock.common_utils import get_bedrock_base_model

    assert (
        get_bedrock_base_model("us.anthropic.claude-opus-4-6-v1[1m]")
        == "anthropic.claude-opus-4-6-v1"
    )
    assert (
        get_bedrock_base_model("us.anthropic.claude-sonnet-4-6[1m]")
        == "anthropic.claude-sonnet-4-6"
    )
    assert (
        get_bedrock_base_model("global.anthropic.claude-opus-4-5-20251101-v1:0[1m]")
        == "anthropic.claude-opus-4-5-20251101-v1:0"
    )
    # Ensure models without suffix are unaffected
    assert (
        get_bedrock_base_model("us.anthropic.claude-opus-4-6-v1")
        == "anthropic.claude-opus-4-6-v1"
    )
    # Ensure :51k throughput suffix still works
    assert (
        get_bedrock_base_model("anthropic.claude-3-5-sonnet-20241022-v2:0:51k")
        == "anthropic.claude-3-5-sonnet-20241022-v2:0"
    )
