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
    remove_custom_field_from_tools,
    strip_custom_from_tools_list,
)


def test_strip_custom_from_tools_list():
    """
    Ensure strip_custom_from_tools_list removes custom field from tools.

    Claude Code sends custom: {eager_input_streaming: true} or custom: {input_examples: [...]}
    on tool definitions. Bedrock Converse rejects these for some models (e.g. Haiku 4.5)
    with "Extra inputs are not permitted".

    Ref: https://github.com/BerriAI/litellm/issues/23825
    Ref: https://github.com/BerriAI/litellm/issues/16679
    """
    # Case 1: OpenAI format - custom at tool level
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather",
                "parameters": {"type": "object", "properties": {}},
            },
            "custom": {"eager_input_streaming": True},
        },
    ]
    strip_custom_from_tools_list(tools)
    assert "custom" not in tools[0]
    assert tools[0]["function"]["name"] == "get_weather"

    # Case 2: OpenAI format - custom nested in function
    tools2 = [
        {
            "type": "function",
            "function": {
                "name": "search",
                "parameters": {},
                "custom": {"input_examples": [{"query": "test"}]},
            },
        },
    ]
    strip_custom_from_tools_list(tools2)
    assert "custom" not in tools2[0]["function"]

    # Case 3: Anthropic format - custom at top level
    tools3 = [
        {
            "name": "some_tool",
            "input_schema": {"type": "object"},
            "custom": {"eager_input_streaming": True},
        },
    ]
    strip_custom_from_tools_list(tools3)
    assert "custom" not in tools3[0]
    assert tools3[0]["name"] == "some_tool"

    # Case 4: empty list (no-op)
    tools4 = []
    strip_custom_from_tools_list(tools4)
    assert tools4 == []

    # Case 5: None (no-op)
    strip_custom_from_tools_list(None)  # type: ignore


def test_remove_custom_field_from_tools_uses_strip_custom_from_tools_list():
    """Ensure remove_custom_field_from_tools delegates to strip_custom_from_tools_list."""
    request = {
        "tools": [
            {"name": "tool1", "input_schema": {}, "custom": {"eager_input_streaming": True}},
        ]
    }
    remove_custom_field_from_tools(request)
    assert "custom" not in request["tools"][0]


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


