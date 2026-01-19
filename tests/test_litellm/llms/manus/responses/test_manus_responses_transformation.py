"""
Tests for Manus Responses API transformation

Tests the ManusResponsesAPIConfig class that handles Manus-specific
transformations for the Responses API.

Source: litellm/llms/manus/responses/transformation.py
"""
import os
import sys

sys.path.insert(0, os.path.abspath("../../../../.."))

from litellm.llms.manus.responses.transformation import ManusResponsesAPIConfig
from litellm.types.llms.openai import ResponsesAPIOptionalRequestParams
from litellm.types.router import GenericLiteLLMParams


def test_extract_agent_profile():
    """Test that agent profile is correctly extracted from model name"""
    config = ManusResponsesAPIConfig()
    
    assert config._extract_agent_profile("manus/manus-1.6") == "manus-1.6"
    assert config._extract_agent_profile("manus/manus-1.6-lite") == "manus-1.6-lite"
    assert config._extract_agent_profile("manus/manus-1.6-max") == "manus-1.6-max"


def test_transform_responses_api_request_adds_manus_params():
    """Test that transform_responses_api_request adds task_mode and agent_profile"""
    config = ManusResponsesAPIConfig()
    
    input_param = [
        {
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": "What's the color of the sky?",
                }
            ],
        }
    ]
    
    optional_params = ResponsesAPIOptionalRequestParams()
    litellm_params = GenericLiteLLMParams()
    headers = {}
    
    result = config.transform_responses_api_request(
        model="manus/manus-1.6",
        input=input_param,
        response_api_optional_request_params=dict(optional_params),
        litellm_params=litellm_params,
        headers=headers,
    )
    
    assert result["task_mode"] == "agent"
    assert result["agent_profile"] == "manus-1.6"
    assert "input" in result
    assert "model" in result

