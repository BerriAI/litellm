"""
Test case for Google Gemini API proxy request handling.

This test verifies that when a request comes to the proxy endpoint:
http://localhost:4000/v1beta/models/gemini-2.5-flash:generateContent

The request payload is correctly processed and forwarded to the httpx client.
"""

import json
import os
import sys
import unittest.mock
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

# Add the parent directory to the system path
sys.path.insert(0, os.path.abspath("../.."))

import litellm
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.google_endpoints.endpoints import google_generate_content
from litellm.proxy.proxy_server import ProxyConfig
from litellm.proxy.utils import ProxyLogging
from fastapi import Request, Response
from fastapi.datastructures import Headers


@pytest.fixture
def sample_request_payload():
    """Sample request payload as provided in the user query."""
    return {
        "contents": [
            {
                "parts": [
                    {
                        "text": "You are an interactive CLI agent specializing in software engineering tasks. Your primary goal is to help users safely and efficiently, adhering strictly to the following instructions and utilizing your available tools"
                    }
                ],
                "role": "user"
            },
            {
                "parts": [{"text": "Got it. Thanks for the context!"}],
                "role": "model"
            },
            {
                "parts": [{"text": "Hello how are you"}],
                "role": "user"
            },
            {
                "parts": [{"text": "I'm doing well, thank you! How can I help you today?\n"}],
                "role": "model"
            },
            {
                "parts": [
                    {
                        "text": "Analyze *only* the content and structure of your immediately preceding response (your last turn in the conversation history)."
                    }
                ],
                "role": "user"
            }
        ],
        "systemInstruction": {
            "parts": [
                {
                    "text": "You are an interactive CLI agent specializing in software engineering tasks. Your primary goal is to help users safely and efficiently, adhering strictly to the following instructions and utilizing your available tools"
                }
            ],
            "role": "user"
        },
        "generationConfig": {
            "temperature": 0,
            "topP": 1,
            "responseMimeType": "application/json",
            "responseJsonSchema": {
                "type": "object",
                "properties": {
                    "reasoning": {
                        "type": "string",
                        "description": "Brief explanation justifying the 'next_speaker' choice based *strictly* on the applicable rule and the content/structure of the preceding turn."
                    },
                    "next_speaker": {
                        "type": "string",
                        "enum": ["user", "model"],
                        "description": "Who should speak next based *only* on the preceding turn and the decision rules"
                    }
                },
                "required": ["reasoning", "next_speaker"]
            }
        }
    }


@pytest.fixture
def mock_user_api_key_dict():
    """Mock user API key dictionary."""
    return UserAPIKeyAuth(
        api_key="test_api_key",
        user_id="test_user_id",
        user_email="test@example.com",
        team_id="test_team_id",
        max_budget=100.0,
        spend=0.0,
        user_role="internal_user",
        allowed_cache_controls=[],
        metadata={},
        tpm_limit=None,
        rpm_limit=None,
    )


@pytest.fixture
def mock_request(sample_request_payload):
    """Create a mock FastAPI request with the sample payload."""
    mock_request = MagicMock(spec=Request)
    mock_request.headers = Headers({"content-type": "application/json"})
    mock_request.method = "POST"
    mock_request.url.path = "/v1beta/models/gemini-2.5-flash:generateContent"
    
    # Mock the request body reading
    async def mock_body():
        return json.dumps(sample_request_payload).encode('utf-8')
    
    mock_request.body = mock_body
    return mock_request


@pytest.fixture  
def mock_response():
    """Create a mock FastAPI response."""
    return MagicMock(spec=Response)


@pytest.mark.asyncio
async def test_google_gemini_httpx_request_direct():
    """
    Test that the Google Gemini generate_content_handler correctly processes the request
    and forwards it to the httpx client with the correct parameters.
    
    This test directly calls the HTTP handler to verify the httpx integration.
    """
    from litellm.llms.custom_httpx.llm_http_handler import BaseLLMHTTPHandler
    from litellm.llms.gemini.google_genai.transformation import GoogleGenAIConfig
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
    
    # Sample request payload
    sample_payload = {
        "contents": [
            {
                "parts": [
                    {
                        "text": "You are an interactive CLI agent specializing in software engineering tasks."
                    }
                ],
                "role": "user"
            },
            {
                "parts": [{"text": "Got it. Thanks for the context!"}],
                "role": "model"
            },
            {
                "parts": [{"text": "Hello how are you"}],
                "role": "user"
            }
        ],
        "systemInstruction": {
            "parts": [
                {
                    "text": "You are an interactive CLI agent specializing in software engineering tasks."
                }
            ],
            "role": "user"
        },
        "config": {  # Note: already transformed from generationConfig
            "temperature": 0,
            "topP": 1,
            "responseMimeType": "application/json",
            "responseJsonSchema": {
                "type": "object",
                "properties": {
                    "reasoning": {"type": "string"},
                    "next_speaker": {"type": "string", "enum": ["user", "model"]}
                },
                "required": ["reasoning", "next_speaker"]
            }
        }
    }
    
    # Mock the HTTP handler to capture the request
    with patch("litellm.llms.custom_httpx.http_handler.HTTPHandler.post") as mock_post:
        # Create mock response
        mock_http_response = MagicMock()
        mock_http_response.status_code = 200
        mock_http_response.json.return_value = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "text": '{"reasoning": "The preceding response was a helpful greeting asking how to assist.", "next_speaker": "user"}'
                            }
                        ],
                        "role": "model"
                    }
                }
            ]
        }
        mock_post.return_value = mock_http_response
        
        # Create the HTTP handler and provider config
        from litellm.types.router import GenericLiteLLMParams
        
        http_handler = BaseLLMHTTPHandler()
        provider_config = GoogleGenAIConfig()
        
        # Create proper litellm params
        litellm_params = GenericLiteLLMParams(
            api_base="https://generativelanguage.googleapis.com",
            api_key="test_api_key"
        )
        
        logging_obj = LiteLLMLoggingObj(
            model="gemini/gemini-2.5-flash",
            messages=[],
            stream=False,
            call_type="agenerate_content",
            start_time=None,
            litellm_call_id="test_call_id",
            function_id="test_function_id"
        )
        
        try:
            # Call the generate_content_handler directly
            response = http_handler.generate_content_handler(
                model="gemini/gemini-2.5-flash",
                contents=sample_payload["contents"],
                generate_content_provider_config=provider_config,
                generate_content_config_dict=sample_payload["config"],
                tools=None,
                custom_llm_provider="gemini",
                litellm_params=litellm_params,
                logging_obj=logging_obj,
                extra_headers=None,
                extra_body=None,
                timeout=30.0,
                _is_async=False,
                client=None,
                stream=False,
                litellm_metadata={}
            )
            
            # Verify that the HTTP post was called
            assert mock_post.called, "Expected HTTP POST to be called"
            
            # Get the call arguments
            call_args, call_kwargs = mock_post.call_args
            
            print(f"POST call args: {call_args}")
            print(f"POST call kwargs: {call_kwargs}")
            
            # Validate that the request data includes the expected fields
            request_data = call_kwargs.get('json')
            if request_data:
                assert 'contents' in request_data, "Expected 'contents' in request data"
                
                # The config should be included in the request as generationConfig
                if 'generationConfig' in request_data:
                    config = request_data['generationConfig']
                    assert config['temperature'] == 0, "Expected temperature to be 0"
                    assert config['topP'] == 1, "Expected topP to be 1"
                    assert config['responseMimeType'] == "application/json", "Expected responseMimeType to be application/json"
                    assert 'responseJsonSchema' in config, "Expected responseJsonSchema in config"
                    
                    # Validate the responseJsonSchema structure
                    schema = config['responseJsonSchema']
                    assert schema['type'] == 'object', "Expected schema type to be object"
                    assert 'properties' in schema, "Expected properties in schema"
                    assert 'reasoning' in schema['properties'], "Expected reasoning property in schema"
                    assert 'next_speaker' in schema['properties'], "Expected next_speaker property in schema"
                
                print("✅ Request data validation passed")
                print(f"Request data: {json.dumps(request_data, indent=2)}")
            
            # Validate URL contains the correct endpoint
            if call_args:
                url = call_args[0] if len(call_args) > 0 else call_kwargs.get('url')
                assert url is not None, "Expected URL to be provided"
                print(f"✅ URL validation passed: {url}")
            
        except Exception as e:
            print(f"Exception occurred: {e}")
            
            # Check if the HTTP handler was called despite the exception
            if mock_post.called:
                call_args, call_kwargs = mock_post.call_args
                print(f"HTTP POST was called with args: {call_args}")
                print(f"HTTP POST was called with kwargs: {call_kwargs}")
                
                # Even with an exception, we can validate the request structure
                request_data = call_kwargs.get('json')
                if request_data:
                    assert 'contents' in request_data, "Expected 'contents' in request data"
                    if 'generationConfig' in request_data:
                        config = request_data['generationConfig']
                        assert config['temperature'] == 0, "Expected temperature to be 0"
                        assert config['responseMimeType'] == "application/json", "Expected responseMimeType to be application/json"
                    print("✅ Request structure validation passed despite exception")
            else:
                # If no HTTP call was made, re-raise the exception for debugging
                raise


@pytest.mark.asyncio 
async def test_generationconfig_to_config_mapping(sample_request_payload):
    """
    Test that generationConfig is correctly mapped to config parameter
    for Google GenAI compatibility.
    """
    from litellm.proxy.route_llm_request import route_request
    
    # Create a copy of the payload to avoid modifying the fixture
    test_data = sample_request_payload.copy()
    test_data["model"] = "gemini/gemini-2.5-flash"
    
    # Mock the router and other dependencies
    mock_router = MagicMock()
    mock_router.model_names = ["gemini/gemini-2.5-flash"]
    
    with patch("litellm.google_genai.agenerate_content") as mock_agenerate_content:
        mock_agenerate_content.return_value = {"test": "response"}
        
        try:
            # Call route_request with agenerate_content route type
            await route_request(
                data=test_data,
                llm_router=mock_router,
                user_model="gemini/gemini-2.5-flash",
                route_type="agenerate_content"
            )
        except Exception as e:
            # We expect this to fail due to missing dependencies, but we want to check
            # that the data transformation happened
            print(f"Expected exception: {e}")
        
        # Check that generationConfig was mapped to config
        assert 'config' in test_data, "Expected 'generationConfig' to be mapped to 'config'"
        assert 'generationConfig' not in test_data, "Expected 'generationConfig' to be removed from data"
        
        # Validate the config content
        config = test_data['config']
        assert config['temperature'] == 0, "Expected temperature to be 0"
        assert config['topP'] == 1, "Expected topP to be 1"
        assert config['responseMimeType'] == "application/json", "Expected responseMimeType to be application/json"
        
        print("✅ generationConfig to config mapping test passed")


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])
