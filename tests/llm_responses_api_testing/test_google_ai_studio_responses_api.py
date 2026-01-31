import os
import sys
import pytest
from unittest.mock import patch, AsyncMock
sys.path.insert(0, os.path.abspath("../.."))
import litellm
import json
from base_responses_api import BaseResponsesAPITest
@pytest.mark.asyncio
async def test_basic_google_ai_studio_responses_api_with_tools():
    litellm._turn_on_debug()
    litellm.set_verbose = True
    request_model = "gemini/gemini-2.5-flash"
    response = await litellm.aresponses(
        model=request_model,
        input="what is the latest version of supabase python package and when was it released?",
        tools=[
            {
                "type": "web_search_preview",
                "search_context_size": "low"
            }
        ]
    )
    print("litellm response=", json.dumps(response, indent=4, default=str))


@pytest.mark.asyncio
async def test_mock_basic_google_ai_studio_responses_api_with_tools():
    """
    - Ensure that this is the request that litellm.completion gets when we pass web search options 

    litellm.acompletion(messages=[{'role': 'user', 'content': 'what is the latest version of supabase python package and when was it released?'}], model='gemini-2.5-flash', tools=[], web_search_options={'search_context_size': 'low', 'user_location': None})
    """
    # Mock the acompletion function
    litellm._turn_on_debug()
    mock_response = litellm.ModelResponse(
        id="test-id",
        created=1234567890,
        model="gemini/gemini-2.5-flash",
        object="chat.completion",
        choices=[
            litellm.utils.Choices(
                index=0,
                message=litellm.utils.Message(
                    role="assistant",
                    content="Test response"
                ),
                finish_reason="stop"
            )
        ]
    )
    
    with patch('litellm.acompletion', new_callable=AsyncMock) as mock_acompletion:
        mock_acompletion.return_value = mock_response
        
        request_model = "gemini/gemini-2.5-flash"
        await litellm.aresponses(
            model=request_model,
            input="what is the latest version of supabase python package and when was it released?",
            tools=[
                {
                    "type": "web_search_preview",
                    "search_context_size": "low"
                }
            ]
        )
        
        # Verify that acompletion was called
        assert mock_acompletion.called
        
        # Get the call arguments
        call_args, call_kwargs = mock_acompletion.call_args
        
        # Verify the expected parameters were passed
        print("call kwargs to litellm.completion=", json.dumps(call_kwargs, indent=4, default=str))
        assert "web_search_options" in call_kwargs
        assert call_kwargs["web_search_options"] is not None
        assert call_kwargs["web_search_options"]["search_context_size"] == "low"
        assert call_kwargs["web_search_options"]["user_location"] is None
        
        # Verify other expected parameters
        assert call_kwargs["model"] == "gemini-2.5-flash"
        assert len(call_kwargs["messages"]) == 1
        assert call_kwargs["messages"][0]["role"] == "user"
        assert call_kwargs["messages"][0]["content"] == "what is the latest version of supabase python package and when was it released?"
        assert call_kwargs["tools"] == []  # web search tools are converted to web_search_options, not kept as tools

@pytest.mark.asyncio
async def test_gemini_3_responses_api_with_thought_signatures():
    """
    Test that Gemini 3 Responses API preserves thought signatures in function calls.
    This test verifies that provider_specific_fields with thought_signature are correctly
    preserved when using the Responses API with Gemini 3.
    """
    if not os.getenv("GEMINI_API_KEY"):
        pytest.skip("GEMINI_API_KEY not set")
    
    litellm.set_verbose = False
    request_model = "gemini/gemini-3-pro-preview"
    
    tools = [
        {
            "type": "function",
            "name": "get_weather",
            "description": "Get the current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City and country e.g. Mumbai, India"
                    },
                    "units": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "Units the temperature will be returned in."
                    }
                },
                "required": ["location", "units"],
                "additionalProperties": False,
            },
            "strict": True
        }
    ]
    
    # Step 1: Initial request with tools
    response = await litellm.aresponses(
        model=request_model,
        input="What is the weather in Mumbai?",
        tools=tools,
        reasoning_effort="low"
    )
    
    # Validate response structure
    from litellm.types.llms.openai import ResponsesAPIResponse
    assert isinstance(response, ResponsesAPIResponse), "Response should be a ResponsesAPIResponse"
    assert hasattr(response, "output") or "output" in response, "Response should have 'output' field"
    assert isinstance(response.output, list), "Output should be a list"
    
    # Find function call in output
    function_call_item = None
    for item in response.output:
        # Convert to dict if it's a Pydantic model for easier access
        if hasattr(item, "model_dump"):
            item_dict = item.model_dump()
        elif hasattr(item, "__dict__"):
            item_dict = dict(item) if not isinstance(item, dict) else item
        else:
            item_dict = item if isinstance(item, dict) else {}
        
        if isinstance(item_dict, dict) and item_dict.get("type") == "function_call":
            function_call_item = item_dict
            break
    
    # Verify function call exists
    assert function_call_item is not None, "Response should contain a function_call item"
    assert function_call_item.get("name") == "get_weather", "Function call should be for get_weather"
    
    # Verify thought signature is present in provider_specific_fields
    provider_specific_fields = function_call_item.get("provider_specific_fields")
    assert provider_specific_fields is not None, "Function call should have provider_specific_fields"
    assert "thought_signature" in provider_specific_fields, "provider_specific_fields should contain thought_signature"
    assert isinstance(provider_specific_fields["thought_signature"], str), "thought_signature should be a string"
    assert len(provider_specific_fields["thought_signature"]) > 0, "thought_signature should not be empty"
    
    print(f"✅ Thought signature preserved: {provider_specific_fields['thought_signature'][:50]}...")


@pytest.mark.asyncio
async def test_gemini_3_responses_api_streaming_with_thought_signatures():
    """
    Test that Gemini 3 Responses API preserves thought signatures in streaming mode.
    This test verifies that provider_specific_fields with thought_signature are correctly
    preserved when using streaming Responses API with Gemini 3.
    """
    if not os.getenv("GEMINI_API_KEY"):
        pytest.skip("GEMINI_API_KEY not set")
    
    litellm.set_verbose = False
    request_model = "gemini/gemini-3-pro-preview"
    
    tools = [
        {
            "type": "function",
            "name": "get_weather",
            "description": "Get the current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City and country e.g. Mumbai, India"
                    },
                    "units": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "Units the temperature will be returned in."
                    }
                },
                "required": ["location", "units"],
                "additionalProperties": False,
            },
            "strict": True
        }
    ]
    
    # Step 1: Streaming request with tools
    response_stream = await litellm.aresponses(
        model=request_model,
        input="What is the weather in Mumbai?",
        tools=tools,
        stream=True,
        reasoning_effort="low"
    )
    
    # Collect all chunks
    chunks = []
    completed_response = None
    
    async for chunk in response_stream:
        chunks.append(chunk)
        # Check if this is the completed response event
        if hasattr(chunk, "type") and chunk.type == "response.completed":
            completed_response = chunk.response
        elif isinstance(chunk, dict) and chunk.get("type") == "response.completed":
            completed_response = chunk.get("response")
    
    # Verify we got chunks
    assert len(chunks) > 0, "Should receive at least one chunk"
    
    # If we have a completed response, check for thought signatures
    if completed_response:
        output = completed_response.get("output", [])
        function_call_item = None
        for item in output:
            if isinstance(item, dict) and item.get("type") == "function_call":
                function_call_item = item
                break
        
        if function_call_item:
            provider_specific_fields = function_call_item.get("provider_specific_fields")
            if provider_specific_fields:
                thought_signature = provider_specific_fields.get("thought_signature")
                if thought_signature:
                    assert isinstance(thought_signature, str), "thought_signature should be a string"
                    assert len(thought_signature) > 0, "thought_signature should not be empty"
                    print(f"✅ Streaming thought signature preserved: {thought_signature[:50]}...")
    
    print(f"✅ Collected {len(chunks)} streaming chunks")


class TestGoogleAIStudioResponsesAPITest(BaseResponsesAPITest):
    def get_base_completion_call_args(self):
        #litellm._turn_on_debug()
        return {
            "model": "gemini/gemini-2.5-flash-lite"
        }
    
    async def test_basic_openai_responses_delete_endpoint(self, sync_mode=False):
        pytest.skip("DELETE responses is not supported for Google AI Studio")
    
    async def test_basic_openai_responses_streaming_delete_endpoint(self, sync_mode=False):
        pytest.skip("DELETE responses is not supported for Google AI Studio")

    async def test_basic_openai_responses_get_endpoint(self, sync_mode=False):
        pytest.skip("GET responses is not supported for Google AI Studio")

    async def test_basic_openai_responses_cancel_endpoint(self, sync_mode=False):
        pytest.skip("CANCEL responses is not supported for Google AI Studio")

    async def test_cancel_responses_invalid_response_id(self, sync_mode=False):
        pytest.skip("CANCEL responses is not supported for Google AI Studio")

    
    


