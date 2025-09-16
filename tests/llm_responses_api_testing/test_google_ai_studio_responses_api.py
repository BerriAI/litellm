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

    
    


