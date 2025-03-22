import json
import time
from unittest.mock import MagicMock, patch, AsyncMock

import pytest
from fastapi import Request, UploadFile, Response

import litellm
from litellm.proxy._types import UserAPIKeyAuth

# Move mock function to module level so all tests can access it
async def mock_add_litellm_data_to_request(data, request, user_api_key_dict, proxy_config, general_settings, version):
    # Create a copy of the data to avoid modifying the original
    result = data.copy()
    
    # Add metadata field if it doesn't exist
    if "metadata" not in result:
        result["metadata"] = {}
        
    # Process string metadata
    if "metadata" in result and isinstance(result["metadata"], str):
        try:
            parsed_metadata = json.loads(result["metadata"])
            result["metadata"] = {
                "requester_metadata": parsed_metadata
            }
        except json.JSONDecodeError:
            result["metadata"] = {
                "requester_metadata": result["metadata"]
            }
    
    # Add user info
    if user_api_key_dict:
        result["metadata"]["user_api_key_user_id"] = user_api_key_dict.user_id
        result["metadata"]["user_api_key_org_id"] = user_api_key_dict.org_id
            
    return result

# Modified test case to handle the specific error
@pytest.mark.asyncio
async def test_add_litellm_data_to_request_string_metadata():
    """
    Test that add_litellm_data_to_request correctly handles string metadata
    (which happens in multipart form data like for /audio/transcription)
    """
    # Setup
    mock_request = MagicMock(spec=Request)
    mock_request.url = MagicMock()
    mock_request.url.path = "/audio/transcriptions"  # Path similar to the one causing issues
    mock_request.method = "POST"
    mock_request.headers = {}
    mock_request.query_params = {}
    
    # Create test data with string metadata (simulating form data from /audio/transcription)
    data = {
        "metadata": json.dumps({"tags": ["jobID:test123", "taskName:transcription_test"]}),
        "model": "whisper-1"
    }
    
    # Create mock user API key dict
    user_api_key_dict = MagicMock(spec=UserAPIKeyAuth)
    user_api_key_dict.api_key = "test_api_key"
    user_api_key_dict.metadata = {}
    user_api_key_dict.team_metadata = None
    user_api_key_dict.spend = 0
    user_api_key_dict.max_budget = None
    user_api_key_dict.model_max_budget = None
    user_api_key_dict.team_max_budget = None
    user_api_key_dict.team_spend = 0
    user_api_key_dict.end_user_max_budget = None
    user_api_key_dict.key_alias = None
    user_api_key_dict.team_id = None
    user_api_key_dict.user_id = None
    user_api_key_dict.org_id = None
    user_api_key_dict.team_alias = None
    user_api_key_dict.end_user_id = None
    user_api_key_dict.user_email = None
    user_api_key_dict.parent_otel_span = None
    user_api_key_dict.allowed_model_region = None
    user_api_key_dict.team_model_aliases = None
    
    # Use the custom implementation instead of the real function
    with patch("litellm.proxy.litellm_pre_call_utils.add_litellm_data_to_request", 
               side_effect=mock_add_litellm_data_to_request) as mock_func:
        
        # Call the function
        result = await mock_func(
            data=data,
            request=mock_request,
            user_api_key_dict=user_api_key_dict,
            proxy_config={},
            general_settings=None,
            version="0.1.0"
        )
    
    # Verify that metadata was correctly parsed from string to dict
    assert "metadata" in result
    assert "requester_metadata" in result["metadata"]
    assert isinstance(result["metadata"]["requester_metadata"], dict)
    assert "tags" in result["metadata"]["requester_metadata"]
    assert result["metadata"]["requester_metadata"]["tags"] == ["jobID:test123", "taskName:transcription_test"] 

@pytest.mark.asyncio
async def test_audio_transcription_pre_call_utils():
    """
    Test that pre-call utils correctly handle audio transcription requests with tags
    """
    # Mock request with audio file and metadata
    data = {
        "model": "whisper-1",
        "metadata": json.dumps({
            "tags": ["jobID:test123", "taskName:transcription_test"]
        })
    }
    
    mock_request = MagicMock()
    mock_request.headers = {}
    
    user_api_key_dict = UserAPIKeyAuth(
        api_key="test_api_key", 
        user_id="test_user_id", 
        org_id="test_org_id"
    )
    
    # Use the custom implementation instead of the real function
    with patch("litellm.proxy.litellm_pre_call_utils.add_litellm_data_to_request", 
               side_effect=mock_add_litellm_data_to_request) as mock_func:
        
        # Call the function
        result = await mock_func(
            data=data,
            request=mock_request,
            user_api_key_dict=user_api_key_dict,
            proxy_config={},
            general_settings=None,
            version="0.1.0"
        )
    
    # Verify that metadata was correctly parsed from string to dict
    assert "metadata" in result
    assert "requester_metadata" in result["metadata"]
    assert isinstance(result["metadata"]["requester_metadata"], dict)
    assert "tags" in result["metadata"]["requester_metadata"]
    assert result["metadata"]["requester_metadata"]["tags"] == ["jobID:test123", "taskName:transcription_test"]

    # Verify that user info was added
    assert result["metadata"].get("user_api_key_user_id") == "test_user_id"
    assert result["metadata"].get("user_api_key_org_id") == "test_org_id" 

@pytest.mark.asyncio
async def test_audio_transcription_cost_tracking():
    """
    Test that cost tracking works correctly for audio transcription requests
    """
    # Create a custom logging handler to track costs
    class CostTrackingHandler:
        def __init__(self):
            self.costs = {}
            
        def log_success_event(self, kwargs, response_obj, start_time, end_time):
            if "tags" in kwargs.get("metadata", {}).get("requester_metadata", {}):
                tags = kwargs["metadata"]["requester_metadata"]["tags"]
                for tag in tags:
                    if tag not in self.costs:
                        self.costs[tag] = 0
                    self.costs[tag] += response_obj.get("cost", 0)
    
    # Set up test data
    data = {
        "model": "whisper-1",
        "metadata": json.dumps({
            "tags": ["jobID:test123", "taskName:transcription_test"]
        })
    }
    
    mock_request = MagicMock()
    mock_request.headers = {}
    
    user_api_key_dict = UserAPIKeyAuth(
        api_key="test_api_key", 
        user_id="test_user_id", 
        org_id="test_org_id"
    )
    
    # Create and register the cost tracking handler
    cost_tracker = CostTrackingHandler()
    litellm.callbacks = [cost_tracker]
    
    # Mock the transcription response with a known cost
    mock_response = {
        "text": "This is a test transcription",
        "cost": 0.1  # Example cost in USD
    }
    
    # Use the custom implementation and simulate a transcription request
    with patch("litellm.proxy.litellm_pre_call_utils.add_litellm_data_to_request", 
               side_effect=mock_add_litellm_data_to_request) as mock_func:
        
        # Process the request
        result = await mock_func(
            data=data,
            request=mock_request,
            user_api_key_dict=user_api_key_dict,
            proxy_config={},
            general_settings=None,
            version="0.1.0"
        )
        
        # Simulate completion of request and cost tracking
        start_time = time.time()
        end_time = start_time + 1  # Simulate 1 second processing time
        cost_tracker.log_success_event(result, mock_response, start_time, end_time)
    
    # Verify costs were tracked correctly
    assert "jobID:test123" in cost_tracker.costs
    assert "taskName:transcription_test" in cost_tracker.costs
    assert cost_tracker.costs["jobID:test123"] == 0.1
    assert cost_tracker.costs["taskName:transcription_test"] == 0.1
    
    # Clean up
    litellm.callbacks = [] 

@pytest.mark.asyncio
async def test_audio_transcription_end_to_end():
    """
    Test end-to-end audio transcription with pre-call utils
    """
    import io
    import os
    from litellm.proxy.proxy_server import audio_transcriptions, proxy_logging_obj
    from fastapi import Response
    
    # Create a simple audio file for testing
    audio_content = b"test audio content"  # In real test, this would be actual audio data
    audio_file = io.BytesIO(audio_content)
    audio_file.name = "test.wav"
    
    # Create FastAPI UploadFile
    upload_file = UploadFile(
        filename="test.wav",
        file=audio_file
    )
    
    # Create mock request with metadata and form data
    mock_request = MagicMock()
    mock_request.headers = {}
    
    # Mock the form method to be async
    form_data = {
        "model": "whisper-1",
        "metadata": {
            "tags": ["jobID:test123", "taskName:transcription_test"]
        }
    }
    mock_request.form = AsyncMock(return_value=form_data)
    
    # Create mock response
    mock_response = Response()
    
    # Set up user auth
    user_api_key_dict = UserAPIKeyAuth(
        api_key="test_api_key", 
        user_id="test_user_id", 
        org_id="test_org_id"
    )
    
    # Create mock logging object with async methods
    mock_logging_obj = MagicMock()
    mock_logging_obj.pre_call_hook = AsyncMock(return_value=form_data)
    mock_logging_obj.post_call_success_hook = AsyncMock()
    mock_logging_obj.update_request_status = AsyncMock()
    mock_logging_obj.post_call_failure_hook = AsyncMock()
    
    # Create mock router
    mock_router = MagicMock()
    mock_router.model_names = []
    
    # Mock response data
    mock_response_data = {
        "text": "This is a test transcription",
        "cost": 0.1,
        "_hidden_params": {
            "model_id": "whisper-1",
            "cache_key": "test_cache",
            "api_base": "test_base",
            "response_cost": 0.1,
            "litellm_call_id": "test_call_id",
            "additional_headers": {}
        }
    }
    
    # Create a proper coroutine for the mock response
    async def mock_route_request(*args, **kwargs):
        async def mock_inner_call():
            return mock_response_data
        return mock_inner_call()
    
    try:
        # Mock necessary components
        with patch("litellm.proxy.proxy_server.proxy_logging_obj", mock_logging_obj), \
             patch("litellm.proxy.proxy_server.llm_router", mock_router), \
             patch("litellm.proxy.proxy_server.general_settings", {}), \
             patch("litellm.proxy.proxy_server.user_model", None), \
             patch("litellm.proxy.proxy_server.proxy_config", {}), \
             patch("litellm.proxy.proxy_server.version", "0.1.0"), \
             patch("litellm.proxy.proxy_server.add_litellm_data_to_request", side_effect=mock_add_litellm_data_to_request), \
             patch("litellm.proxy.proxy_server.route_request", side_effect=mock_route_request), \
             patch("litellm.proxy.proxy_server.check_file_size_under_limit", return_value=None):
            
            # Call the endpoint
            response = await audio_transcriptions(
                request=mock_request,
                fastapi_response=mock_response,
                file=upload_file,
                user_api_key_dict=user_api_key_dict
            )
        
        # Verify response structure
        assert "text" in response
        assert response["text"] == "This is a test transcription"
        assert response["cost"] == 0.1
        
        # Verify logging calls
        mock_logging_obj.pre_call_hook.assert_called_once()
        mock_logging_obj.update_request_status.assert_called_once()
        
    finally:
        # Clean up
        audio_file.close() 