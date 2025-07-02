import pytest
import time
from litellm.llms.openai.responses.transformation import OpenAIResponsesAPIConfig
from litellm.types.llms.openai import ResponsesAPIResponse
import httpx

def test_created_at_conversion():
    """Test that created_at is properly converted to an integer"""
    config = OpenAIResponsesAPIConfig()
    
    # Test with float timestamp
    mock_response = httpx.Response(
        status_code=200,
        json=lambda: {
            "id": "test_id",
            "created_at": 1751443898.0,
            "model": "test-model",
            "object": "response",
            "output": [],
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": []
        }
    )
    
    result = config.transform_response_api_response(
        model="test-model",
        raw_response=mock_response,
        logging_obj=None
    )
    
    assert isinstance(result.created_at, int)
    assert result.created_at == 1751443898

def test_store_field_presence():
    """Test that store field is present in the response object"""
    # Create a response object with store=True
    response = ResponsesAPIResponse(
        id="test_id",
        created_at=int(time.time()),
        model="test-model",
        object="response",
        output=[],
        parallel_tool_calls=True,
        tool_choice="auto",
        tools=[],
        store=True
    )
    
    # Verify store field is present and has the correct value
    assert hasattr(response, "store")
    assert response.store is True
    
    # Create a response object without store field
    response_without_store = ResponsesAPIResponse(
        id="test_id",
        created_at=int(time.time()),
        model="test-model",
        object="response",
        output=[],
        parallel_tool_calls=True,
        tool_choice="auto",
        tools=[]
    )
    
    # Verify store field is present but None
    assert hasattr(response_without_store, "store")
    assert response_without_store.store is None 