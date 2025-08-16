"""
Test for Vertex AI tool call deepcopy issue with Pydantic models.

This test verifies that the safe_copy_optional_params function correctly handles
Pydantic models that fail with deepcopy by falling back to shallow copy.

Related to issue #12096
"""
import pytest
from pydantic import BaseModel
from typing import List
from litellm.main import safe_copy_optional_params


class ToolOutput(BaseModel):
    """Pydantic model that may cause deepcopy issues with list return types."""
    results: List[str]
    status: str


def test_safe_copy_optional_params_with_pydantic_model():
    """Test that safe_copy_optional_params handles Pydantic models correctly."""
    # Create a tool definition with a Pydantic model that could cause deepcopy issues
    tool_with_pydantic = {
        "type": "function",
        "function": {
            "name": "test_tool",
            "description": "A test tool",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"}
                }
            }
        },
        "_return_type": ToolOutput,  # This could cause deepcopy to fail
    }
    
    optional_params = {
        "tools": [tool_with_pydantic],
        "temperature": 0.7,
        "max_tokens": 100
    }
    
    # This should not raise an exception
    copied_params = safe_copy_optional_params(optional_params)
    
    # Verify the copy worked
    assert copied_params is not optional_params  # Different objects
    assert copied_params["tools"][0]["function"]["name"] == "test_tool"
    assert copied_params["temperature"] == 0.7
    assert copied_params["max_tokens"] == 100


def test_safe_copy_optional_params_with_regular_dict():
    """Test that safe_copy_optional_params works normally with regular dicts."""
    optional_params = {
        "temperature": 0.5,
        "max_tokens": 200,
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "regular_tool",
                    "description": "A regular tool"
                }
            }
        ]
    }
    
    copied_params = safe_copy_optional_params(optional_params)
    
    # Should use deepcopy for regular objects
    assert copied_params is not optional_params
    assert copied_params["tools"] is not optional_params["tools"]
    assert copied_params["tools"][0] is not optional_params["tools"][0]
    assert copied_params["temperature"] == 0.5


def test_safe_copy_optional_params_empty_dict():
    """Test that safe_copy_optional_params handles empty dict."""
    optional_params = {}
    copied_params = safe_copy_optional_params(optional_params)
    
    assert copied_params == {}
    assert copied_params is not optional_params


if __name__ == "__main__":
    test_safe_copy_optional_params_with_pydantic_model()
    test_safe_copy_optional_params_with_regular_dict()
    test_safe_copy_optional_params_empty_dict()
    print("All tests passed!")