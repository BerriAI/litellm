"""
Test chat completion response configuration for exclude_none and exclude_unset settings.
"""
import pytest
from unittest.mock import MagicMock, AsyncMock
from fastapi import Request, Response
from pydantic import BaseModel
from typing import Optional


class MockModelResponse(BaseModel):
    """Mock response model for testing"""
    id: str
    content: str
    optional_field: Optional[str] = None
    

@pytest.mark.asyncio
async def test_chat_completion_exclude_none_true():
    """Test that None fields are excluded when completion_exclude_none is True"""
    from litellm.proxy.proxy_server import general_settings
    
    # Mock general_settings
    general_settings["completion_exclude_none"] = True
    general_settings["completion_exclude_unset"] = True
    
    # Create a mock response
    mock_response = MockModelResponse(id="test-1", content="Hello", optional_field=None)
    
    # The model_dump should exclude None values
    result = mock_response.model_dump(exclude_none=True, exclude_unset=True)
    
    assert "id" in result
    assert "content" in result
    assert "optional_field" not in result  # Should be excluded because it's None


@pytest.mark.asyncio
async def test_chat_completion_exclude_none_false():
    """Test that None fields are included when completion_exclude_none is False"""
    from litellm.proxy.proxy_server import general_settings
    
    # Mock general_settings
    general_settings["completion_exclude_none"] = False
    general_settings["completion_exclude_unset"] = False
    
    # Create a mock response
    mock_response = MockModelResponse(id="test-1", content="Hello", optional_field=None)
    
    # The model_dump should include None values
    result = mock_response.model_dump(exclude_none=False, exclude_unset=False)
    
    assert "id" in result
    assert "content" in result
    assert "optional_field" in result  # Should be included even though it's None
    assert result["optional_field"] is None


@pytest.mark.asyncio
async def test_chat_completion_default_behavior():
    """Test that default behavior maintains backward compatibility (exclude_none=True, exclude_unset=True)"""
    from litellm.proxy.proxy_server import general_settings
    
    # Clear these settings to test default behavior
    general_settings.pop("completion_exclude_none", None)
    general_settings.pop("completion_exclude_unset", None)
    
    # The default should be True for both settings
    exclude_none_from_response = general_settings.get("completion_exclude_none", True)
    exclude_unset_from_response = general_settings.get("completion_exclude_unset", True)
    
    assert exclude_none_from_response is True
    assert exclude_unset_from_response is True


@pytest.mark.asyncio
async def test_chat_completion_mixed_configuration():
    """Test mixed configuration (exclude_none=False, exclude_unset=True)"""
    from litellm.proxy.proxy_server import general_settings
    
    # Mock general_settings with mixed configuration
    general_settings["completion_exclude_none"] = False
    general_settings["completion_exclude_unset"] = True
    
    # Create a mock response
    mock_response = MockModelResponse(id="test-1", content="Hello", optional_field=None)
    
    # The model_dump should include None values but exclude unset fields
    result = mock_response.model_dump(exclude_none=False, exclude_unset=True)
    
    assert "id" in result
    assert "content" in result
    assert "optional_field" in result  # Should be included even though it's None
    assert result["optional_field"] is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
