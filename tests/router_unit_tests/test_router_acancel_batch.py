"""
Test router.acancel_batch() functionality

This ensures the router's batch cancellation method has test coverage.
"""
import sys
import os

sys.path.insert(0, os.path.abspath("../.."))

import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from litellm import Router
import litellm


@pytest.fixture
def router():
    """Create a router with a mock deployment"""
    return Router(
        model_list=[
            {
                "model_name": "gpt-4",
                "litellm_params": {
                    "model": "gpt-4",
                    "api_key": "fake-key",
                },
            }
        ]
    )


@pytest.mark.asyncio
async def test_router_acancel_batch(router):
    """Test that router.acancel_batch() calls litellm.acancel_batch with correct params"""
    mock_response = MagicMock()
    mock_response.id = "batch_123"
    mock_response.status = "cancelled"
    
    with patch.object(litellm, "acancel_batch", new_callable=AsyncMock) as mock_cancel:
        mock_cancel.return_value = mock_response
        
        # This tests that the router method exists and can be called
        # The actual API call is mocked
        response = await router.acancel_batch(
            model="gpt-4",
            batch_id="batch_123",
        )
        
        # Verify the mock was called
        assert mock_cancel.called
        assert response.id == "batch_123"
        assert response.status == "cancelled"
