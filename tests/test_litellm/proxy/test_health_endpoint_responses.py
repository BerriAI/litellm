import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import sys
import os
from datetime import datetime

sys.path.insert(0, os.path.abspath("../../.."))

from litellm.proxy.health_endpoints._health_endpoints import (
    latest_health_checks_endpoint,
    _convert_health_check_to_dict
)


class MockHealthCheck:
    """Mock health check object that mimics database record"""
    def __init__(self, model_name, model_id, status="healthy"):
        self.health_check_id = "test-id"
        self.model_name = model_name
        self.model_id = model_id
        self.status = status
        self.healthy_count = 1 if status == "healthy" else 0
        self.unhealthy_count = 0 if status == "healthy" else 1
        self.error_message = None if status == "healthy" else "Test error"
        self.response_time_ms = 100
        self.details = {"test": "details"}
        self.checked_by = "test-user"
        self.checked_at = datetime.now()
        self.created_at = datetime.now()


@pytest.mark.asyncio
async def test_latest_health_checks_uses_model_name_as_key():
    """Test that /health/latest endpoint uses model_name as the key in response"""
    
    # Create mock health checks with both model_name and model_id
    mock_checks = [
        MockHealthCheck(model_name="gpt-3.5-turbo", model_id="model-123", status="healthy"),
        MockHealthCheck(model_name="claude-2", model_id="model-456", status="unhealthy"),
        MockHealthCheck(model_name="llama-2", model_id="model-789", status="healthy"),
    ]
    
    # Mock dependencies
    mock_user_api_key_dict = MagicMock()
    mock_user_api_key_dict.user_id = "test-user"
    
    with patch('litellm.proxy.health_endpoints._health_endpoints._check_prisma_client') as mock_check_prisma:
        mock_prisma = MagicMock()
        mock_prisma.get_all_latest_health_checks = AsyncMock(return_value=mock_checks)
        mock_check_prisma.return_value = mock_prisma
        
        # Call the endpoint function directly
        result = await latest_health_checks_endpoint(user_api_key_dict=mock_user_api_key_dict)
        
        # Verify the response structure
        assert "latest_health_checks" in result
        assert "total_models" in result
        assert result["total_models"] == 3
        
        # Verify that model_name is used as key, not model_id
        health_checks = result["latest_health_checks"]
        assert "gpt-3.5-turbo" in health_checks
        assert "claude-2" in health_checks
        assert "llama-2" in health_checks
        
        # Verify model_id is NOT used as key
        assert "model-123" not in health_checks
        assert "model-456" not in health_checks
        assert "model-789" not in health_checks
        
        # Verify the data structure for each health check
        gpt_check = health_checks["gpt-3.5-turbo"]
        assert gpt_check["model_name"] == "gpt-3.5-turbo"
        assert gpt_check["model_id"] == "model-123"
        assert gpt_check["status"] == "healthy"
        assert gpt_check["healthy_count"] == 1
        assert gpt_check["unhealthy_count"] == 0
        
        claude_check = health_checks["claude-2"]
        assert claude_check["status"] == "unhealthy"
        assert claude_check["error_message"] == "Test error"


@pytest.mark.asyncio
async def test_latest_health_checks_handles_missing_model_id():
    """Test that endpoint handles health checks without model_id"""
    
    # Create mock health check without model_id
    mock_checks = [
        MockHealthCheck(model_name="gpt-4", model_id=None, status="healthy"),
    ]
    
    mock_user_api_key_dict = MagicMock()
    mock_user_api_key_dict.user_id = "test-user"
    
    with patch('litellm.proxy.health_endpoints._health_endpoints._check_prisma_client') as mock_check_prisma:
        mock_prisma = MagicMock()
        mock_prisma.get_all_latest_health_checks = AsyncMock(return_value=mock_checks)
        mock_check_prisma.return_value = mock_prisma
        
        result = await latest_health_checks_endpoint(user_api_key_dict=mock_user_api_key_dict)
        
        # Verify model_name is still used as key when model_id is None
        health_checks = result["latest_health_checks"]
        assert "gpt-4" in health_checks
        assert health_checks["gpt-4"]["model_id"] is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])