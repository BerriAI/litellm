import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock
import sys
import os

sys.path.insert(0, os.path.abspath("../../.."))

from litellm.proxy.utils import PrismaClient
from litellm.proxy.health_endpoints._health_endpoints import _save_health_check_to_db


@pytest.fixture
def mock_prisma():
    """Simplified mock PrismaClient with bound methods"""
    client = MagicMock()
    client.db.litellm_healthchecktable.create = AsyncMock(return_value={"id": "test-id"})
    client.db.litellm_healthchecktable.find_many = AsyncMock(return_value=[{"id": "1", "model_name": "test"}])
    
    # Bind actual methods
    import types
    for method in ['save_health_check_result', '_validate_response_time', '_clean_details', 
                   'get_health_check_history', 'get_all_latest_health_checks']:
        setattr(client, method, types.MethodType(getattr(PrismaClient, method), client))
    
    return client


@pytest.mark.asyncio
@pytest.mark.parametrize("status,healthy,unhealthy,should_succeed", [
    ("healthy", 1, 0, True),
    ("unhealthy", 0, 1, True),
    ("healthy", 1, 0, False),  # Database error case
])
async def test_save_health_check_result(mock_prisma, status, healthy, unhealthy, should_succeed):
    """Test health check result saving with various scenarios"""
    if not should_succeed:
        mock_prisma.db.litellm_healthchecktable.create.side_effect = Exception("DB Error")
    
    result = await mock_prisma.save_health_check_result(
        model_name="test-model", status=status, healthy_count=healthy, unhealthy_count=unhealthy
    )
    
    if should_succeed:
        mock_prisma.db.litellm_healthchecktable.create.assert_called_once()
    else:
        assert result is None


@pytest.mark.asyncio
async def test_get_health_check_history(mock_prisma):
    """Test health check history retrieval"""
    result = await mock_prisma.get_health_check_history(model_name="test", limit=50)
    mock_prisma.db.litellm_healthchecktable.find_many.assert_called_once()
    assert len(result) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("healthy_count,unhealthy_count,expected_status", [
    (1, 0, "healthy"),
    (0, 1, "unhealthy"),
    (2, 1, "healthy"),
])
async def test_save_health_check_to_db(healthy_count, unhealthy_count, expected_status):
    """Test _save_health_check_to_db function with different endpoint counts"""
    mock_client = MagicMock()
    mock_client.save_health_check_result = AsyncMock()
    
    healthy_endpoints = [{"model": "test"}] * healthy_count
    unhealthy_endpoints = [{"error": "test error"}] * unhealthy_count
    
    await _save_health_check_to_db(
        mock_client, "test-model", healthy_endpoints, unhealthy_endpoints, 
        1234567890.0, "test-user"
    )
    
    call_args = mock_client.save_health_check_result.call_args[1]
    assert call_args["status"] == expected_status
    assert call_args["healthy_count"] == healthy_count
    assert call_args["unhealthy_count"] == unhealthy_count


@pytest.mark.asyncio
async def test_save_health_check_to_db_no_client():
    """Test graceful handling when no database client"""
    result = await _save_health_check_to_db(None, "test", [], [], 0.0, "user")
    assert result is None


if __name__ == "__main__":
    pytest.main([__file__]) 