import asyncio
import json
import os
import sys
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path

from litellm.proxy.utils import PrismaClient, ProxyLogging
from litellm.proxy.health_endpoints._health_endpoints import _save_health_check_to_db
from litellm._logging import verbose_proxy_logger


@pytest.fixture
def mock_prisma_client():
    """Create a mock PrismaClient for testing"""
    client = MagicMock()
    client.db = MagicMock()
    client.db.litellm_healthchecktable = MagicMock()
    
    # Bind the actual methods from PrismaClient to the mock instance
    import types
    client.save_health_check_result = types.MethodType(PrismaClient.save_health_check_result, client)
    client._validate_response_time = types.MethodType(PrismaClient._validate_response_time, client)
    client._clean_details = types.MethodType(PrismaClient._clean_details, client)
    client.get_health_check_history = types.MethodType(PrismaClient.get_health_check_history, client)
    client.get_all_latest_health_checks = types.MethodType(PrismaClient.get_all_latest_health_checks, client)
    
    return client


@pytest.fixture
def mock_proxy_logging():
    """Create a mock ProxyLogging for testing"""
    return MagicMock(spec=ProxyLogging)


class TestPrismaClientHealthCheckMethods:
    """Test the health check methods added to PrismaClient"""

    @pytest.mark.asyncio
    async def test_save_health_check_result_success(self, mock_prisma_client):
        """Test successful health check result saving"""
        # Mock the database create operation
        mock_result = {
            "id": "test-id",
            "model_name": "gpt-3.5-turbo",
            "status": "healthy",
            "response_time_ms": 150.5,
            "checked_at": datetime.now()
        }
        mock_prisma_client.db.litellm_healthchecktable.create = AsyncMock(return_value=mock_result)
        
        # Call the method
        result = await mock_prisma_client.save_health_check_result(
            model_name="gpt-3.5-turbo",
            status="healthy",
            healthy_count=1,
            unhealthy_count=0,
            response_time_ms=150.5,
            checked_by="test-user"
        )
        
        # Verify the mock was called correctly
        mock_prisma_client.db.litellm_healthchecktable.create.assert_called_once()
        call_args = mock_prisma_client.db.litellm_healthchecktable.create.call_args[1]["data"]
        
        assert call_args["model_name"] == "gpt-3.5-turbo"
        assert call_args["status"] == "healthy"
        assert call_args["healthy_count"] == 1
        assert call_args["unhealthy_count"] == 0
        assert call_args["response_time_ms"] == 150.5
        assert call_args["checked_by"] == "test-user"

    @pytest.mark.asyncio
    async def test_save_health_check_result_database_error(self, mock_prisma_client):
        """Test handling of database errors during save"""
        # Mock database error
        mock_prisma_client.db.litellm_healthchecktable.create = AsyncMock(
            side_effect=Exception("Database connection failed")
        )
        
        # Should not raise exception, just return None
        result = await mock_prisma_client.save_health_check_result(
            model_name="test-model",
            status="healthy"
        )
        
        assert result is None



    @pytest.mark.asyncio
    async def test_get_health_check_history_success(self, mock_prisma_client):
        """Test retrieving health check history"""
        mock_results = [
            {"id": "1", "model_name": "gpt-3.5-turbo", "status": "healthy"},
            {"id": "2", "model_name": "gpt-3.5-turbo", "status": "unhealthy"}
        ]
        mock_prisma_client.db.litellm_healthchecktable.find_many = AsyncMock(return_value=mock_results)
        
        result = await mock_prisma_client.get_health_check_history(
            model_name="gpt-3.5-turbo",
            limit=50,
            offset=10
        )
        
        mock_prisma_client.db.litellm_healthchecktable.find_many.assert_called_once()
        assert result == mock_results


class TestHealthCheckEndpoints:
    """Test the health check endpoint functions"""

    @pytest.mark.asyncio
    async def test_save_health_check_to_db_success(self):
        """Test the _save_health_check_to_db function"""
        mock_prisma_client = MagicMock()
        mock_prisma_client.save_health_check_result = AsyncMock(return_value={"id": "test-id"})
        
        healthy_endpoints = [{"model": "gpt-3.5-turbo", "api_base": "https://api.openai.com"}]
        unhealthy_endpoints = []
        start_time = 1234567890.0
        
        await _save_health_check_to_db(
            prisma_client=mock_prisma_client,
            model_name="gpt-3.5-turbo",
            healthy_endpoints=healthy_endpoints,
            unhealthy_endpoints=unhealthy_endpoints,
            start_time=start_time,
            user_id="test-user"
        )
        
        # Verify save_health_check_result was called
        mock_prisma_client.save_health_check_result.assert_called_once()
        call_args = mock_prisma_client.save_health_check_result.call_args[1]
        
        assert call_args["model_name"] == "gpt-3.5-turbo"
        assert call_args["status"] == "healthy"
        assert call_args["healthy_count"] == 1
        assert call_args["unhealthy_count"] == 0
        assert call_args["checked_by"] == "test-user"

    @pytest.mark.asyncio
    async def test_save_health_check_to_db_with_failures(self):
        """Test _save_health_check_to_db with failed endpoints"""
        mock_prisma_client = MagicMock()
        mock_prisma_client.save_health_check_result = AsyncMock(return_value={"id": "test-id"})
        
        healthy_endpoints = []
        unhealthy_endpoints = [
            {
                "model": "gpt-4",
                "api_base": "https://api.openai.com",
                "error": "Connection timeout"
            }
        ]
        start_time = 1234567890.0
        
        await _save_health_check_to_db(
            prisma_client=mock_prisma_client,
            model_name="gpt-4",
            healthy_endpoints=healthy_endpoints,
            unhealthy_endpoints=unhealthy_endpoints,
            start_time=start_time,
            user_id="test-user"
        )
        
        call_args = mock_prisma_client.save_health_check_result.call_args[1]
        
        assert call_args["model_name"] == "gpt-4"
        assert call_args["status"] == "unhealthy"
        assert call_args["healthy_count"] == 0
        assert call_args["unhealthy_count"] == 1
        assert "Connection timeout" in call_args["error_message"]

    @pytest.mark.asyncio
    async def test_save_health_check_to_db_no_prisma_client(self):
        """Test _save_health_check_to_db when prisma_client is None"""
        # Should handle gracefully when no database client
        result = await _save_health_check_to_db(
            prisma_client=None,
            model_name="test-model",
            healthy_endpoints=[],
            unhealthy_endpoints=[],
            start_time=1234567890.0,
            user_id="test-user"
        )
        
        # Should not raise exception
        assert result is None


if __name__ == "__main__":
    pytest.main([__file__]) 