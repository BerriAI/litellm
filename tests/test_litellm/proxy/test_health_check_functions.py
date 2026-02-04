import asyncio
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, os.path.abspath("../../.."))

from litellm.proxy.health_endpoints._health_endpoints import (
    _aggregate_health_check_results,
    _build_model_param_to_info_mapping,
    _save_background_health_checks_to_db,
    _save_health_check_results_if_changed,
    _save_health_check_to_db,
)
from litellm.proxy.utils import PrismaClient


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


# Tests for background health check functions

def test_build_model_param_to_info_mapping():
    """Test building model parameter to info mapping"""
    model_list = [
        {
            "model_name": "gpt-3.5-turbo",
            "model_info": {"id": "model-123"},
            "litellm_params": {"model": "gpt-3.5-turbo"},
        },
        {
            "model_name": "gpt-4",
            "model_info": {"id": "model-456"},
            "litellm_params": {"model": "gpt-4"},
        },
        {
            "model_name": "gpt-3.5-turbo-alias",
            "model_info": {"id": "model-789"},
            "litellm_params": {"model": "gpt-3.5-turbo"},  # Same model param
        },
    ]
    
    result = _build_model_param_to_info_mapping(model_list)
    
    assert "gpt-3.5-turbo" in result
    assert "gpt-4" in result
    assert len(result["gpt-3.5-turbo"]) == 2  # Two models share same param
    assert len(result["gpt-4"]) == 1
    assert result["gpt-3.5-turbo"][0]["model_name"] == "gpt-3.5-turbo"
    assert result["gpt-3.5-turbo"][0]["model_id"] == "model-123"
    assert result["gpt-3.5-turbo"][1]["model_name"] == "gpt-3.5-turbo-alias"
    assert result["gpt-3.5-turbo"][1]["model_id"] == "model-789"


def test_build_model_param_to_info_mapping_no_model_name():
    """Test mapping skips models without model_name"""
    model_list = [
        {
            "model_info": {"id": "model-123"},
            "litellm_params": {"model": "gpt-3.5-turbo"},
        },
    ]
    
    result = _build_model_param_to_info_mapping(model_list)
    assert len(result) == 0


def test_aggregate_health_check_results():
    """Test aggregating health check results per model"""
    model_param_to_info = {
        "gpt-3.5-turbo": [
            {"model_name": "gpt-3.5-turbo", "model_id": "model-123"},
        ],
        "gpt-4": [
            {"model_name": "gpt-4", "model_id": "model-456"},
        ],
    }
    
    healthy_endpoints = [
        {"model": "gpt-3.5-turbo"},
    ]
    unhealthy_endpoints = [
        {"model": "gpt-4", "error": "Rate limit exceeded"},
    ]
    
    result = _aggregate_health_check_results(
        model_param_to_info, healthy_endpoints, unhealthy_endpoints
    )
    
    # Check gpt-3.5-turbo is healthy
    gpt35_key = ("model-123", "gpt-3.5-turbo")
    assert gpt35_key in result
    assert result[gpt35_key]["healthy_count"] == 1
    assert result[gpt35_key]["unhealthy_count"] == 0
    assert result[gpt35_key]["error_message"] is None
    
    # Check gpt-4 is unhealthy
    gpt4_key = ("model-456", "gpt-4")
    assert gpt4_key in result
    assert result[gpt4_key]["healthy_count"] == 0
    assert result[gpt4_key]["unhealthy_count"] == 1
    assert "Rate limit" in result[gpt4_key]["error_message"]


def test_aggregate_health_check_results_multiple_endpoints():
    """Test aggregation with multiple endpoints for same model"""
    model_param_to_info = {
        "gpt-3.5-turbo": [
            {"model_name": "gpt-3.5-turbo", "model_id": "model-123"},
        ],
    }
    
    healthy_endpoints = [
        {"model": "gpt-3.5-turbo"},
        {"model": "gpt-3.5-turbo"},
    ]
    unhealthy_endpoints = []
    
    result = _aggregate_health_check_results(
        model_param_to_info, healthy_endpoints, unhealthy_endpoints
    )
    
    key = ("model-123", "gpt-3.5-turbo")
    assert result[key]["healthy_count"] == 2
    assert result[key]["unhealthy_count"] == 0


@pytest.mark.asyncio
async def test_save_health_check_results_if_changed_status_changed():
    """Test saving when status changes"""
    mock_prisma = MagicMock()
    mock_prisma.save_health_check_result = AsyncMock()
    
    model_results = {
        ("model-123", "gpt-3.5-turbo"): {
            "model_name": "gpt-3.5-turbo",
            "model_id": "model-123",
            "healthy_count": 1,
            "unhealthy_count": 0,
            "error_message": None,
        },
    }
    
    # Latest check shows unhealthy, new result is healthy (status changed)
    latest_checks_map = {
        "model-123": MagicMock(
            status="unhealthy",
            checked_at=datetime.now(timezone.utc) - timedelta(minutes=5),
        ),
    }
    
    start_time = 1234567890.0
    await _save_health_check_results_if_changed(
        mock_prisma, model_results, latest_checks_map, start_time, "background_health_check"
    )
    
    # Should save because status changed
    mock_prisma.save_health_check_result.assert_called_once()
    call_kwargs = mock_prisma.save_health_check_result.call_args[1]
    assert call_kwargs["status"] == "healthy"
    assert call_kwargs["model_name"] == "gpt-3.5-turbo"
    assert call_kwargs["checked_by"] == "background_health_check"


@pytest.mark.asyncio
async def test_save_health_check_results_if_changed_status_unchanged_recent():
    """Test skipping save when status unchanged and checked recently"""
    mock_prisma = MagicMock()
    mock_prisma.save_health_check_result = AsyncMock()
    
    model_results = {
        ("model-123", "gpt-3.5-turbo"): {
            "model_name": "gpt-3.5-turbo",
            "model_id": "model-123",
            "healthy_count": 1,
            "unhealthy_count": 0,
            "error_message": None,
        },
    }
    
    # Latest check shows healthy, new result is healthy (status unchanged)
    # And checked recently (within 1 hour)
    latest_checks_map = {
        "model-123": MagicMock(
            status="healthy",
            checked_at=datetime.now(timezone.utc) - timedelta(minutes=30),
        ),
    }
    
    start_time = 1234567890.0
    await _save_health_check_results_if_changed(
        mock_prisma, model_results, latest_checks_map, start_time, "background_health_check"
    )
    
    # Should NOT save because status unchanged and checked recently
    mock_prisma.save_health_check_result.assert_not_called()


@pytest.mark.asyncio
async def test_save_health_check_results_if_changed_status_unchanged_old():
    """Test saving when status unchanged but last check is old (>1 hour)"""
    mock_prisma = MagicMock()
    mock_prisma.save_health_check_result = AsyncMock()
    
    model_results = {
        ("model-123", "gpt-3.5-turbo"): {
            "model_name": "gpt-3.5-turbo",
            "model_id": "model-123",
            "healthy_count": 1,
            "unhealthy_count": 0,
            "error_message": None,
        },
    }
    
    # Latest check shows healthy, new result is healthy (status unchanged)
    # But checked >1 hour ago
    latest_checks_map = {
        "model-123": MagicMock(
            status="healthy",
            checked_at=datetime.now(timezone.utc) - timedelta(hours=2),
        ),
    }
    
    start_time = 1234567890.0
    await _save_health_check_results_if_changed(
        mock_prisma, model_results, latest_checks_map, start_time, "background_health_check"
    )
    
    # Should save because last check is old (>1 hour)
    mock_prisma.save_health_check_result.assert_called_once()


@pytest.mark.asyncio
async def test_save_health_check_results_if_changed_no_previous_check():
    """Test saving when there's no previous check"""
    mock_prisma = MagicMock()
    mock_prisma.save_health_check_result = AsyncMock()
    
    model_results = {
        ("model-123", "gpt-3.5-turbo"): {
            "model_name": "gpt-3.5-turbo",
            "model_id": "model-123",
            "healthy_count": 1,
            "unhealthy_count": 0,
            "error_message": None,
        },
    }
    
    # No previous check
    latest_checks_map = {}
    
    start_time = 1234567890.0
    await _save_health_check_results_if_changed(
        mock_prisma, model_results, latest_checks_map, start_time, "background_health_check"
    )
    
    # Should save because no previous check
    mock_prisma.save_health_check_result.assert_called_once()


@pytest.mark.asyncio
async def test_save_background_health_checks_to_db():
    """Test the main background health check save function"""
    mock_prisma = MagicMock()
    mock_prisma.save_health_check_result = AsyncMock()
    mock_prisma.get_all_latest_health_checks = AsyncMock(return_value=[])
    
    model_list = [
        {
            "model_name": "gpt-3.5-turbo",
            "model_info": {"id": "model-123"},
            "litellm_params": {"model": "gpt-3.5-turbo"},
        },
    ]
    
    healthy_endpoints = [{"model": "gpt-3.5-turbo"}]
    unhealthy_endpoints = []
    
    start_time = 1234567890.0
    
    await _save_background_health_checks_to_db(
        mock_prisma, model_list, healthy_endpoints, unhealthy_endpoints, start_time, "background_health_check"
    )
    
    # Should call get_all_latest_health_checks and save_health_check_result
    mock_prisma.get_all_latest_health_checks.assert_called_once()
    mock_prisma.save_health_check_result.assert_called_once()
    
    call_kwargs = mock_prisma.save_health_check_result.call_args[1]
    assert call_kwargs["model_name"] == "gpt-3.5-turbo"
    assert call_kwargs["model_id"] == "model-123"
    assert call_kwargs["status"] == "healthy"
    assert call_kwargs["checked_by"] == "background_health_check"


@pytest.mark.asyncio
async def test_save_background_health_checks_to_db_no_prisma():
    """Test graceful handling when no prisma client"""
    result = await _save_background_health_checks_to_db(
        None, [], [], [], 0.0, "background_health_check"
    )
    assert result is None


@pytest.mark.asyncio
async def test_save_background_health_checks_to_db_exception_handling():
    """Test exception handling in background health check save"""
    mock_prisma = MagicMock()
    mock_prisma.get_all_latest_health_checks = AsyncMock(side_effect=Exception("DB Error"))
    
    model_list = [
        {
            "model_name": "gpt-3.5-turbo",
            "model_info": {"id": "model-123"},
            "litellm_params": {"model": "gpt-3.5-turbo"},
        },
    ]
    
    # Should not raise exception, should handle gracefully
    await _save_background_health_checks_to_db(
        mock_prisma, model_list, [], [], 0.0, "background_health_check"
    )
    
    # Function should complete without raising


@pytest.mark.asyncio
async def test_get_all_latest_health_checks_with_model_id(mock_prisma):
    """Test get_all_latest_health_checks properly groups by model_id"""
    # Create mock checks with same model_name but different model_id
    mock_check1 = MagicMock()
    mock_check1.model_id = "model-123"
    mock_check1.model_name = "gpt-3.5-turbo"
    mock_check1.checked_at = datetime.now(timezone.utc) - timedelta(minutes=10)
    
    mock_check2 = MagicMock()
    mock_check2.model_id = "model-456"
    mock_check2.model_name = "gpt-3.5-turbo"
    mock_check2.checked_at = datetime.now(timezone.utc) - timedelta(minutes=5)
    
    mock_check3 = MagicMock()
    mock_check3.model_id = "model-123"
    mock_check3.model_name = "gpt-3.5-turbo"
    mock_check3.checked_at = datetime.now(timezone.utc) - timedelta(minutes=1)  # Latest for model-123
    
    # Order by checked_at desc
    mock_prisma.db.litellm_healthchecktable.find_many = AsyncMock(
        return_value=[mock_check3, mock_check2, mock_check1]
    )
    
    result = await mock_prisma.get_all_latest_health_checks()
    
    # Should return 2 unique models (by model_id)
    assert len(result) == 2
    
    # Should have latest check for each model_id
    model_ids = {check.model_id for check in result}
    assert "model-123" in model_ids
    assert "model-456" in model_ids
    
    # model-123 should have the latest check (1 minute ago)
    model123_check = next(c for c in result if c.model_id == "model-123")
    assert model123_check.checked_at == mock_check3.checked_at


@pytest.mark.asyncio
async def test_get_all_latest_health_checks_without_model_id(mock_prisma):
    """Test get_all_latest_health_checks groups by model_name when model_id is None"""
    mock_check1 = MagicMock()
    mock_check1.model_id = None
    mock_check1.model_name = "gpt-3.5-turbo"
    mock_check1.checked_at = datetime.now(timezone.utc) - timedelta(minutes=10)
    
    mock_check2 = MagicMock()
    mock_check2.model_id = None
    mock_check2.model_name = "gpt-3.5-turbo"
    mock_check2.checked_at = datetime.now(timezone.utc) - timedelta(minutes=1)  # Latest
    
    mock_prisma.db.litellm_healthchecktable.find_many = AsyncMock(
        return_value=[mock_check2, mock_check1]
    )
    
    result = await mock_prisma.get_all_latest_health_checks()
    
    # Should return 1 unique model (by model_name)
    assert len(result) == 1
    assert result[0].model_name == "gpt-3.5-turbo"
    assert result[0].checked_at == mock_check2.checked_at  # Latest


if __name__ == "__main__":
    pytest.main([__file__]) 