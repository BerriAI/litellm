"""
Tests for Admin User Usage Analytics Endpoint
"""

import pytest
import sys
import os
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

# Add parent directory to path
sys.path.insert(0, os.path.abspath("../.."))

from litellm.proxy.management_endpoints.admin_user_usage_endpoints import (
    get_admin_users_usage,
)


@pytest.mark.asyncio
async def test_get_admin_users_usage_basic():
    """Test basic admin user usage query with mock data"""

    # Mock prisma client
    mock_prisma = MagicMock()
    mock_db = MagicMock()
    mock_prisma.db = mock_db

    # Mock summary query result
    mock_db.query_raw = AsyncMock()

    # First call: summary query
    mock_db.query_raw.side_effect = [
        # Summary result
        [
            {
                "total_users": 10,
                "total_spend": 1000.0,
                "total_requests": 5000,
                "total_successful_requests": 4900,
                "total_failed_requests": 100,
                "total_tokens": 100000,
                "avg_spend_per_user": 100.0,
                "power_users_count": 2,
                "low_users_count": 3,
            }
        ],
        # Top users result
        [
            {
                "user_id": "user_1",
                "user_email": "user1@example.com",
                "total_spend": 500.0,
                "total_requests": 2000,
                "total_successful_requests": 1980,
                "total_failed_requests": 20,
                "total_prompt_tokens": 40000,
                "total_completion_tokens": 10000,
                "total_tokens": 50000,
                "days_active": 15,
                "first_request_date": "2026-01-01",
                "last_request_date": "2026-01-15",
                "tags": ["User-Agent:claude-code"],
                "models_used": ["claude-sonnet-4"],
            }
        ],
        # Count query for pagination
        [{"total_count": 10}],
        # Paginated users result
        [
            {
                "user_id": "user_1",
                "user_email": "user1@example.com",
                "total_spend": 500.0,
                "total_requests": 2000,
                "total_successful_requests": 1980,
                "total_failed_requests": 20,
                "total_prompt_tokens": 40000,
                "total_completion_tokens": 10000,
                "total_tokens": 50000,
                "days_active": 15,
                "first_request_date": "2026-01-01",
                "last_request_date": "2026-01-15",
                "tags": ["User-Agent:claude-code"],
                "models_used": ["claude-sonnet-4"],
            },
            {
                "user_id": "user_2",
                "user_email": "user2@example.com",
                "total_spend": 300.0,
                "total_requests": 1500,
                "total_successful_requests": 1490,
                "total_failed_requests": 10,
                "total_prompt_tokens": 30000,
                "total_completion_tokens": 7500,
                "total_tokens": 37500,
                "days_active": 12,
                "first_request_date": "2026-01-03",
                "last_request_date": "2026-01-15",
                "tags": ["User-Agent:claude-code"],
                "models_used": ["claude-opus-4"],
            },
        ],
    ]

    # Call the function
    result = await get_admin_users_usage(
        prisma_client=mock_prisma,
        start_date="2026-01-01",
        end_date="2026-01-15",
        tag_filters=["User-Agent:claude-code"],
        min_spend=None,
        max_spend=None,
        sort_by="spend",
        sort_order="desc",
        page=1,
        page_size=50,
        top_n=10,
    )

    # Assertions
    assert result is not None
    assert "summary" in result
    assert "top_users" in result
    assert "users" in result
    assert "pagination" in result

    # Check summary
    assert result["summary"]["total_users"] == 10
    assert result["summary"]["total_spend"] == 1000.0
    assert result["summary"]["power_users_count"] == 2

    # Check top users
    assert len(result["top_users"]) == 1
    assert result["top_users"][0]["user_email"] == "user1@example.com"
    assert result["top_users"][0]["spend"] == 500.0

    # Check paginated users
    assert len(result["users"]) == 2
    assert result["users"][0]["user_email"] == "user1@example.com"
    assert result["users"][1]["user_email"] == "user2@example.com"

    # Check pagination
    assert result["pagination"]["page"] == 1
    assert result["pagination"]["total_count"] == 10
    assert result["pagination"]["total_pages"] == 1


@pytest.mark.asyncio
async def test_get_admin_users_usage_with_filters():
    """Test admin user usage with spend filters"""

    # Mock prisma client
    mock_prisma = MagicMock()
    mock_db = MagicMock()
    mock_prisma.db = mock_db
    mock_db.query_raw = AsyncMock()

    # Mock results for filtered query
    mock_db.query_raw.side_effect = [
        # Summary result
        [
            {
                "total_users": 2,
                "total_spend": 800.0,
                "total_requests": 3500,
                "total_successful_requests": 3470,
                "total_failed_requests": 30,
                "total_tokens": 87500,
                "avg_spend_per_user": 400.0,
                "power_users_count": 2,
                "low_users_count": 0,
            }
        ],
        # Top users result
        [
            {
                "user_id": "user_1",
                "user_email": "user1@example.com",
                "total_spend": 500.0,
                "total_requests": 2000,
                "total_successful_requests": 1980,
                "total_failed_requests": 20,
                "total_prompt_tokens": 40000,
                "total_completion_tokens": 10000,
                "total_tokens": 50000,
                "days_active": 15,
                "first_request_date": "2026-01-01",
                "last_request_date": "2026-01-15",
                "tags": ["User-Agent:claude-code"],
                "models_used": ["claude-sonnet-4"],
            }
        ],
        # Count query
        [{"total_count": 2}],
        # Paginated users result
        [
            {
                "user_id": "user_1",
                "user_email": "user1@example.com",
                "total_spend": 500.0,
                "total_requests": 2000,
                "total_successful_requests": 1980,
                "total_failed_requests": 20,
                "total_prompt_tokens": 40000,
                "total_completion_tokens": 10000,
                "total_tokens": 50000,
                "days_active": 15,
                "first_request_date": "2026-01-01",
                "last_request_date": "2026-01-15",
                "tags": ["User-Agent:claude-code"],
                "models_used": ["claude-sonnet-4"],
            },
            {
                "user_id": "user_2",
                "user_email": "user2@example.com",
                "total_spend": 300.0,
                "total_requests": 1500,
                "total_successful_requests": 1490,
                "total_failed_requests": 10,
                "total_prompt_tokens": 30000,
                "total_completion_tokens": 7500,
                "total_tokens": 37500,
                "days_active": 12,
                "first_request_date": "2026-01-03",
                "last_request_date": "2026-01-15",
                "tags": ["User-Agent:claude-code"],
                "models_used": ["claude-opus-4"],
            },
        ],
    ]

    # Call with min_spend filter
    result = await get_admin_users_usage(
        prisma_client=mock_prisma,
        start_date="2026-01-01",
        end_date="2026-01-15",
        tag_filters=["User-Agent:claude-code"],
        min_spend=200.0,  # Filter for users spending > $200
        max_spend=None,
        sort_by="spend",
        sort_order="desc",
        page=1,
        page_size=50,
        top_n=10,
    )

    # Assertions
    assert result is not None
    assert result["summary"]["total_users"] == 2
    assert result["summary"]["power_users_count"] == 2
    assert len(result["users"]) == 2

    # All users should have spend >= 200
    for user in result["users"]:
        assert user["spend"] >= 200.0


@pytest.mark.asyncio
async def test_get_admin_users_usage_pagination():
    """Test pagination works correctly"""

    # Mock prisma client
    mock_prisma = MagicMock()
    mock_db = MagicMock()
    mock_prisma.db = mock_db
    mock_db.query_raw = AsyncMock()

    # Mock results for page 2
    mock_db.query_raw.side_effect = [
        # Summary result
        [
            {
                "total_users": 150,
                "total_spend": 15000.0,
                "total_requests": 75000,
                "total_successful_requests": 74000,
                "total_failed_requests": 1000,
                "total_tokens": 1500000,
                "avg_spend_per_user": 100.0,
                "power_users_count": 25,
                "low_users_count": 30,
            }
        ],
        # Top users result
        [],
        # Count query
        [{"total_count": 150}],
        # Paginated users result (page 2: users 51-100)
        [
            {
                "user_id": f"user_{i}",
                "user_email": f"user{i}@example.com",
                "total_spend": 100.0 - i,
                "total_requests": 500,
                "total_successful_requests": 490,
                "total_failed_requests": 10,
                "total_prompt_tokens": 10000,
                "total_completion_tokens": 5000,
                "total_tokens": 15000,
                "days_active": 10,
                "first_request_date": "2026-01-01",
                "last_request_date": "2026-01-15",
                "tags": [],
                "models_used": ["claude-sonnet-4"],
            }
            for i in range(51, 101)
        ],
    ]

    # Call with page 2
    result = await get_admin_users_usage(
        prisma_client=mock_prisma,
        start_date="2026-01-01",
        end_date="2026-01-15",
        tag_filters=None,
        min_spend=None,
        max_spend=None,
        sort_by="spend",
        sort_order="desc",
        page=2,
        page_size=50,
        top_n=10,
    )

    # Assertions
    assert result["pagination"]["page"] == 2
    assert result["pagination"]["page_size"] == 50
    assert result["pagination"]["total_count"] == 150
    assert result["pagination"]["total_pages"] == 3
    assert len(result["users"]) == 50


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
