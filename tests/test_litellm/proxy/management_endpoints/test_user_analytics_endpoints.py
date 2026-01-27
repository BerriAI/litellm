
# tests/test_litellm/proxy/management_endpoints/test_user_analytics_endpoints.py

import os
import sys
import pytest
from unittest.mock import AsyncMock, MagicMock
from fastapi.testclient import TestClient

import litellm.proxy.proxy_server as ps
from litellm.proxy.proxy_server import app
from litellm.proxy._types import UserAPIKeyAuth, LitellmUserRoles, CommonProxyErrors

sys.path.insert(
    0, os.path.abspath("../../../")
)  # Adds the parent directory to the system path


@pytest.fixture
def client_and_mocks(monkeypatch):
    # Setup MagicMock Prisma
    mock_prisma = MagicMock()
    mock_db = MagicMock()

    # Mock query_raw for SQL queries
    mock_db.query_raw = AsyncMock(return_value=[])

    mock_prisma.db = mock_db

    # Monkeypatch Mocked Prisma client into the server module
    monkeypatch.setattr(ps, "prisma_client", mock_prisma)

    # Override returned auth user
    fake_user = UserAPIKeyAuth(
        user_id="test_user",
        user_role=LitellmUserRoles.INTERNAL_USER,
    )
    app.dependency_overrides[ps.user_api_key_auth] = lambda: fake_user

    client = TestClient(app)

    yield client, mock_prisma, mock_db

    # teardown
    app.dependency_overrides.clear()
    monkeypatch.setattr(ps, "prisma_client", ps.prisma_client)


@pytest.mark.asyncio
async def test_user_dau_success(client_and_mocks):
    """Test /user/dau endpoint with successful response"""
    client, _, mock_db = client_and_mocks

    # Mock DB response
    mock_db.query_raw.return_value = [
        {"date": "2024-01-15", "active_users": 10},
        {"date": "2024-01-14", "active_users": 8},
    ]

    # Call /user/dau endpoint
    resp = client.get("/user/dau?start_date=2024-01-14&end_date=2024-01-15")
    assert resp.status_code == 200, resp.text

    body = resp.json()
    assert "results" in body
    assert len(body["results"]) == 2
    assert body["results"][0]["date"] == "2024-01-15"
    assert body["results"][0]["active_users"] == 10
    assert body["results"][1]["date"] == "2024-01-14"
    assert body["results"][1]["active_users"] == 8

    # Verify query_raw was called with correct parameters
    mock_db.query_raw.assert_awaited_once()


@pytest.mark.asyncio
async def test_user_dau_default_dates(client_and_mocks):
    """Test /user/dau endpoint with default date parameters"""
    client, _, mock_db = client_and_mocks

    mock_db.query_raw.return_value = [
        {"date": "2024-01-15", "active_users": 5},
    ]

    # Call /user/dau endpoint without dates (should use defaults)
    resp = client.get("/user/dau")
    assert resp.status_code == 200, resp.text

    body = resp.json()
    assert "results" in body
    assert len(body["results"]) == 1


@pytest.mark.asyncio
async def test_user_dau_with_custom_llm_provider(client_and_mocks):
    """Test /user/dau endpoint with custom_llm_provider filter"""
    client, _, mock_db = client_and_mocks

    mock_db.query_raw.return_value = [
        {"date": "2024-01-15", "active_users": 3},
    ]

    # Call /user/dau endpoint with custom_llm_provider filter
    resp = client.get("/user/dau?custom_llm_provider=hosted_vllm")
    assert resp.status_code == 200, resp.text

    body = resp.json()
    assert "results" in body

    # Verify query_raw was called
    mock_db.query_raw.assert_awaited_once()


@pytest.mark.asyncio
async def test_user_dau_invalid_date_format(client_and_mocks):
    """Test /user/dau endpoint with invalid date format"""
    client, _, _ = client_and_mocks

    # Call /user/dau endpoint with invalid date format
    resp = client.get("/user/dau?start_date=invalid-date")
    assert resp.status_code == 400, resp.text

    detail = resp.json()["detail"]
    assert "Invalid date format" in detail


@pytest.mark.asyncio
async def test_user_dau_db_not_connected(client_and_mocks, monkeypatch):
    """Test /user/dau endpoint when database is not connected"""
    client, _, _ = client_and_mocks

    # Override prisma_client to None
    monkeypatch.setattr(ps, "prisma_client", None)

    resp = client.get("/user/dau")
    assert resp.status_code == 500, resp.text

    detail = resp.json()["detail"]
    assert detail["error"] == CommonProxyErrors.db_not_connected_error.value


@pytest.mark.asyncio
async def test_user_wau_success(client_and_mocks):
    """Test /user/wau endpoint with successful response"""
    client, _, mock_db = client_and_mocks

    # Mock DB response for 7 weeks
    mock_db.query_raw.return_value = [
        {"date": "Week 7 (Jan 15)", "active_users": 15, "period_start": "2024-01-09", "period_end": "2024-01-15"},
        {"date": "Week 6 (Jan 08)", "active_users": 12, "period_start": "2024-01-02", "period_end": "2024-01-08"},
        {"date": "Week 5 (Jan 01)", "active_users": 10, "period_start": "2023-12-26", "period_end": "2024-01-01"},
    ]

    # Call /user/wau endpoint
    resp = client.get("/user/wau")
    assert resp.status_code == 200, resp.text

    body = resp.json()
    assert "results" in body
    assert len(body["results"]) == 3
    assert body["results"][0]["date"] == "Week 7 (Jan 15)"
    assert body["results"][0]["active_users"] == 15
    assert "period_start" in body["results"][0]
    assert "period_end" in body["results"][0]

    # Verify query_raw was called
    mock_db.query_raw.assert_awaited_once()


@pytest.mark.asyncio
async def test_user_wau_with_custom_llm_provider(client_and_mocks):
    """Test /user/wau endpoint with custom_llm_provider filter"""
    client, _, mock_db = client_and_mocks

    mock_db.query_raw.return_value = [
        {"date": "Week 7 (Jan 15)", "active_users": 8, "period_start": "2024-01-09", "period_end": "2024-01-15"},
    ]

    # Call /user/wau endpoint with custom_llm_provider filter
    resp = client.get("/user/wau?custom_llm_provider=hosted_vllm")
    assert resp.status_code == 200, resp.text

    body = resp.json()
    assert "results" in body


@pytest.mark.asyncio
async def test_user_wau_db_not_connected(client_and_mocks, monkeypatch):
    """Test /user/wau endpoint when database is not connected"""
    client, _, _ = client_and_mocks

    # Override prisma_client to None
    monkeypatch.setattr(ps, "prisma_client", None)

    resp = client.get("/user/wau")
    assert resp.status_code == 500, resp.text

    detail = resp.json()["detail"]
    assert detail["error"] == CommonProxyErrors.db_not_connected_error.value


@pytest.mark.asyncio
async def test_user_mau_success(client_and_mocks):
    """Test /user/mau endpoint with successful response"""
    client, _, mock_db = client_and_mocks

    # Mock DB response for 7 months
    mock_db.query_raw.return_value = [
        {"date": "Jan 2024", "active_users": 25, "period_start": "2023-12-27", "period_end": "2024-01-26"},
        {"date": "Dec 2023", "active_users": 20, "period_start": "2023-11-27", "period_end": "2023-12-26"},
        {"date": "Nov 2023", "active_users": 18, "period_start": "2023-10-28", "period_end": "2023-11-27"},
    ]

    # Call /user/mau endpoint
    resp = client.get("/user/mau?months=7")
    assert resp.status_code == 200, resp.text

    body = resp.json()
    assert "results" in body
    assert len(body["results"]) == 3
    assert body["results"][0]["date"] == "Jan 2024"
    assert body["results"][0]["active_users"] == 25
    assert "period_start" in body["results"][0]
    assert "period_end" in body["results"][0]

    # Verify query_raw was called
    mock_db.query_raw.assert_awaited_once()


@pytest.mark.asyncio
async def test_user_mau_default_months(client_and_mocks):
    """Test /user/mau endpoint with default months parameter"""
    client, _, mock_db = client_and_mocks

    mock_db.query_raw.return_value = [
        {"date": "Jan 2024", "active_users": 30, "period_start": "2023-12-27", "period_end": "2024-01-26"},
    ]

    # Call /user/mau endpoint without months (should default to 7)
    resp = client.get("/user/mau")
    assert resp.status_code == 200, resp.text

    body = resp.json()
    assert "results" in body


@pytest.mark.asyncio
async def test_user_mau_with_custom_llm_provider(client_and_mocks):
    """Test /user/mau endpoint with custom_llm_provider filter"""
    client, _, mock_db = client_and_mocks

    mock_db.query_raw.return_value = [
        {"date": "Jan 2024", "active_users": 12, "period_start": "2023-12-27", "period_end": "2024-01-26"},
    ]

    # Call /user/mau endpoint with custom_llm_provider filter
    resp = client.get("/user/mau?custom_llm_provider=hosted_vllm")
    assert resp.status_code == 200, resp.text

    body = resp.json()
    assert "results" in body


@pytest.mark.asyncio
async def test_user_mau_invalid_months(client_and_mocks):
    """Test /user/mau endpoint with invalid months parameter (out of range)"""
    client, _, _ = client_and_mocks

    # Call /user/mau endpoint with months > 12 (should fail validation)
    resp = client.get("/user/mau?months=15")
    assert resp.status_code == 422, resp.text  # Validation error


@pytest.mark.asyncio
async def test_user_mau_db_not_connected(client_and_mocks, monkeypatch):
    """Test /user/mau endpoint when database is not connected"""
    client, _, _ = client_and_mocks

    # Override prisma_client to None
    monkeypatch.setattr(ps, "prisma_client", None)

    resp = client.get("/user/mau")
    assert resp.status_code == 500, resp.text

    detail = resp.json()["detail"]
    assert detail["error"] == CommonProxyErrors.db_not_connected_error.value


@pytest.mark.asyncio
async def test_user_dau_empty_results(client_and_mocks):
    """Test /user/dau endpoint with empty results"""
    client, _, mock_db = client_and_mocks

    # Mock empty DB response
    mock_db.query_raw.return_value = []

    resp = client.get("/user/dau")
    assert resp.status_code == 200, resp.text

    body = resp.json()
    assert "results" in body
    assert len(body["results"]) == 0


@pytest.mark.asyncio
async def test_user_wau_empty_results(client_and_mocks):
    """Test /user/wau endpoint with empty results"""
    client, _, mock_db = client_and_mocks

    # Mock empty DB response
    mock_db.query_raw.return_value = []

    resp = client.get("/user/wau")
    assert resp.status_code == 200, resp.text

    body = resp.json()
    assert "results" in body
    assert len(body["results"]) == 0


@pytest.mark.asyncio
async def test_user_mau_empty_results(client_and_mocks):
    """Test /user/mau endpoint with empty results"""
    client, _, mock_db = client_and_mocks

    # Mock empty DB response
    mock_db.query_raw.return_value = []

    resp = client.get("/user/mau")
    assert resp.status_code == 200, resp.text

    body = resp.json()
    assert "results" in body
    assert len(body["results"]) == 0