from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from litellm.proxy import proxy_server
from litellm.proxy.analytics_endpoints.analytics_endpoints import router
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth


@pytest.fixture
def client() -> TestClient:
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[user_api_key_auth] = lambda: MagicMock()
    return TestClient(app, raise_server_exceptions=False)


def _set_prisma(monkeypatch: pytest.MonkeyPatch, query_raw: AsyncMock) -> None:
    pc = MagicMock()
    pc.db.query_raw = query_raw
    monkeypatch.setattr(proxy_server, "prisma_client", pc)


def test_prompt_cache_activity_returns_provider_cache_tokens(client, monkeypatch):
    query_raw = AsyncMock(
        return_value=[
            {
                "api_key": "prod-key",
                "model": "anthropic/claude-haiku-4-5",
                "prompt_tokens": 8865,
                "cache_read_input_tokens": 4402,
                "cache_creation_input_tokens": 4402,
                "api_requests": 5,
            }
        ]
    )
    _set_prisma(monkeypatch, query_raw)

    response = client.get(
        "/global/activity/cache_hits/prompt_caching",
        params={"start_date": "2026-07-10", "end_date": "2026-07-18"},
    )

    assert response.status_code == 200
    assert response.json() == [
        {
            "api_key": "prod-key",
            "model": "anthropic/claude-haiku-4-5",
            "prompt_tokens": 8865,
            "cache_read_input_tokens": 4402,
            "cache_creation_input_tokens": 4402,
            "api_requests": 5,
        }
    ]


def test_prompt_cache_activity_reads_daily_spend_cache_columns(client, monkeypatch):
    query_raw = AsyncMock(return_value=[])
    _set_prisma(monkeypatch, query_raw)

    client.get(
        "/global/activity/cache_hits/prompt_caching",
        params={"start_date": "2026-07-10", "end_date": "2026-07-18"},
    )

    sql = query_raw.await_args.args[0]
    assert "LiteLLM_DailyUserSpend" in sql
    assert "cache_read_input_tokens" in sql
    assert "cache_creation_input_tokens" in sql
    assert query_raw.await_args.args[1:] == ("2026-07-10", "2026-07-18")


def test_prompt_cache_activity_empty_response(client, monkeypatch):
    _set_prisma(monkeypatch, AsyncMock(return_value=None))

    response = client.get(
        "/global/activity/cache_hits/prompt_caching",
        params={"start_date": "2026-07-10", "end_date": "2026-07-18"},
    )

    assert response.status_code == 200
    assert response.json() == []


def test_prompt_cache_activity_requires_dates(client, monkeypatch):
    _set_prisma(monkeypatch, AsyncMock(return_value=[]))

    response = client.get("/global/activity/cache_hits/prompt_caching")

    assert response.status_code == 400


def test_prompt_cache_activity_rejects_bad_date_format(client, monkeypatch):
    query_raw = AsyncMock(return_value=[])
    _set_prisma(monkeypatch, query_raw)

    response = client.get(
        "/global/activity/cache_hits/prompt_caching",
        params={"start_date": "07/10/2026", "end_date": "2026-07-18"},
    )

    assert response.status_code == 400
    assert "start_date must be in YYYY-MM-DD format" in response.text
    query_raw.assert_not_awaited()


def test_prompt_cache_activity_no_prisma(client, monkeypatch):
    monkeypatch.setattr(proxy_server, "prisma_client", None)

    response = client.get(
        "/global/activity/cache_hits/prompt_caching",
        params={"start_date": "2026-07-10", "end_date": "2026-07-18"},
    )

    assert response.status_code == 400
    assert "Database not connected" in response.text
