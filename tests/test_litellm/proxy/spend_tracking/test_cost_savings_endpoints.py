import os
import sys
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath("../../../.."))

import litellm
import litellm.proxy.proxy_server as ps
from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.proxy_server import app
from litellm.proxy.spend_tracking.cost_savings_endpoints import (
    _DailySavingsRow,
    _SpendLogRow,
    build_activity_response,
    compute_savings_amounts,
    resolve_model_pricing,
    summarize_optimized_request,
)

COST_MAP = {
    "anthropic/claude-x": {
        "input_cost_per_token": 3e-06,
        "cache_read_input_token_cost": 3e-07,
        "cache_creation_input_token_cost": 3.75e-06,
    },
    "gpt-x": {
        "input_cost_per_token": 2e-06,
        "cache_read_input_token_cost": 1e-06,
    },
    "no-cache-price-model": {"input_cost_per_token": 5e-06},
    "free-model": {"input_cost_per_token": 0.0},
    "malformed-model": {"input_cost_per_token": "not-a-number"},
}


@pytest.fixture
def client():
    return TestClient(app)


class TestResolveModelPricing:
    def test_provider_qualified_key_wins(self):
        pricing = resolve_model_pricing("claude-x", "anthropic", COST_MAP)
        assert pricing is not None
        assert pricing.input_cost_per_token == 3e-06
        assert pricing.cache_read_cost_per_token == 3e-07
        assert pricing.cache_creation_cost_per_token == 3.75e-06

    def test_bare_key_fallback(self):
        pricing = resolve_model_pricing("gpt-x", "openai", COST_MAP)
        assert pricing is not None
        assert pricing.input_cost_per_token == 2e-06
        assert pricing.cache_creation_cost_per_token is None

    def test_provider_prefixed_stored_model_resolves_bare_key(self):
        pricing = resolve_model_pricing("openai/gpt-x", "openai", COST_MAP)
        assert pricing is not None
        assert pricing.input_cost_per_token == 2e-06

    def test_unknown_model_returns_none(self):
        assert resolve_model_pricing("nope", "openai", COST_MAP) is None

    def test_zero_input_price_returns_none(self):
        assert resolve_model_pricing("free-model", "", COST_MAP) is None

    def test_malformed_entry_returns_none(self):
        assert resolve_model_pricing("malformed-model", "", COST_MAP) is None


class TestComputeSavingsAmounts:
    def test_net_cache_savings_subtracts_write_premium(self):
        pricing = resolve_model_pricing("claude-x", "anthropic", COST_MAP)
        amounts = compute_savings_amounts(
            cache_read_tokens=1000, cache_creation_tokens=100, compression_saved_tokens=0, pricing=pricing
        )
        assert amounts.cache_savings == pytest.approx(1000 * (3e-06 - 3e-07) - 100 * (3.75e-06 - 3e-06))

    def test_compression_savings_use_input_price(self):
        pricing = resolve_model_pricing("gpt-x", "", COST_MAP)
        amounts = compute_savings_amounts(
            cache_read_tokens=0, cache_creation_tokens=0, compression_saved_tokens=500, pricing=pricing
        )
        assert amounts.compression_savings == pytest.approx(500 * 2e-06)

    def test_missing_cache_read_price_yields_zero_cache_savings(self):
        pricing = resolve_model_pricing("no-cache-price-model", "", COST_MAP)
        amounts = compute_savings_amounts(
            cache_read_tokens=1000, cache_creation_tokens=0, compression_saved_tokens=0, pricing=pricing
        )
        assert amounts.cache_savings == 0.0

    def test_none_pricing_yields_zero(self):
        amounts = compute_savings_amounts(
            cache_read_tokens=1000, cache_creation_tokens=10, compression_saved_tokens=500, pricing=None
        )
        assert amounts.cache_savings == 0.0
        assert amounts.compression_savings == 0.0


def _row(**overrides):
    defaults = {
        "date": "2026-07-15",
        "model": "claude-x",
        "custom_llm_provider": "anthropic",
        "cache_read_input_tokens": 0,
        "cache_creation_input_tokens": 0,
        "compression_saved_tokens": 0,
        "spend": 0.0,
    }
    return _DailySavingsRow(**{**defaults, **overrides})


class TestBuildActivityResponse:
    def test_days_grouped_and_totals_summed(self):
        rows = [
            _row(date="2026-07-15", cache_read_input_tokens=1000, spend=1.0),
            _row(date="2026-07-15", model="gpt-x", custom_llm_provider="openai", compression_saved_tokens=500, spend=2.0),
            _row(date="2026-07-16", cache_read_input_tokens=2000, cache_creation_input_tokens=100, spend=3.0),
        ]
        response = build_activity_response(rows, COST_MAP)
        assert [daily.date for daily in response.results] == ["2026-07-15", "2026-07-16"]
        day_one, day_two = response.results
        assert day_one.metrics.cache_savings == pytest.approx(1000 * (3e-06 - 3e-07))
        assert day_one.metrics.compression_savings == pytest.approx(500 * 2e-06)
        assert day_two.metrics.cache_savings == pytest.approx(2000 * (3e-06 - 3e-07) - 100 * 7.5e-07)
        assert response.totals.spend == pytest.approx(6.0)
        assert response.totals.total_savings == pytest.approx(
            day_one.metrics.total_savings + day_two.metrics.total_savings
        )
        assert response.totals.cache_read_input_tokens == 3000
        assert response.totals.compression_saved_tokens == 500
        assert response.unpriced_models == []

    def test_unpriced_models_reported(self):
        rows = [
            _row(model="mystery-model", custom_llm_provider="", cache_read_input_tokens=100),
            _row(model="no-cache-price-model", custom_llm_provider="", cache_read_input_tokens=100),
            _row(model="mystery-write-only", custom_llm_provider="", cache_creation_input_tokens=100),
            _row(model="mystery-idle", custom_llm_provider="", spend=1.0),
        ]
        response = build_activity_response(rows, COST_MAP)
        assert response.unpriced_models == ["mystery-model", "mystery-write-only", "no-cache-price-model"]
        assert response.totals.cache_savings == 0.0


def _spend_log_row(metadata, **overrides):
    defaults = {
        "request_id": "req_1",
        "startTime": datetime(2026, 7, 15, 12, 0, 0, tzinfo=timezone.utc),
        "model": "claude-x",
        "custom_llm_provider": "anthropic",
        "total_tokens": 1500,
        "spend": 0.01,
        "metadata": metadata,
    }
    return _SpendLogRow(**{**defaults, **overrides})


class TestSummarizeOptimizedRequest:
    def test_anthropic_style_cache_read(self):
        row = _spend_log_row({"usage_object": {"cache_read_input_tokens": 1000}})
        summary = summarize_optimized_request(row, COST_MAP)
        assert summary is not None
        assert summary.optimizations == ["caching"]
        assert summary.savings == pytest.approx(1000 * (3e-06 - 3e-07))
        assert summary.original_cost == pytest.approx(summary.optimized_cost + summary.savings)

    def test_openai_style_cached_tokens(self):
        row = _spend_log_row(
            {"usage_object": {"prompt_tokens_details": {"cached_tokens": 800}}},
            model="gpt-x",
            custom_llm_provider="openai",
        )
        summary = summarize_optimized_request(row, COST_MAP)
        assert summary is not None
        assert summary.optimizations == ["caching"]
        assert summary.savings == pytest.approx(800 * (2e-06 - 1e-06))

    def test_headroom_guardrail_compression(self):
        row = _spend_log_row(
            {
                "usage_object": {},
                "guardrail_information": [
                    {"guardrail_provider": "headroom", "guardrail_response": {"tokens_saved": 400}}
                ],
            }
        )
        summary = summarize_optimized_request(row, COST_MAP)
        assert summary is not None
        assert summary.optimizations == ["compression"]
        assert summary.savings == pytest.approx(400 * 3e-06)

    def test_compression_and_caching_both(self):
        row = _spend_log_row(
            {
                "usage_object": {"cache_read_input_tokens": 1000},
                "compression_savings": {"tokens_saved": 600},
            }
        )
        summary = summarize_optimized_request(row, COST_MAP)
        assert summary is not None
        assert summary.optimizations == ["caching", "compression"]
        assert summary.savings == pytest.approx(1000 * (3e-06 - 3e-07) + 600 * 3e-06)

    def test_metadata_as_json_string(self):
        row = _spend_log_row('{"usage_object": {"cache_read_input_tokens": 100}}')
        summary = summarize_optimized_request(row, COST_MAP)
        assert summary is not None
        assert summary.optimizations == ["caching"]

    def test_unoptimized_request_returns_none(self):
        assert summarize_optimized_request(_spend_log_row({"usage_object": {}}), COST_MAP) is None
        assert summarize_optimized_request(_spend_log_row(None), COST_MAP) is None


def _override_auth(role, user_id="user-1"):
    app.dependency_overrides[ps.user_api_key_auth] = lambda: UserAPIKeyAuth(user_role=role, user_id=user_id)


class TestActivityEndpoint:
    def _setup(self, monkeypatch, rows):
        mock_prisma = MagicMock()
        mock_prisma.db.query_raw = AsyncMock(return_value=rows)
        monkeypatch.setattr(ps, "prisma_client", mock_prisma)
        monkeypatch.setattr(litellm, "model_cost", COST_MAP)
        return mock_prisma

    def test_admin_gets_global_view(self, client, monkeypatch):
        mock_prisma = self._setup(
            monkeypatch,
            [
                {
                    "date": "2026-07-15",
                    "model": "claude-x",
                    "custom_llm_provider": "anthropic",
                    "cache_read_input_tokens": 1000,
                    "cache_creation_input_tokens": 0,
                    "compression_saved_tokens": 0,
                    "spend": 1.0,
                }
            ],
        )
        _override_auth(LitellmUserRoles.PROXY_ADMIN)
        try:
            response = client.get(
                "/cost_savings/activity", params={"start_date": "2026-07-09", "end_date": "2026-07-15"}
            )
            assert response.status_code == 200
            body = response.json()
            assert body["totals"]["cache_savings"] == pytest.approx(1000 * (3e-06 - 3e-07))
            sql, *params = mock_prisma.db.query_raw.await_args.args
            assert "user_id" not in sql
            assert params == ["2026-07-09", "2026-07-15"]
        finally:
            app.dependency_overrides.pop(ps.user_api_key_auth, None)

    def test_non_admin_scoped_to_own_user_id(self, client, monkeypatch):
        mock_prisma = self._setup(monkeypatch, [])
        _override_auth(LitellmUserRoles.INTERNAL_USER, user_id="user-42")
        try:
            response = client.get(
                "/cost_savings/activity", params={"start_date": "2026-07-09", "end_date": "2026-07-15"}
            )
            assert response.status_code == 200
            sql, *params = mock_prisma.db.query_raw.await_args.args
            assert "user_id = $3" in sql
            assert params == ["2026-07-09", "2026-07-15", "user-42"]
        finally:
            app.dependency_overrides.pop(ps.user_api_key_auth, None)

    def test_invalid_date_rejected(self, client, monkeypatch):
        self._setup(monkeypatch, [])
        _override_auth(LitellmUserRoles.PROXY_ADMIN)
        try:
            response = client.get(
                "/cost_savings/activity", params={"start_date": "not-a-date", "end_date": "2026-07-15"}
            )
            assert response.status_code == 400
        finally:
            app.dependency_overrides.pop(ps.user_api_key_auth, None)


class TestRecentRequestsEndpoint:
    def test_non_admin_scoped_and_filtered(self, client, monkeypatch):
        optimized = {
            "request_id": "req_hit",
            "startTime": datetime(2026, 7, 15, 12, 0, 0, tzinfo=timezone.utc),
            "model": "claude-x",
            "custom_llm_provider": "anthropic",
            "total_tokens": 1500,
            "spend": 0.01,
            "metadata": {"usage_object": {"cache_read_input_tokens": 1000}},
        }
        plain = {**optimized, "request_id": "req_plain", "metadata": {"usage_object": {}}}
        mock_prisma = MagicMock()
        mock_prisma.db.litellm_spendlogs.find_many = AsyncMock(
            return_value=[MagicMock(**optimized), MagicMock(**plain)]
        )
        monkeypatch.setattr(ps, "prisma_client", mock_prisma)
        monkeypatch.setattr(litellm, "model_cost", COST_MAP)
        _override_auth(LitellmUserRoles.INTERNAL_USER, user_id="user-42")
        try:
            response = client.get(
                "/cost_savings/recent_requests",
                params={"start_date": "2026-07-09", "end_date": "2026-07-15"},
            )
            assert response.status_code == 200
            body = response.json()
            assert [request["request_id"] for request in body["requests"]] == ["req_hit"]
            assert body["scanned_requests"] == 2
            where = mock_prisma.db.litellm_spendlogs.find_many.await_args.kwargs["where"]
            assert where["user"] == "user-42"
        finally:
            app.dependency_overrides.pop(ps.user_api_key_auth, None)
