"""
Tests for the Postgres fallback added to the per-model budget read path.

The read methods ``_get_virtual_key_spend_for_model`` and
``_get_end_user_spend_for_model`` now check the in-memory cache first and,
on a miss, query the daily-spend Postgres tables before returning.

These tests verify:
- Cache hits short-circuit without DB queries
- Cache misses query Postgres and populate the cache
- Postgres returning no rows yields 0.0
- Postgres errors fail open (return None, don't block)
- No prisma_client returns None
- Helper functions (_budget_window_start_date, _strip_provider_prefix)
- End-to-end: budget enforcement with DB fallback
"""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import litellm
from litellm.caching.caching import DualCache
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.hooks.model_max_budget_limiter import (
    END_USER_SPEND_CACHE_KEY_PREFIX,
    VIRTUAL_KEY_SPEND_CACHE_KEY_PREFIX,
    _MODEL_SPEND_CACHE_TTL_SECONDS,
    _PROXY_VirtualKeyModelMaxBudgetLimiter,
    _budget_window_start_date,
    _strip_provider_prefix,
)
from litellm.types.utils import BudgetConfig

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def budget_limiter():
    dual_cache = DualCache()
    return _PROXY_VirtualKeyModelMaxBudgetLimiter(dual_cache=dual_cache)


@pytest.fixture
def budget_config_1d():
    return BudgetConfig(budget_limit=100.0, time_period="1d")


# ---------------------------------------------------------------------------
# Unit tests: _strip_provider_prefix
# ---------------------------------------------------------------------------


class TestStripProviderPrefix:
    def test_should_strip_openai_prefix(self):
        assert _strip_provider_prefix("openai/gpt-4") == "gpt-4"

    def test_should_strip_azure_prefix(self):
        assert _strip_provider_prefix("azure/gpt-4") == "gpt-4"

    def test_should_leave_bare_model_unchanged(self):
        assert _strip_provider_prefix("gpt-4") == "gpt-4"

    def test_should_handle_multiple_slashes(self):
        assert _strip_provider_prefix("azure/openai/gpt-4") == "openai/gpt-4"

    def test_should_handle_empty_string(self):
        assert _strip_provider_prefix("") == ""


# ---------------------------------------------------------------------------
# Unit tests: _budget_window_start_date
# ---------------------------------------------------------------------------


class TestBudgetWindowStartDate:
    def test_should_return_date_string_for_1d(self):
        result = _budget_window_start_date("1d")
        # Should be yesterday or today depending on timing
        parsed = datetime.strptime(result, "%Y-%m-%d")
        assert parsed is not None
        # Should be within 2 days of now
        now = datetime.now(timezone.utc)
        delta = now.replace(tzinfo=None) - parsed
        assert 0 <= delta.days <= 2

    def test_should_return_date_string_for_30d(self):
        result = _budget_window_start_date("30d")
        parsed = datetime.strptime(result, "%Y-%m-%d")
        now = datetime.now(timezone.utc)
        delta = now.replace(tzinfo=None) - parsed
        assert 29 <= delta.days <= 31

    def test_should_return_at_least_1_day_for_short_durations(self):
        """Even for durations shorter than 1 day (e.g. 1h), we should
        look back at least 1 day since daily tables are per-day."""
        result = _budget_window_start_date("1h")
        parsed = datetime.strptime(result, "%Y-%m-%d")
        now = datetime.now(timezone.utc)
        delta = now.replace(tzinfo=None) - parsed
        assert delta.days >= 1


# ---------------------------------------------------------------------------
# Unit tests: _query_db_and_cache
# ---------------------------------------------------------------------------


class TestQueryDbAndCache:
    @pytest.mark.asyncio
    async def test_should_query_db(self, budget_limiter):
        """Should call db_lookup and return the result."""
        db_lookup = AsyncMock(return_value=25.0)

        result = await budget_limiter._query_db_and_cache(
            cache_key="miss-key",
            db_lookup=db_lookup,
            entity_id="ent-1",
            model="gpt-4",
            budget_duration="1d",
        )
        assert result == 25.0
        db_lookup.assert_awaited_once_with(
            entity_id="ent-1", model="gpt-4", budget_duration="1d"
        )

    @pytest.mark.asyncio
    async def test_should_populate_cache_after_db_hit(self, budget_limiter):
        """After a DB query returns a value, it should be cached."""
        db_lookup = AsyncMock(return_value=25.0)

        await budget_limiter._query_db_and_cache(
            cache_key="populate-key",
            db_lookup=db_lookup,
            entity_id="ent-1",
            model="gpt-4",
            budget_duration="1d",
        )

        # Now verify the cache was populated
        cached = await budget_limiter.dual_cache.async_get_cache(key="populate-key")
        assert cached == 25.0

    @pytest.mark.asyncio
    async def test_should_not_populate_cache_on_db_none(self, budget_limiter):
        """When DB returns None (e.g. no prisma_client), cache should not be
        populated with None."""
        db_lookup = AsyncMock(return_value=None)

        result = await budget_limiter._query_db_and_cache(
            cache_key="none-key",
            db_lookup=db_lookup,
            entity_id="ent-1",
            model="gpt-4",
            budget_duration="1d",
        )
        assert result is None

        cached = await budget_limiter.dual_cache.async_get_cache(key="none-key")
        assert cached is None

    @pytest.mark.asyncio
    async def test_should_return_none_on_db_exception(self, budget_limiter):
        """DB errors should fail open — return None, don't block."""
        db_lookup = AsyncMock(side_effect=Exception("connection refused"))

        result = await budget_limiter._query_db_and_cache(
            cache_key="error-key",
            db_lookup=db_lookup,
            entity_id="ent-1",
            model="gpt-4",
            budget_duration="1d",
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_should_return_zero_from_db(self, budget_limiter):
        """When DB returns 0.0 (no spend yet), cache it and return 0.0."""
        db_lookup = AsyncMock(return_value=0.0)

        result = await budget_limiter._query_db_and_cache(
            cache_key="zero-key",
            db_lookup=db_lookup,
            entity_id="ent-1",
            model="gpt-4",
            budget_duration="1d",
        )
        assert result == 0.0

        # 0.0 should be cached
        cached = await budget_limiter.dual_cache.async_get_cache(key="zero-key")
        assert cached == 0.0


# ---------------------------------------------------------------------------
# Unit tests: Postgres query methods
# ---------------------------------------------------------------------------


class TestQueryEndUserModelSpend:
    @pytest.mark.asyncio
    async def test_should_return_none_when_no_prisma(self):
        """When prisma_client is None, return None."""
        with patch(
            "litellm.proxy.proxy_server.prisma_client",
            None,
        ):
            result = await _PROXY_VirtualKeyModelMaxBudgetLimiter._query_end_user_model_spend(
                entity_id="user-1",
                model="gpt-4",
                budget_duration="1d",
            )
            assert result is None

    @pytest.mark.asyncio
    async def test_should_sum_spend_from_rows(self):
        """Should sum the spend field from returned rows."""
        row1 = SimpleNamespace(spend=10.0)
        row2 = SimpleNamespace(spend=5.5)
        mock_prisma = MagicMock()
        mock_prisma.db.litellm_dailyenduserspend.find_many = AsyncMock(
            return_value=[row1, row2]
        )

        with patch(
            "litellm.proxy.proxy_server.prisma_client",
            mock_prisma,
        ):
            result = await _PROXY_VirtualKeyModelMaxBudgetLimiter._query_end_user_model_spend(
                entity_id="user-1",
                model="gpt-4",
                budget_duration="1d",
            )
            assert result == 15.5

    @pytest.mark.asyncio
    async def test_should_return_zero_when_no_rows(self):
        """No rows means zero spend."""
        mock_prisma = MagicMock()
        mock_prisma.db.litellm_dailyenduserspend.find_many = AsyncMock(return_value=[])

        with patch(
            "litellm.proxy.proxy_server.prisma_client",
            mock_prisma,
        ):
            result = await _PROXY_VirtualKeyModelMaxBudgetLimiter._query_end_user_model_spend(
                entity_id="user-1",
                model="gpt-4",
                budget_duration="1d",
            )
            assert result == 0.0

    @pytest.mark.asyncio
    async def test_should_pass_correct_where_clause(self):
        """Verify the Prisma query uses model_group and date filter."""
        mock_prisma = MagicMock()
        mock_find = AsyncMock(return_value=[])
        mock_prisma.db.litellm_dailyenduserspend.find_many = mock_find

        with patch(
            "litellm.proxy.proxy_server.prisma_client",
            mock_prisma,
        ):
            await _PROXY_VirtualKeyModelMaxBudgetLimiter._query_end_user_model_spend(
                entity_id="user-1",
                model="openai/gpt-4",
                budget_duration="7d",
            )

        mock_find.assert_awaited_once()
        where = mock_find.call_args.kwargs["where"]
        assert where["end_user_id"] == "user-1"
        assert "gte" in where["date"]
        # Should include both model variants in OR clause
        or_clause = where["OR"]
        model_groups = [c["model_group"] for c in or_clause]
        assert "openai/gpt-4" in model_groups
        assert "gpt-4" in model_groups


class TestQueryVirtualKeyModelSpend:
    @pytest.mark.asyncio
    async def test_should_return_none_when_no_prisma(self):
        with patch(
            "litellm.proxy.proxy_server.prisma_client",
            None,
        ):
            result = await _PROXY_VirtualKeyModelMaxBudgetLimiter._query_virtual_key_model_spend(
                entity_id="hash-1",
                model="gpt-4",
                budget_duration="1d",
            )
            assert result is None

    @pytest.mark.asyncio
    async def test_should_sum_spend_from_rows(self):
        row1 = SimpleNamespace(spend=20.0)
        row2 = SimpleNamespace(spend=3.0)
        mock_prisma = MagicMock()
        mock_prisma.db.litellm_dailyuserspend.find_many = AsyncMock(
            return_value=[row1, row2]
        )

        with patch(
            "litellm.proxy.proxy_server.prisma_client",
            mock_prisma,
        ):
            result = await _PROXY_VirtualKeyModelMaxBudgetLimiter._query_virtual_key_model_spend(
                entity_id="hash-1",
                model="gpt-4",
                budget_duration="1d",
            )
            assert result == 23.0

    @pytest.mark.asyncio
    async def test_should_use_api_key_in_where_clause(self):
        mock_prisma = MagicMock()
        mock_find = AsyncMock(return_value=[])
        mock_prisma.db.litellm_dailyuserspend.find_many = mock_find

        with patch(
            "litellm.proxy.proxy_server.prisma_client",
            mock_prisma,
        ):
            await _PROXY_VirtualKeyModelMaxBudgetLimiter._query_virtual_key_model_spend(
                entity_id="hash-abc",
                model="gpt-4",
                budget_duration="30d",
            )

        where = mock_find.call_args.kwargs["where"]
        assert where["api_key"] == "hash-abc"


# ---------------------------------------------------------------------------
# Unit tests: two-key cache lookup (provider prefix stripping)
# ---------------------------------------------------------------------------


class TestTwoKeyCacheLookup:
    @pytest.mark.asyncio
    async def test_should_hit_cache_with_stripped_prefix(
        self, budget_limiter, budget_config_1d
    ):
        """If cache was written with bare model name but read is requested
        with provider prefix, the stripped-key fallback should find it."""
        # Write path uses bare model name
        bare_key = f"{VIRTUAL_KEY_SPEND_CACHE_KEY_PREFIX}:key-hash:gpt-4:1d"
        await budget_limiter.dual_cache.async_set_cache(key=bare_key, value=30.0)

        with patch.object(budget_limiter, "_query_virtual_key_model_spend") as mock_db:
            result = await budget_limiter._get_virtual_key_spend_for_model(
                user_api_key_hash="key-hash",
                model="openai/gpt-4",  # request uses provider prefix
                key_budget_config=budget_config_1d,
            )
            assert result == 30.0
            mock_db.assert_not_called()

    @pytest.mark.asyncio
    async def test_should_hit_cache_with_primary_key_first(
        self, budget_limiter, budget_config_1d
    ):
        """If cache has BOTH keys, the primary (with prefix) should win."""
        primary_key = f"{VIRTUAL_KEY_SPEND_CACHE_KEY_PREFIX}:key-hash:openai/gpt-4:1d"
        bare_key = f"{VIRTUAL_KEY_SPEND_CACHE_KEY_PREFIX}:key-hash:gpt-4:1d"
        await budget_limiter.dual_cache.async_set_cache(key=primary_key, value=10.0)
        await budget_limiter.dual_cache.async_set_cache(key=bare_key, value=99.0)

        result = await budget_limiter._get_virtual_key_spend_for_model(
            user_api_key_hash="key-hash",
            model="openai/gpt-4",
            key_budget_config=budget_config_1d,
        )
        assert result == 10.0  # primary wins

    @pytest.mark.asyncio
    async def test_should_not_try_stripped_key_when_no_prefix(
        self, budget_limiter, budget_config_1d
    ):
        """When model has no prefix, stripped == model, so only one lookup."""
        with patch.object(
            budget_limiter.dual_cache,
            "async_get_cache",
            new_callable=AsyncMock,
            return_value=None,
        ) as mock_cache, patch.object(
            budget_limiter,
            "_query_virtual_key_model_spend",
            new_callable=AsyncMock,
            return_value=0.0,
        ):
            await budget_limiter._get_virtual_key_spend_for_model(
                user_api_key_hash="key-hash",
                model="gpt-4",  # no provider prefix
                key_budget_config=budget_config_1d,
            )
            # Should only call cache once (no alt key when stripped == model)
            assert mock_cache.call_count == 1

    @pytest.mark.asyncio
    async def test_end_user_should_hit_stripped_cache(
        self, budget_limiter, budget_config_1d
    ):
        """End-user path should also support the two-key lookup."""
        bare_key = f"{END_USER_SPEND_CACHE_KEY_PREFIX}:eu-1:gpt-4:1d"
        await budget_limiter.dual_cache.async_set_cache(key=bare_key, value=7.5)

        with patch.object(budget_limiter, "_query_end_user_model_spend") as mock_db:
            result = await budget_limiter._get_end_user_spend_for_model(
                end_user_id="eu-1",
                model="openai/gpt-4",
                key_budget_config=budget_config_1d,
            )
            assert result == 7.5
            mock_db.assert_not_called()


# ---------------------------------------------------------------------------
# Integration tests: read path with DB fallback
# ---------------------------------------------------------------------------


class TestVirtualKeySpendWithDbFallback:
    @pytest.mark.asyncio
    async def test_should_use_cache_when_available(
        self, budget_limiter, budget_config_1d
    ):
        """Cache hit should return without querying DB."""
        cache_key = f"{VIRTUAL_KEY_SPEND_CACHE_KEY_PREFIX}:key-hash:gpt-4:1d"
        await budget_limiter.dual_cache.async_set_cache(key=cache_key, value=42.0)

        with patch.object(budget_limiter, "_query_virtual_key_model_spend") as mock_db:
            result = await budget_limiter._get_virtual_key_spend_for_model(
                user_api_key_hash="key-hash",
                model="gpt-4",
                key_budget_config=budget_config_1d,
            )
            assert result == 42.0
            mock_db.assert_not_called()

    @pytest.mark.asyncio
    async def test_should_fall_back_to_db_on_cache_miss(
        self, budget_limiter, budget_config_1d
    ):
        """Cache miss should query DB and return the result."""
        with patch.object(
            budget_limiter,
            "_query_virtual_key_model_spend",
            new_callable=AsyncMock,
            return_value=33.0,
        ):
            result = await budget_limiter._get_virtual_key_spend_for_model(
                user_api_key_hash="key-hash-2",
                model="gpt-4",
                key_budget_config=budget_config_1d,
            )
            assert result == 33.0

    @pytest.mark.asyncio
    async def test_should_enforce_budget_with_db_spend(self, budget_limiter):
        """End-to-end: DB reports spend over budget -> BudgetExceededError."""
        user_api_key = UserAPIKeyAuth(
            token="test-key-hash",
            key_alias="test-alias",
            model_max_budget={"gpt-4": {"budget_limit": 50.0, "time_period": "1d"}},
        )

        with patch.object(
            budget_limiter,
            "_query_virtual_key_model_spend",
            new_callable=AsyncMock,
            return_value=75.0,
        ):
            with pytest.raises(litellm.BudgetExceededError):
                await budget_limiter.is_key_within_model_budget(user_api_key, "gpt-4")

    @pytest.mark.asyncio
    async def test_should_pass_when_db_spend_within_budget(self, budget_limiter):
        """End-to-end: DB reports spend within budget -> no error."""
        user_api_key = UserAPIKeyAuth(
            token="test-key-hash",
            key_alias="test-alias",
            model_max_budget={"gpt-4": {"budget_limit": 50.0, "time_period": "1d"}},
        )

        with patch.object(
            budget_limiter,
            "_query_virtual_key_model_spend",
            new_callable=AsyncMock,
            return_value=25.0,
        ):
            result = await budget_limiter.is_key_within_model_budget(
                user_api_key, "gpt-4"
            )
            assert result is True


class TestEndUserSpendWithDbFallback:
    @pytest.mark.asyncio
    async def test_should_use_cache_when_available(
        self, budget_limiter, budget_config_1d
    ):
        cache_key = f"{END_USER_SPEND_CACHE_KEY_PREFIX}:eu-1:gpt-4:1d"
        await budget_limiter.dual_cache.async_set_cache(key=cache_key, value=18.0)

        with patch.object(budget_limiter, "_query_end_user_model_spend") as mock_db:
            result = await budget_limiter._get_end_user_spend_for_model(
                end_user_id="eu-1",
                model="gpt-4",
                key_budget_config=budget_config_1d,
            )
            assert result == 18.0
            mock_db.assert_not_called()

    @pytest.mark.asyncio
    async def test_should_fall_back_to_db_on_cache_miss(
        self, budget_limiter, budget_config_1d
    ):
        with patch.object(
            budget_limiter,
            "_query_end_user_model_spend",
            new_callable=AsyncMock,
            return_value=12.0,
        ):
            result = await budget_limiter._get_end_user_spend_for_model(
                end_user_id="eu-2",
                model="gpt-4",
                key_budget_config=budget_config_1d,
            )
            assert result == 12.0

    @pytest.mark.asyncio
    async def test_should_enforce_budget_with_db_spend(self, budget_limiter):
        """End-to-end: DB reports spend over budget -> BudgetExceededError."""
        with patch.object(
            budget_limiter,
            "_query_end_user_model_spend",
            new_callable=AsyncMock,
            return_value=200.0,
        ):
            with pytest.raises(litellm.BudgetExceededError):
                await budget_limiter.is_end_user_within_model_budget(
                    end_user_id="eu-3",
                    end_user_model_max_budget={
                        "gpt-4": {"budget_limit": 100.0, "time_period": "1d"}
                    },
                    model="gpt-4",
                )

    @pytest.mark.asyncio
    async def test_should_pass_when_db_spend_within_budget(self, budget_limiter):
        with patch.object(
            budget_limiter,
            "_query_end_user_model_spend",
            new_callable=AsyncMock,
            return_value=50.0,
        ):
            result = await budget_limiter.is_end_user_within_model_budget(
                end_user_id="eu-4",
                end_user_model_max_budget={
                    "gpt-4": {"budget_limit": 100.0, "time_period": "1d"}
                },
                model="gpt-4",
            )
            assert result is True


# ---------------------------------------------------------------------------
# Tests: DB failure does not block requests
# ---------------------------------------------------------------------------


class TestFailOpen:
    @pytest.mark.asyncio
    async def test_should_not_block_when_db_fails_for_virtual_key(self, budget_limiter):
        """If DB query raises, budget check should pass (fail open)."""
        user_api_key = UserAPIKeyAuth(
            token="test-key-hash",
            key_alias="test-alias",
            model_max_budget={"gpt-4": {"budget_limit": 50.0, "time_period": "1d"}},
        )

        with patch.object(
            budget_limiter,
            "_query_virtual_key_model_spend",
            new_callable=AsyncMock,
            side_effect=Exception("DB connection refused"),
        ):
            # Should NOT raise — fail open
            result = await budget_limiter.is_key_within_model_budget(
                user_api_key, "gpt-4"
            )
            assert result is True

    @pytest.mark.asyncio
    async def test_should_not_block_when_db_fails_for_end_user(self, budget_limiter):
        with patch.object(
            budget_limiter,
            "_query_end_user_model_spend",
            new_callable=AsyncMock,
            side_effect=Exception("DB connection refused"),
        ):
            result = await budget_limiter.is_end_user_within_model_budget(
                end_user_id="eu-5",
                end_user_model_max_budget={
                    "gpt-4": {"budget_limit": 100.0, "time_period": "1d"}
                },
                model="gpt-4",
            )
            assert result is True

    @pytest.mark.asyncio
    async def test_should_not_block_when_no_prisma_client(self, budget_limiter):
        """No DB connection at all should fail open."""
        user_api_key = UserAPIKeyAuth(
            token="test-key-hash",
            key_alias="test-alias",
            model_max_budget={"gpt-4": {"budget_limit": 50.0, "time_period": "1d"}},
        )

        with patch(
            "litellm.proxy.proxy_server.prisma_client",
            None,
        ):
            result = await budget_limiter.is_key_within_model_budget(
                user_api_key, "gpt-4"
            )
            assert result is True


# ---------------------------------------------------------------------------
# Tests: write path still uses parent's _increment_spend_for_key
# ---------------------------------------------------------------------------


class TestWritePathPreserved:
    @pytest.mark.asyncio
    async def test_should_still_call_increment_spend_for_key(self, budget_limiter):
        """async_log_success_event should still call the inherited
        _increment_spend_for_key so single-instance/no-DB setups keep
        working."""
        virtual_key = "test-key-hash"
        model = "gpt-4"
        budget_duration = "1d"
        kwargs = {
            "standard_logging_object": {
                "response_cost": 0.05,
                "model": model,
                "metadata": {"user_api_key_hash": virtual_key},
            },
            "litellm_params": {
                "metadata": {
                    "user_api_key_model_max_budget": {
                        model: {
                            "budget_limit": 100.0,
                            "time_period": budget_duration,
                        },
                    }
                },
            },
        }
        with patch.object(
            budget_limiter,
            "_increment_spend_for_key",
            new_callable=AsyncMock,
        ) as mock_increment:
            await budget_limiter.async_log_success_event(
                kwargs, response_obj=None, start_time=None, end_time=None
            )
            mock_increment.assert_awaited_once()
            call_kwargs = mock_increment.call_args.kwargs
            assert call_kwargs["response_cost"] == 0.05
            assert (
                f"{VIRTUAL_KEY_SPEND_CACHE_KEY_PREFIX}:{virtual_key}:{model}:{budget_duration}"
                in call_kwargs["spend_key"]
            )


# ---------------------------------------------------------------------------
# Tests: None budget_duration skips DB fallback
# ---------------------------------------------------------------------------


class TestNoneBudgetDurationSkipsDb:
    @pytest.mark.asyncio
    async def test_should_return_none_for_virtual_key_when_no_duration(
        self, budget_limiter
    ):
        """When budget_duration is None the DB fallback should be skipped
        entirely and the method should return None."""
        config = BudgetConfig(budget_limit=100.0)  # no time_period
        with patch.object(
            budget_limiter,
            "_query_db_and_cache",
            new_callable=AsyncMock,
        ) as mock_db:
            result = await budget_limiter._get_virtual_key_spend_for_model(
                user_api_key_hash="key-hash",
                model="gpt-4",
                key_budget_config=config,
            )
            assert result is None
            mock_db.assert_not_called()

    @pytest.mark.asyncio
    async def test_should_return_none_for_end_user_when_no_duration(
        self, budget_limiter
    ):
        """Same as above but for the end-user path."""
        config = BudgetConfig(budget_limit=100.0)  # no time_period
        with patch.object(
            budget_limiter,
            "_query_db_and_cache",
            new_callable=AsyncMock,
        ) as mock_db:
            result = await budget_limiter._get_end_user_spend_for_model(
                end_user_id="eu-1",
                model="gpt-4",
                key_budget_config=config,
            )
            assert result is None
            mock_db.assert_not_called()
