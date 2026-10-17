"""Tests for the per-team cumulative token budget check.

team_max_tokens_check mirrors team_max_budget_check: a team can cap tokens
directly (max_tokens_budget + budget_duration) because a dollar cap drifts once
the team mixes models at different per-token prices. Usage is read from the
cross-pod tokens:team:{id} counter (get_current_tokens_used), falling back to
the tokens_used column loaded in auth.
"""

from unittest.mock import AsyncMock, patch

import pytest

import litellm
from litellm.proxy._types import Litellm_EntityType, UserAPIKeyAuth
from litellm.models.team import LiteLLM_TeamTable
from litellm.proxy.auth.auth_checks import team_max_tokens_check
from litellm.proxy.utils import ProxyLogging


def _proxy_logging_obj() -> ProxyLogging:
    obj = ProxyLogging.__new__(ProxyLogging)

    async def _noop(*args, **kwargs):
        return None

    obj.budget_alerts = _noop  # type: ignore[attr-defined]
    return obj


def _patch_tokens(value):
    async def _effect(counter_key, fallback_tokens):
        return value if value is not None else int(fallback_tokens or 0)

    return patch(
        "litellm.proxy.proxy_server.get_current_tokens_used",
        new_callable=AsyncMock,
        side_effect=_effect,
    )


@pytest.mark.asyncio
async def test_no_budget_set_is_skipped():
    """A team with no token budget is never blocked, so existing teams are unaffected."""
    team = LiteLLM_TeamTable(team_id="t1", tokens_used=10**9)
    token = UserAPIKeyAuth(token="hashed", team_id="t1")
    with _patch_tokens(10**9) as get_tokens:
        result = await team_max_tokens_check(
            team_object=team, valid_token=token, proxy_logging_obj=_proxy_logging_obj()
        )
    assert result is None
    assert get_tokens.await_count == 0


@pytest.mark.asyncio
async def test_under_budget_passes():
    team = LiteLLM_TeamTable(team_id="t1", max_tokens_budget=1_000_000, tokens_used=0)
    token = UserAPIKeyAuth(token="hashed", team_id="t1")
    with _patch_tokens(500_000) as get_tokens:
        result = await team_max_tokens_check(
            team_object=team, valid_token=token, proxy_logging_obj=_proxy_logging_obj()
        )
    assert result is None
    assert get_tokens.await_count == 1


@pytest.mark.asyncio
async def test_over_budget_raises():
    team = LiteLLM_TeamTable(team_id="t1", max_tokens_budget=1_000_000, tokens_used=0)
    token = UserAPIKeyAuth(token="hashed", team_id="t1")
    with _patch_tokens(1_200_000):
        with pytest.raises(litellm.BudgetExceededError) as exc:
            await team_max_tokens_check(team_object=team, valid_token=token, proxy_logging_obj=_proxy_logging_obj())
    assert exc.value.current_cost == 1_200_000
    assert exc.value.max_budget == 1_000_000
    assert exc.value.entity_type == Litellm_EntityType.TEAM.value


@pytest.mark.asyncio
async def test_cold_counter_falls_back_to_db_tokens_used():
    """When the counter is cold (None), usage comes from the tokens_used column
    loaded in auth; an over-budget column still blocks."""
    team = LiteLLM_TeamTable(team_id="t1", max_tokens_budget=1_000_000, tokens_used=1_500_000)
    token = UserAPIKeyAuth(token="hashed", team_id="t1")
    with _patch_tokens(None):
        with pytest.raises(litellm.BudgetExceededError):
            await team_max_tokens_check(team_object=team, valid_token=token, proxy_logging_obj=_proxy_logging_obj())


@pytest.mark.asyncio
async def test_exactly_at_cap_passes():
    """Enforcement is strictly greater-than, matching the dollar budget check."""
    team = LiteLLM_TeamTable(team_id="t1", max_tokens_budget=1_000_000, tokens_used=0)
    token = UserAPIKeyAuth(token="hashed", team_id="t1")
    with _patch_tokens(1_000_000) as get_tokens:
        result = await team_max_tokens_check(
            team_object=team, valid_token=token, proxy_logging_obj=_proxy_logging_obj()
        )
    assert result is None
    assert get_tokens.await_count == 1


@pytest.mark.asyncio
async def test_get_current_tokens_used_prefers_the_cross_pod_counter():
    """A populated tokens:team counter is authoritative over the DB column."""
    from litellm.proxy.proxy_server import get_current_tokens_used

    async def _counter(counter_key):
        return 500_000, None

    with patch("litellm.proxy.proxy_server.read_spend_counter_cache_value", _counter):
        used = await get_current_tokens_used(counter_key="tokens:team:t1", fallback_tokens=7)
    assert used == 500_000


@pytest.mark.asyncio
async def test_get_current_tokens_used_falls_back_to_db_when_counter_cold():
    """A cold counter (None) falls back to the tokens_used column loaded in auth."""
    from litellm.proxy.proxy_server import get_current_tokens_used

    async def _cold(counter_key):
        return None, None

    with patch("litellm.proxy.proxy_server.read_spend_counter_cache_value", _cold):
        used = await get_current_tokens_used(counter_key="tokens:team:t1", fallback_tokens=1_234)
    assert used == 1_234


def test_extract_total_tokens_reads_the_standard_logging_payload():
    """The counter delta is the StandardLoggingPayload total, the DB write's source."""
    from litellm.proxy.hooks.proxy_track_cost_callback import _extract_total_tokens

    assert _extract_total_tokens({"total_tokens": 4242}) == 4242


def test_extract_total_tokens_is_zero_without_a_payload():
    """No payload leaves the counter untouched rather than raising."""
    from litellm.proxy.hooks.proxy_track_cost_callback import _extract_total_tokens

    assert _extract_total_tokens(None) == 0


@pytest.mark.asyncio
async def test_get_current_tokens_used_floors_at_recorded_db_usage():
    """A counter that reseeded low after a TTL expiry, or missed an increment when its
    logging payload failed, must not read below the tokens_used column loaded in auth,
    otherwise a team slips past the cap once the counter sits under recorded usage."""
    from litellm.proxy.proxy_server import get_current_tokens_used

    async def _stale_low(counter_key):
        return 200, None

    with patch("litellm.proxy.proxy_server.read_spend_counter_cache_value", _stale_low):
        used = await get_current_tokens_used(counter_key="tokens:team:t1", fallback_tokens=1_100)
    assert used == 1_100


@pytest.mark.asyncio
async def test_unpriced_model_advances_the_team_token_counter():  # test-quality-ok: no in-process counter state
    """An unpriced model returns response_cost=None and skips the spend pipeline, but its
    tokens must still count against the team, otherwise it bypasses max_tokens_budget. The
    cross-pod counter is Redis-backed, so the issued increment is the only in-process signal."""
    from datetime import datetime

    from litellm.proxy.hooks.proxy_track_cost_callback import ProxyDBLogger

    kwargs = {
        "model": "custom-unpriced",
        "call_type": "acompletion",
        "litellm_params": {"metadata": {"user_api_key_team_id": "t1"}},
        "standard_logging_object": {
            "response_cost": None,
            "response_cost_failure_debug_info": "missing custom price",
            "request_tags": None,
            "total_tokens": 321,
        },
        "stream": False,
    }

    with patch("litellm.proxy.proxy_server.increment_spend_counter_cache", new_callable=AsyncMock) as inc:
        await ProxyDBLogger()._PROXY_track_cost_callback(
            kwargs=kwargs,
            completion_response=None,
            start_time=datetime(2026, 1, 1),
            end_time=datetime(2026, 1, 1),
        )
    inc.assert_awaited_once_with(counter_key="tokens:team:t1", increment=321)
