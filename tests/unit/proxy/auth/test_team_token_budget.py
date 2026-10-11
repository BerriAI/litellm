"""Tests for the per-team cumulative token budget check.

team_max_tokens_check mirrors team_max_budget_check: a team can cap tokens
directly (max_tokens_budget + budget_duration) because a dollar cap drifts once
the team mixes models at different per-token prices. Usage is read from the
cross-pod tokens:team:{id} counter (get_current_tokens_used), falling back to
the tokens_used column loaded in auth.
"""

from unittest.mock import patch

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
    async def _mock_get_current_tokens_used(counter_key, fallback_tokens):
        return value if value is not None else int(fallback_tokens or 0)

    return patch(
        "litellm.proxy.proxy_server.get_current_tokens_used",
        _mock_get_current_tokens_used,
    )


@pytest.mark.asyncio
async def test_no_budget_set_is_skipped():
    """A team with no token budget is never blocked, so existing teams are unaffected."""
    team = LiteLLM_TeamTable(team_id="t1", tokens_used=10**9)
    token = UserAPIKeyAuth(token="hashed", team_id="t1")
    with _patch_tokens(10**9):
        await team_max_tokens_check(
            team_object=team, valid_token=token, proxy_logging_obj=_proxy_logging_obj()
        )  # no raise


@pytest.mark.asyncio
async def test_under_budget_passes():
    team = LiteLLM_TeamTable(team_id="t1", max_tokens_budget=1_000_000, tokens_used=0)
    token = UserAPIKeyAuth(token="hashed", team_id="t1")
    with _patch_tokens(500_000):
        await team_max_tokens_check(
            team_object=team, valid_token=token, proxy_logging_obj=_proxy_logging_obj()
        )  # no raise


@pytest.mark.asyncio
async def test_over_budget_raises():
    team = LiteLLM_TeamTable(team_id="t1", max_tokens_budget=1_000_000, tokens_used=0)
    token = UserAPIKeyAuth(token="hashed", team_id="t1")
    with _patch_tokens(1_200_000):
        with pytest.raises(litellm.BudgetExceededError) as exc:
            await team_max_tokens_check(
                team_object=team, valid_token=token, proxy_logging_obj=_proxy_logging_obj()
            )
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
            await team_max_tokens_check(
                team_object=team, valid_token=token, proxy_logging_obj=_proxy_logging_obj()
            )


@pytest.mark.asyncio
async def test_exactly_at_cap_passes():
    """Enforcement is strictly greater-than, matching the dollar budget check."""
    team = LiteLLM_TeamTable(team_id="t1", max_tokens_budget=1_000_000, tokens_used=0)
    token = UserAPIKeyAuth(token="hashed", team_id="t1")
    with _patch_tokens(1_000_000):
        await team_max_tokens_check(
            team_object=team, valid_token=token, proxy_logging_obj=_proxy_logging_obj()
        )  # no raise
