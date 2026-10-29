"""The spend of a team key must not feed the user budget unless ``general_settings.
apply_user_budget_to_team_keys`` is True: the user budget check already skips team keys
(auth_checks.py), so measuring their spend against ``spend:user:`` and the user's
``LiteLLM_UserTable.spend`` row would fill the owner's personal envelope with the team's
spend (422 on their personal key). The same predicate drives the admission counter keys,
the post-call Redis increment, and the DB writer, so a reseed of ``spend:user:`` from
``LiteLLM_UserTable.spend`` cannot reintroduce team spend."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import litellm
import litellm.proxy.proxy_server as ps
from litellm.proxy._types import Litellm_EntityType, UserAPIKeyAuth
from litellm.proxy.db.db_spend_update_writer import DBSpendUpdateWriter
from litellm.proxy.db.db_transaction_queue.spend_update_queue import SpendUpdateQueue
from litellm.proxy.db.spend_counter_reseed import SpendCounterReseed
from litellm.proxy.spend_tracking.spend_counter_batch import (
    admission_counter_keys,
    post_call_counter_keys,
)

TEAM_TOKEN = UserAPIKeyAuth(token="hashed", team_id="t1", user_id="u1", org_id=None)


@pytest.fixture
def toggle_off(monkeypatch):
    """Default proxy setting: team keys are exempt from the user budget."""
    monkeypatch.setitem(ps.general_settings, "apply_user_budget_to_team_keys", False)


@pytest.fixture
def toggle_on(monkeypatch):
    monkeypatch.setitem(ps.general_settings, "apply_user_budget_to_team_keys", True)


def _increment_spend_counters_setup(monkeypatch):
    fake_cache = MagicMock()
    fake_cache.in_memory_cache.get_cache = MagicMock(return_value=None)
    fake_cache.redis_cache = MagicMock()
    fake_cache.redis_cache.async_get_cache = AsyncMock(return_value=None)
    recorded: dict[str, float] = {}

    async def _record_pipeline(increment_list, **_):
        results = []
        for op in increment_list:
            recorded[op["key"]] = recorded.get(op["key"], 0.0) + op["increment_value"]
            results.append(recorded[op["key"]])
        return results

    fake_cache.redis_cache.async_increment_pipeline = AsyncMock(side_effect=_record_pipeline)
    fake_user_cache = MagicMock()
    fake_user_cache.async_get_cache = AsyncMock(return_value=None)
    monkeypatch.setattr(ps, "spend_counter_cache", fake_cache)
    monkeypatch.setattr(ps, "user_api_key_cache", fake_user_cache)
    monkeypatch.setattr(ps, "prisma_client", None)
    monkeypatch.setattr(ps.SpendCounterReseed, "coalesced", AsyncMock(return_value=None))
    return recorded


async def _post_call_user_transaction(response_cost=5.0, user_id="u1", team_id="t1", end_user_id=None):
    """Drive the DB-side construction the way ``update_database`` does and flush the queue."""
    db_writer = DBSpendUpdateWriter()
    db_writer.spend_update_queue = SpendUpdateQueue()
    with patch.object(litellm, "max_budget", 0):
        await db_writer._update_user_db(
            response_cost=response_cost,
            user_id=user_id,
            prisma_client=MagicMock(),
            litellm_proxy_budget_name=None,
            end_user_id=end_user_id,
            team_id=team_id,
        )
    return await db_writer.spend_update_queue.flush_all_updates_from_in_memory_queue()


# 1. Team key, toggle off: the user counter and the user DB row are left alone,
#    the team counters still get their spend.
@pytest.mark.asyncio
async def test_team_key_toggle_off_never_touches_the_user_counter_or_user_row(toggle_off, monkeypatch):
    recorded = _increment_spend_counters_setup(monkeypatch)

    await ps.increment_spend_counters(token="hashed", team_id="t1", user_id="u1", response_cost=5.0)

    assert recorded == {
        "spend:key:hashed": 5.0,
        "spend:team:t1": 5.0,
        "spend:team_member:u1:t1": 5.0,
    }

    queued = await _post_call_user_transaction()
    assert [u["entity_type"] for u in queued] == []  # no user_list transaction in the DB payload


@pytest.mark.asyncio
async def test_team_key_toggle_off_keeps_team_transactions_in_the_db_payload(toggle_off):
    db_writer = DBSpendUpdateWriter()
    db_writer.spend_update_queue = SpendUpdateQueue()
    await db_writer._update_team_db(
        response_cost=5.0, team_id="t1", user_id="u1", prisma_client=MagicMock()
    )
    queued = await db_writer.spend_update_queue.flush_all_updates_from_in_memory_queue()
    team_transactions = [u for u in queued if u["entity_type"] == Litellm_EntityType.TEAM]
    assert [u["entity_id"] for u in team_transactions] == ["t1"]  # team transaction present
    assert all(u["entity_type"] != Litellm_EntityType.USER for u in queued)


# 2. Team key, toggle on: historical behavior, the user is fully counted.
@pytest.mark.asyncio
async def test_team_key_toggle_on_counts_the_user_everywhere(toggle_on, monkeypatch):
    recorded = _increment_spend_counters_setup(monkeypatch)

    await ps.increment_spend_counters(token="hashed", team_id="t1", user_id="u1", response_cost=5.0)

    assert recorded["spend:user:u1"] == 5.0
    assert recorded["spend:team:t1"] == 5.0

    queued = await _post_call_user_transaction()
    user_transactions = [u for u in queued if u["entity_type"] == Litellm_EntityType.USER]
    assert [u["entity_id"] for u in user_transactions] == ["u1"]


# 3. Personal key (no team_id), toggle off: unchanged.
@pytest.mark.asyncio
async def test_personal_key_toggle_off_is_unchanged(toggle_off, monkeypatch):
    recorded = _increment_spend_counters_setup(monkeypatch)

    await ps.increment_spend_counters(token="hashed", team_id=None, user_id="u1", response_cost=5.0)

    assert recorded["spend:user:u1"] == 5.0

    queued = await _post_call_user_transaction(team_id=None)
    user_transactions = [u for u in queued if u["entity_type"] == Litellm_EntityType.USER]
    assert [u["entity_id"] for u in user_transactions] == ["u1"]


# 4. Admission: a team key's admission counter keys exclude spend:user: unless the toggle is on.
@pytest.mark.asyncio
async def test_admission_counter_keys_exclude_user_counter_for_team_keys(toggle_off):
    without_toggle = admission_counter_keys(TEAM_TOKEN, end_user_id=None)
    assert "spend:user:u1" not in without_toggle
    assert {"spend:key:hashed", "spend:team:t1", "spend:team_member:u1:t1"} <= without_toggle

    post_call_off = post_call_counter_keys(
        token="hashed", team_id="t1", user_id="u1", org_id=None, end_user_id=None, tags=None, model_access_groups=None
    )
    assert "spend:user:u1" not in post_call_off


@pytest.mark.asyncio
async def test_admission_counter_keys_include_user_counter_when_toggle_is_on(toggle_on):
    assert "spend:user:u1" in admission_counter_keys(TEAM_TOKEN, end_user_id=None)
    assert "spend:user:u1" in post_call_counter_keys(
        token="hashed", team_id="t1", user_id="u1", org_id=None, end_user_id=None, tags=None, model_access_groups=None
    )


# 5. Re-seed coherence: the user counter is reseeded from LiteLLM_UserTable.spend alone, and
#    the DB writer (pinned above) never adds team spend to that row, so a reseed cannot
#    reintroduce the team's spend into spend:user:.
@pytest.mark.asyncio
async def test_user_counter_reseed_reads_only_the_user_row(toggle_off):
    prisma_client = MagicMock()
    prisma_client.db.litellm_usertable.find_unique = AsyncMock(
        return_value=SimpleNamespace(spend=51.77)  # personal-key spend only, team spend never written here
    )

    reseeded = await SpendCounterReseed.from_db(prisma_client, counter_key="spend:user:u1")

    assert reseeded == 51.77
    prisma_client.db.litellm_usertable.find_unique.assert_awaited_once_with(where={"user_id": "u1"})
