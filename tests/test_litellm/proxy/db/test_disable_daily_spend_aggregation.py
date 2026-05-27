"""Tests for ``general_settings.disable_daily_spend_aggregation`` (LIT-3332)."""
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest


def _writer():
    from litellm.proxy.db.db_spend_update_writer import DBSpendUpdateWriter

    w = DBSpendUpdateWriter()
    w.add_spend_log_transaction_to_daily_user_transaction = AsyncMock()
    w.add_spend_log_transaction_to_daily_end_user_transaction = AsyncMock()
    w.add_spend_log_transaction_to_daily_agent_transaction = AsyncMock()
    w.add_spend_log_transaction_to_daily_team_transaction = AsyncMock()
    w.add_spend_log_transaction_to_daily_org_transaction = AsyncMock()
    w.add_spend_log_transaction_to_daily_tag_transaction = AsyncMock()
    w._update_user_db = AsyncMock()
    w._update_key_db = AsyncMock()
    w._update_team_db = AsyncMock()
    w._update_org_db = AsyncMock()
    w._update_tag_db = AsyncMock()
    w._update_agent_db = AsyncMock()
    return w


def _payload():
    return {
        "spend": 1.5,
        "team_id": "team-1",
        "user": "user-1",
        "end_user": "eu-1",
        "agent_id": "agent-1",
        "model": "gpt-4o-mini",
        "startTime": datetime.now().isoformat(),
        "endTime": datetime.now().isoformat(),
    }


async def _run_batch(w):
    await w._batch_database_updates(
        response_cost=1.5,
        user_id="user-1",
        hashed_token="hash",
        team_id="team-1",
        org_id="org-1",
        end_user_id="eu-1",
        prisma_client=MagicMock(),
        user_api_key_cache=MagicMock(),
        litellm_proxy_budget_name=None,
        payload_copy=_payload(),
        request_tags=["tag-a", "tag-b"],
    )


def _daily_methods(w):
    return [
        w.add_spend_log_transaction_to_daily_user_transaction,
        w.add_spend_log_transaction_to_daily_end_user_transaction,
        w.add_spend_log_transaction_to_daily_agent_transaction,
        w.add_spend_log_transaction_to_daily_team_transaction,
        w.add_spend_log_transaction_to_daily_org_transaction,
        w.add_spend_log_transaction_to_daily_tag_transaction,
    ]


def _non_daily_methods(w):
    return [
        w._update_user_db,
        w._update_key_db,
        w._update_team_db,
        w._update_org_db,
        w._update_tag_db,
        w._update_agent_db,
    ]


@pytest.mark.asyncio
async def test_default_runs_all_daily_writers(monkeypatch):
    monkeypatch.delenv("LITELLM_DISABLE_DAILY_SPEND_AGGREGATION", raising=False)
    import litellm.proxy.proxy_server as ps
    monkeypatch.setattr(ps, "disable_daily_spend_aggregation", False, raising=False)
    w = _writer()
    await _run_batch(w)
    for m in _daily_methods(w):
        assert m.await_count == 1


@pytest.mark.asyncio
async def test_disable_flag_skips_all_daily_writers(monkeypatch):
    monkeypatch.delenv("LITELLM_DISABLE_DAILY_SPEND_AGGREGATION", raising=False)
    import litellm.proxy.proxy_server as ps
    monkeypatch.setattr(ps, "disable_daily_spend_aggregation", True, raising=False)
    w = _writer()
    await _run_batch(w)
    for m in _daily_methods(w):
        assert m.await_count == 0
    for m in _non_daily_methods(w):
        assert m.await_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("env_value", ["1", "true", "TRUE", "yes", "on"])
async def test_env_var_skips_all_daily_writers(monkeypatch, env_value):
    monkeypatch.setenv("LITELLM_DISABLE_DAILY_SPEND_AGGREGATION", env_value)
    import litellm.proxy.proxy_server as ps
    monkeypatch.setattr(ps, "disable_daily_spend_aggregation", False, raising=False)
    w = _writer()
    await _run_batch(w)
    for m in _daily_methods(w):
        assert m.await_count == 0


@pytest.mark.parametrize(
    "env_value,expected",
    [
        (None, False), ("", False), ("0", False), ("false", False),
        ("no", False), ("off", False),
        ("1", True), ("true", True), ("TRUE", True), ("Yes", True), ("ON", True),
    ],
)
def test_should_skip_helper_env_parsing(monkeypatch, env_value, expected):
    if env_value is None:
        monkeypatch.delenv("LITELLM_DISABLE_DAILY_SPEND_AGGREGATION", raising=False)
    else:
        monkeypatch.setenv("LITELLM_DISABLE_DAILY_SPEND_AGGREGATION", env_value)
    import litellm.proxy.proxy_server as ps
    monkeypatch.setattr(ps, "disable_daily_spend_aggregation", False, raising=False)
    from litellm.proxy.db.db_spend_update_writer import DBSpendUpdateWriter
    assert DBSpendUpdateWriter()._should_skip_daily_aggregation() is expected


def test_should_skip_helper_proxy_global_takes_precedence(monkeypatch):
    monkeypatch.delenv("LITELLM_DISABLE_DAILY_SPEND_AGGREGATION", raising=False)
    import litellm.proxy.proxy_server as ps
    monkeypatch.setattr(ps, "disable_daily_spend_aggregation", True, raising=False)
    from litellm.proxy.db.db_spend_update_writer import DBSpendUpdateWriter
    assert DBSpendUpdateWriter()._should_skip_daily_aggregation() is True


def test_should_skip_helper_default_is_false(monkeypatch):
    monkeypatch.delenv("LITELLM_DISABLE_DAILY_SPEND_AGGREGATION", raising=False)
    import litellm.proxy.proxy_server as ps
    monkeypatch.setattr(ps, "disable_daily_spend_aggregation", False, raising=False)
    from litellm.proxy.db.db_spend_update_writer import DBSpendUpdateWriter
    assert DBSpendUpdateWriter()._should_skip_daily_aggregation() is False
