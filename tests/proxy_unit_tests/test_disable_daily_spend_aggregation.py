"""
Tests for the disable_daily_spend_aggregation general_settings flag.

Verifies that when the flag is set, daily spend queue enqueuing is skipped
(preventing Redis buffer key growth that caused customer OOM), while normal
key/user/team balance updates continue to work.

Test matrix
===========
| Scenario                                        | daily queued? | balance queued? |
|-------------------------------------------------|---------------|-----------------|
| disable_daily_spend_aggregation absent/False    | yes           | yes             |
| disable_daily_spend_aggregation=True            | no            | yes             |
| ProxyUpdateSpend helper - flag absent           | returns False |                 |
| ProxyUpdateSpend helper - flag=True             | returns True  |                 |
| ProxyUpdateSpend helper - flag=False explicitly | returns False |                 |
"""

import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from litellm.proxy.db.db_spend_update_writer import DBSpendUpdateWriter
from litellm.proxy.utils import ProxyUpdateSpend

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_proxy_server(general_settings: dict) -> MagicMock:
    """Return a mock proxy_server module with the given general_settings."""
    mock = MagicMock()
    mock.general_settings = general_settings
    return mock


def _make_writer_with_mock_queues() -> DBSpendUpdateWriter:
    writer = DBSpendUpdateWriter.__new__(DBSpendUpdateWriter)
    writer.spend_update_queue = AsyncMock()
    writer.spend_update_queue.add_update = AsyncMock()
    writer.daily_spend_update_queue = AsyncMock()
    writer.daily_team_spend_update_queue = AsyncMock()
    writer.daily_end_user_spend_update_queue = AsyncMock()
    writer.daily_agent_spend_update_queue = AsyncMock()
    writer.daily_org_spend_update_queue = AsyncMock()
    writer.daily_tag_spend_update_queue = AsyncMock()
    writer.tool_discovery_queue = MagicMock()
    writer.redis_update_buffer = MagicMock()
    writer.pod_lock_manager = MagicMock()
    return writer


def _make_minimal_payload() -> dict:
    return {
        "startTime": "2024-01-01T00:00:00",
        "endTime": "2024-01-01T00:00:01",
        "user": "test-user",
        "api_key": "hashed-key",
        "model": "gpt-4",
        "model_group": "gpt-4",
        "custom_llm_provider": "openai",
        "spend": 0.001,
        "metadata": "{}",
        "request_tags": "[]",
        "team_id": "team-1",
        "organization_id": "org-1",
        "end_user_id": "end-user-1",
        "agent_id": None,
    }


_DAILY_HELPERS = [
    "add_spend_log_transaction_to_daily_user_transaction",
    "add_spend_log_transaction_to_daily_team_transaction",
    "add_spend_log_transaction_to_daily_org_transaction",
    "add_spend_log_transaction_to_daily_end_user_transaction",
    "add_spend_log_transaction_to_daily_agent_transaction",
    "add_spend_log_transaction_to_daily_tag_transaction",
]

_BALANCE_HELPERS = [
    "_update_user_db",
    "_update_key_db",
    "_update_team_db",
    "_update_org_db",
    "_update_tag_db",
    "_update_agent_db",
]


# ---------------------------------------------------------------------------
# ProxyUpdateSpend.disable_daily_spend_aggregation() unit tests
# ---------------------------------------------------------------------------


def test_disable_daily_spend_aggregation_returns_false_by_default():
    """Flag absent from general_settings → returns False (aggregation enabled)."""
    mock_ps = _mock_proxy_server({})
    with patch.dict(sys.modules, {"litellm.proxy.proxy_server": mock_ps}):
        result = ProxyUpdateSpend.disable_daily_spend_aggregation()
    assert result is False


def test_disable_daily_spend_aggregation_returns_true_when_set():
    """Flag set to True in general_settings → returns True (aggregation disabled)."""
    mock_ps = _mock_proxy_server({"disable_daily_spend_aggregation": True})
    with patch.dict(sys.modules, {"litellm.proxy.proxy_server": mock_ps}):
        result = ProxyUpdateSpend.disable_daily_spend_aggregation()
    assert result is True


def test_disable_daily_spend_aggregation_false_when_explicitly_false():
    """Flag explicitly False → returns False (aggregation enabled)."""
    mock_ps = _mock_proxy_server({"disable_daily_spend_aggregation": False})
    with patch.dict(sys.modules, {"litellm.proxy.proxy_server": mock_ps}):
        result = ProxyUpdateSpend.disable_daily_spend_aggregation()
    assert result is False


# ---------------------------------------------------------------------------
# _enqueue_daily_spend_updates: the core gate on daily aggregation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_enqueue_daily_spend_updates_calls_all_helpers_when_enabled():
    """When flag is False/absent, all 6 daily helpers are awaited."""
    writer = _make_writer_with_mock_queues()
    payload = _make_minimal_payload()

    for attr in _DAILY_HELPERS:
        setattr(writer, attr, AsyncMock(return_value=None))

    mock_ps = _mock_proxy_server({"disable_daily_spend_aggregation": False})
    with patch.dict(sys.modules, {"litellm.proxy.proxy_server": mock_ps}):
        await writer._enqueue_daily_spend_updates(
            payload_copy=payload,
            org_id="org-1",
            prisma_client=MagicMock(),
        )

    for attr in _DAILY_HELPERS:
        getattr(writer, attr).assert_awaited_once()


@pytest.mark.asyncio
async def test_enqueue_daily_spend_updates_skips_all_helpers_when_disabled():
    """When disable_daily_spend_aggregation=True, no daily helper is awaited."""
    writer = _make_writer_with_mock_queues()
    payload = _make_minimal_payload()

    for attr in _DAILY_HELPERS:
        setattr(writer, attr, AsyncMock(return_value=None))

    mock_ps = _mock_proxy_server({"disable_daily_spend_aggregation": True})
    with patch.dict(sys.modules, {"litellm.proxy.proxy_server": mock_ps}):
        await writer._enqueue_daily_spend_updates(
            payload_copy=payload,
            org_id="org-1",
            prisma_client=MagicMock(),
        )

    for attr in _DAILY_HELPERS:
        getattr(writer, attr).assert_not_awaited()


@pytest.mark.asyncio
async def test_enqueue_daily_spend_updates_default_enables_all():
    """When general_settings is empty (no flag), all 6 helpers still fire."""
    writer = _make_writer_with_mock_queues()
    payload = _make_minimal_payload()

    for attr in _DAILY_HELPERS:
        setattr(writer, attr, AsyncMock(return_value=None))

    mock_ps = _mock_proxy_server({})
    with patch.dict(sys.modules, {"litellm.proxy.proxy_server": mock_ps}):
        await writer._enqueue_daily_spend_updates(
            payload_copy=payload,
            org_id="org-1",
            prisma_client=MagicMock(),
        )

    for attr in _DAILY_HELPERS:
        getattr(writer, attr).assert_awaited_once()


# ---------------------------------------------------------------------------
# _batch_database_updates: balance updates must still run when disabled
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_batch_database_updates_balance_updates_run_when_daily_disabled():
    """
    Even with disable_daily_spend_aggregation=True, key/user/team balance
    update helpers must still be called.
    """
    writer = _make_writer_with_mock_queues()
    payload = _make_minimal_payload()

    for attr in _BALANCE_HELPERS:
        setattr(writer, attr, AsyncMock(return_value=None))

    # Stub _enqueue_daily_spend_updates so it records whether it was called
    writer._enqueue_daily_spend_updates = AsyncMock(return_value=None)

    mock_ps = _mock_proxy_server({"disable_daily_spend_aggregation": True})
    with patch.dict(sys.modules, {"litellm.proxy.proxy_server": mock_ps}):
        await writer._batch_database_updates(
            response_cost=0.005,
            user_id="test-user",
            hashed_token="hashed-key",
            team_id="team-1",
            org_id="org-1",
            end_user_id="end-user-1",
            prisma_client=MagicMock(),
            user_api_key_cache=MagicMock(),
            litellm_proxy_budget_name=None,
            payload_copy=payload,
            request_tags=None,
        )

    writer._update_key_db.assert_awaited_once()
    writer._update_user_db.assert_awaited_once()
    writer._update_team_db.assert_awaited_once()
