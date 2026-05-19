"""Tests for the webhook delivery dispatcher (S6-05) + event emission (S6-06)."""

import hashlib
import hmac
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from litellm.proxy.webhook_endpoints import dispatcher as disp_mod
from litellm.proxy.webhook_endpoints.dispatcher import (
    _matches_filters,
    _sign,
    dispatch_to_subscription,
    emit_event,
)


def _mock_subscription(**overrides):
    base = dict(
        subscription_id="sub-1",
        target_url="https://hooks.example.com/x",
        secret_hash="abc-hash-deadbeef",
        filters=None,
        consecutive_failures=0,
        events=["capability.invoked"],
        is_active=True,
    )
    base.update(overrides)
    m = MagicMock()
    for k, v in base.items():
        setattr(m, k, v)
    m.model_dump = MagicMock(return_value=base)
    return m


def _mock_prisma():
    p = MagicMock()
    t = p.db.litellm_webhooksubscriptiontable
    t.find_many = AsyncMock(return_value=[])
    t.find_unique = AsyncMock(return_value=None)
    t.update = AsyncMock()
    p.db.litellm_webhookdlq.create = AsyncMock()
    return p


@pytest.fixture(autouse=True)
def _patch_backoff(monkeypatch):
    """Shrink the backoff schedule so tests don't sleep for 12.5 minutes."""
    monkeypatch.setattr(disp_mod, "_BACKOFF_SECONDS", [0, 0, 0, 0, 0])


# ---------------------------------------------------------------------------
# Pure-function checks
# ---------------------------------------------------------------------------


def test_sign_produces_sha256_hmac_hex():
    secret = "topsecret"
    body = b'{"event":"x"}'
    expected = "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    assert _sign(secret, body) == expected


def test_filters_no_op_when_none():
    assert _matches_filters(None, {"data": {"x": 1}}) is True
    assert _matches_filters({}, {"data": {"x": 1}}) is True


def test_filters_and_must_all_match():
    envelope = {"data": {"app_id": "xct-chat", "entity_type": "agent"}}
    assert _matches_filters({"app_id": "xct-chat"}, envelope) is True
    assert _matches_filters({"app_id": "xct-home"}, envelope) is False
    assert (
        _matches_filters({"app_id": "xct-chat", "entity_type": "model"}, envelope)
        is False
    )


# ---------------------------------------------------------------------------
# dispatch_to_subscription happy + retry + DLQ
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dispatch_delivers_on_first_try_and_persists_success():
    prisma = _mock_prisma()
    sub = _mock_subscription()
    sent = []

    class _OKClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def post(self, url, content, headers):
            sent.append((url, content, headers))
            resp = MagicMock()
            resp.status_code = 200
            resp.text = ""
            return resp

    with (
        patch("litellm.proxy.proxy_server.prisma_client", prisma),
        patch.object(disp_mod.httpx, "AsyncClient", lambda **kw: _OKClient()),
    ):
        result = await dispatch_to_subscription(
            subscription=sub,
            event_type="capability.invoked",
            payload={"app_id": "xct-chat"},
        )
    assert result["delivered"] is True
    assert result["attempts"] == 1
    # Success path: update was called with last_success_at + reset failures
    update_data = prisma.db.litellm_webhooksubscriptiontable.update.call_args.kwargs[
        "data"
    ]
    assert update_data["consecutive_failures"] == 0
    assert "last_success_at" in update_data
    # Outgoing request: signed, JSON, with our headers.
    url, body, hdrs = sent[0]
    assert url == "https://hooks.example.com/x"
    assert hdrs["X-XCT-Event"] == "capability.invoked"
    assert hdrs["X-XCT-Signature"].startswith("sha256=")
    parsed = json.loads(body)
    assert parsed["event"] == "capability.invoked"
    assert parsed["data"] == {"app_id": "xct-chat"}


@pytest.mark.asyncio
async def test_dispatch_retries_then_dlqs_on_repeated_failure():
    prisma = _mock_prisma()
    prisma.db.litellm_webhooksubscriptiontable.find_unique.return_value = (
        _mock_subscription(consecutive_failures=5)
    )
    sub = _mock_subscription(consecutive_failures=5)

    class _FailClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def post(self, url, content, headers):
            resp = MagicMock()
            resp.status_code = 502
            resp.text = "Bad Gateway"
            return resp

    with (
        patch("litellm.proxy.proxy_server.prisma_client", prisma),
        patch.object(disp_mod.httpx, "AsyncClient", lambda **kw: _FailClient()),
    ):
        result = await dispatch_to_subscription(
            subscription=sub,
            event_type="capability.invoked",
            payload={"x": 1},
        )

    assert result["delivered"] is False
    assert result["attempts"] == 5  # exhausted all 5 attempts
    # DLQ row created exactly once
    prisma.db.litellm_webhookdlq.create.assert_awaited_once()
    dlq_data = prisma.db.litellm_webhookdlq.create.call_args.kwargs["data"]
    assert dlq_data["subscription_id"] == "sub-1"
    assert dlq_data["event_type"] == "capability.invoked"
    assert dlq_data["attempts"] == 5
    # Subscription update: incremented failure count
    update = prisma.db.litellm_webhooksubscriptiontable.update.call_args.kwargs["data"]
    assert update["consecutive_failures"] == 6


@pytest.mark.asyncio
async def test_dispatch_auto_disables_after_threshold():
    prisma = _mock_prisma()
    prisma.db.litellm_webhooksubscriptiontable.find_unique.return_value = (
        _mock_subscription(consecutive_failures=19)
    )
    sub = _mock_subscription(consecutive_failures=19)

    class _FailClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def post(self, url, content, headers):
            resp = MagicMock()
            resp.status_code = 500
            resp.text = ""
            return resp

    with (
        patch("litellm.proxy.proxy_server.prisma_client", prisma),
        patch.object(disp_mod.httpx, "AsyncClient", lambda **kw: _FailClient()),
    ):
        await dispatch_to_subscription(
            subscription=sub,
            event_type="capability.invoked",
            payload={"x": 1},
            max_attempts=1,
        )

    update = prisma.db.litellm_webhooksubscriptiontable.update.call_args.kwargs["data"]
    assert update["consecutive_failures"] == 20
    assert update["is_active"] is False


# ---------------------------------------------------------------------------
# emit_event fan-out
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_emit_event_fans_out_to_active_subscribers():
    prisma = _mock_prisma()
    prisma.db.litellm_webhooksubscriptiontable.find_many.return_value = [
        _mock_subscription(subscription_id="a", filters=None),
        _mock_subscription(
            subscription_id="b", filters={"app_id": "xct-home"}
        ),  # filter mismatch
        _mock_subscription(subscription_id="c", filters={"app_id": "xct-chat"}),
    ]
    dispatched = []

    async def fake_dispatch(*, subscription, event_type, payload):
        dispatched.append(subscription.subscription_id)
        return {"delivered": True}

    with (
        patch("litellm.proxy.proxy_server.prisma_client", prisma),
        patch.object(disp_mod, "dispatch_to_subscription", fake_dispatch),
    ):
        await emit_event(
            "capability.invoked",
            {"app_id": "xct-chat", "spend": 0.01},
            app_id="xct-chat",
        )
        # Tasks are created with asyncio.create_task; give the loop a tick.
        import asyncio

        for _ in range(3):
            await asyncio.sleep(0)

    # 'a' (no filter) + 'c' (filter matches); 'b' filtered out.
    assert set(dispatched) == {"a", "c"}
