"""
Regression tests for LIT-3362: /budget/update must recompute
budget_reset_at when budget_duration changes and the caller did
not pass an explicit budget_reset_at.

Before the fix, callers updating only budget_duration would leave the
existing budget_reset_at in place. Shortening a budget therefore did
not bring its reset forward, so the row would not reset until the prior
(longer) schedule fired \u2014 days or weeks after the new duration was in
effect.

This module covers four cases via the production FastAPI app + a mocked
Prisma client (matches the existing test pattern in
test_budget_endpoints.py):

1. budget_duration in payload, no budget_reset_at \u2192 recompute.
2. budget_duration AND explicit budget_reset_at \u2192 caller wins.
3. budget_duration NOT in payload \u2192 budget_reset_at untouched.
4. budget_duration explicitly None in payload \u2192 do not recompute.
"""

import os
import sys
import types
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, MagicMock

import litellm.proxy.proxy_server as ps
from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.proxy_server import app

sys.path.insert(0, os.path.abspath("../../../"))


@pytest.fixture
def client_and_table(monkeypatch):
    """Build a TestClient against the real proxy app with a mocked Prisma."""
    mock_prisma = MagicMock()
    mock_table = MagicMock()

    captured = {"create": [], "update": []}

    def capture_create(*, data):
        captured["create"].append(data)
        return data

    def capture_update(*, where, data):
        captured["update"].append({"where": where, "data": data})
        return {**where, **data}

    mock_table.create = AsyncMock(side_effect=capture_create)
    mock_table.update = AsyncMock(side_effect=capture_update)

    mock_prisma.db = types.SimpleNamespace(
        litellm_budgettable=mock_table,
        litellm_dailyspend=mock_table,
    )
    monkeypatch.setattr(ps, "prisma_client", mock_prisma)

    fake_user = UserAPIKeyAuth(
        user_id="tester",
        user_role=LitellmUserRoles.PROXY_ADMIN,
    )
    app.dependency_overrides[ps.user_api_key_auth] = lambda: fake_user

    yield TestClient(app), captured

    app.dependency_overrides.clear()


def _last_update_payload(captured):
    assert captured["update"], "expected at least one /budget/update call"
    return captured["update"][-1]["data"]


def test_update_with_duration_only_recomputes_reset_at(client_and_table):
    """budget_duration changes \u2192 reset_at is recomputed forward."""
    client, captured = client_and_table

    before = datetime.now(timezone.utc)
    resp = client.post(
        "/budget/update",
        json={"budget_id": "b1", "budget_duration": "1d"},
    )
    assert resp.status_code == 200, resp.text

    data = _last_update_payload(captured)
    assert data["budget_duration"] == "1d"
    assert "budget_reset_at" in data, "fix must set budget_reset_at"

    reset_at = data["budget_reset_at"]
    if isinstance(reset_at, str):
        reset_at = datetime.fromisoformat(reset_at.replace("Z", "+00:00"))

    # New reset must be ~1 day from now (allow generous slack for the
    # standardized day-boundary timezone logic).
    delta = reset_at - before
    assert timedelta(0) < delta <= timedelta(days=2), (
        f"budget_reset_at={reset_at} not within ~1d of now ({before})"
    )


def test_update_with_explicit_reset_at_preserves_caller_value(client_and_table):
    """Explicit budget_reset_at wins \u2014 we never silently overwrite it."""
    client, captured = client_and_table

    explicit_reset = "2099-01-01T00:00:00+00:00"
    resp = client.post(
        "/budget/update",
        json={
            "budget_id": "b2",
            "budget_duration": "1h",
            "budget_reset_at": explicit_reset,
        },
    )
    assert resp.status_code == 200, resp.text

    data = _last_update_payload(captured)
    assert data["budget_duration"] == "1h"
    # Pydantic may parse the string into a datetime \u2014 normalize either form.
    stored = data["budget_reset_at"]
    if hasattr(stored, "isoformat"):
        stored = stored.isoformat()
    assert "2099-01-01" in str(stored), (
        f"explicit budget_reset_at must be preserved, got {stored!r}"
    )


def test_update_without_budget_duration_does_not_touch_reset_at(client_and_table):
    """Updates that omit budget_duration must not introduce budget_reset_at."""
    client, captured = client_and_table

    resp = client.post(
        "/budget/update",
        json={"budget_id": "b3", "max_budget": 50.0},
    )
    assert resp.status_code == 200, resp.text

    data = _last_update_payload(captured)
    assert "budget_reset_at" not in data, (
        "must not auto-set reset_at when caller did not touch budget_duration"
    )
    assert "budget_duration" not in data
    assert data["max_budget"] == 50.0


def test_update_with_budget_duration_explicit_none_does_not_recompute(client_and_table):
    """Setting budget_duration=None (clearing it) must not recompute reset_at."""
    client, captured = client_and_table

    resp = client.post(
        "/budget/update",
        json={"budget_id": "b4", "budget_duration": None},
    )
    assert resp.status_code == 200, resp.text

    data = _last_update_payload(captured)
    # exclude_unset preserves explicitly-set None, so the field IS sent
    assert "budget_duration" in data and data["budget_duration"] is None
    # but reset_at must not be auto-computed against a None duration
    assert "budget_reset_at" not in data
