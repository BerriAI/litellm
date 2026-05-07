"""Unit tests for `GET /v2/sessions/:id` (LIT-2923).

Approach:
- Build a minimal FastAPI app with only `sessions.router` mounted (Wave 3
  owns proxy_server registration; the test should not depend on it).
- Stub `litellm.proxy.proxy_server.prisma_client` with a MagicMock whose
  `db.litellm_managedagentsession.find_first` returns a configurable row.
- Override `user_api_key_auth` via `app.dependency_overrides` to simulate
  different callers (and test the no-auth path with no override).

Contract-critical assertions:
- Response NEVER contains `sandbox_url` or `sandbox_metadata` — these are
  internal-only (contract §6.2 / §6.3).
- A session belonging to a different `created_by` returns 404, not 403,
  to avoid leaking session existence.
"""

import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath("../../.."))

from litellm.managed_agents.endpoints import sessions as sessions_module
from litellm.proxy._types import CommonProxyErrors, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_session_row(
    *,
    session_id: str = "ses_abc",
    agent_id: str = "agt_xyz",
    created_by: Optional[str] = "user_a",
    status: str = "ready",
    sandbox_type: str = "opencode",
    sandbox_size: str = "small",
    sandbox_timeout_minutes: int = 60,
    sandbox_idle_timeout_minutes: int = 10,
    sandbox_image: Optional[str] = "litellm/opencode:latest",
    sandbox_url: Optional[str] = "http://127.0.0.1:1234",
    sandbox_metadata: Optional[Dict[str, Any]] = None,
    repos: Optional[list] = None,
    env_vars: Optional[Dict[str, Any]] = None,
    terminated_at: Optional[datetime] = None,
) -> MagicMock:
    """Return a MagicMock row that mirrors a Prisma row shape for the
    LiteLLM_ManagedAgentSession table."""
    if sandbox_metadata is None:
        sandbox_metadata = {"opencode_session_id": "oc_sid_xxx"}
    if repos is None:
        repos = []
    if env_vars is None:
        env_vars = {}

    now = datetime.now(timezone.utc)
    row = MagicMock()
    # Make `model_dump()` return the dict the helper actually consumes.
    row.model_dump.return_value = {
        "id": session_id,
        "agent_id": agent_id,
        "sandbox_type": sandbox_type,
        "sandbox_size": sandbox_size,
        "sandbox_timeout_minutes": sandbox_timeout_minutes,
        "sandbox_idle_timeout_minutes": sandbox_idle_timeout_minutes,
        "sandbox_image": sandbox_image,
        "sandbox_url": sandbox_url,
        "sandbox_metadata": sandbox_metadata,
        "status": status,
        "repos": repos,
        "env_vars": env_vars,
        "created_by": created_by,
        "created_at": now,
        "updated_at": now,
        "terminated_at": terminated_at,
    }
    return row


def _make_prisma(find_first_return: Any = None) -> MagicMock:
    """Build a MagicMock prisma_client with `db.litellm_managedagentsession.find_first`."""
    prisma = MagicMock()
    prisma.db = MagicMock()
    prisma.db.litellm_managedagentsession = MagicMock()
    prisma.db.litellm_managedagentsession.find_first = AsyncMock(
        return_value=find_first_return
    )
    return prisma


def _build_app(auth: Optional[UserAPIKeyAuth] = None) -> FastAPI:
    """Construct a minimal app with the sessions router mounted.

    If `auth` is provided, override the auth dependency to return it. If
    omitted, the real `user_api_key_auth` runs (used to test the 401 path).
    """
    app = FastAPI()
    app.include_router(sessions_module.router)
    if auth is not None:
        app.dependency_overrides[user_api_key_auth] = lambda: auth
    return app


def _user_auth(user_id: str = "user_a") -> UserAPIKeyAuth:
    return UserAPIKeyAuth(api_key="sk-test", user_id=user_id)


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_get_session_returns_row_for_owner(monkeypatch):
    """200 with full SessionRow shape when the session belongs to the caller.

    Contract-critical: response must NOT include `sandbox_url` or
    `sandbox_metadata` — these are internal-only.
    """
    row = _make_session_row(session_id="ses_abc", created_by="user_a")
    prisma = _make_prisma(find_first_return=row)

    import litellm.proxy.proxy_server as ps

    monkeypatch.setattr(ps, "prisma_client", prisma)

    app = _build_app(auth=_user_auth("user_a"))
    client = TestClient(app)

    resp = client.get("/v2/sessions/ses_abc")
    assert resp.status_code == 200, resp.text
    body = resp.json()

    # Public fields present.
    assert body["id"] == "ses_abc"
    assert body["agent_id"] == "agt_xyz"
    assert body["status"] == "ready"
    assert body["created_by"] == "user_a"
    assert body["terminated_at"] is None

    # Sandbox sub-object correctly reassembled from flat columns.
    assert body["sandbox"]["type"] == "opencode"
    assert body["sandbox"]["size"] == "small"
    assert body["sandbox"]["timeout_minutes"] == 60
    assert body["sandbox"]["idle_timeout_minutes"] == 10
    assert body["sandbox"]["image"] == "litellm/opencode:latest"

    # Internal-only fields MUST be stripped.
    assert "sandbox_url" not in body, "sandbox_url must NOT be returned"
    assert "sandbox_metadata" not in body, "sandbox_metadata must NOT be returned"
    # Belt-and-suspenders: also not nested under `sandbox`.
    assert "sandbox_url" not in body["sandbox"]
    assert "sandbox_metadata" not in body["sandbox"]

    # The DB lookup was scoped by created_by.
    prisma.db.litellm_managedagentsession.find_first.assert_awaited_once_with(
        where={"id": "ses_abc", "created_by": "user_a"}
    )


def test_get_session_returns_repos_array(monkeypatch):
    """`repos` from the row is mapped onto the response."""
    row = _make_session_row(
        session_id="ses_with_repos",
        created_by="user_a",
        repos=[
            {
                "url": "https://github.com/org/repo-a",
                "starting_ref": "main",
                "checked_out_sha": "abc123",
            }
        ],
    )
    prisma = _make_prisma(find_first_return=row)

    import litellm.proxy.proxy_server as ps

    monkeypatch.setattr(ps, "prisma_client", prisma)

    app = _build_app(auth=_user_auth("user_a"))
    client = TestClient(app)

    resp = client.get("/v2/sessions/ses_with_repos")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert len(body["repos"]) == 1
    assert body["repos"][0]["url"] == "https://github.com/org/repo-a"
    assert body["repos"][0]["starting_ref"] == "main"
    assert body["repos"][0]["checked_out_sha"] == "abc123"


# ---------------------------------------------------------------------------
# 404 paths
# ---------------------------------------------------------------------------


def test_get_session_returns_404_when_missing(monkeypatch):
    """If the session does not exist for ANY caller, return 404."""
    prisma = _make_prisma(find_first_return=None)

    import litellm.proxy.proxy_server as ps

    monkeypatch.setattr(ps, "prisma_client", prisma)

    app = _build_app(auth=_user_auth("user_a"))
    client = TestClient(app)

    resp = client.get("/v2/sessions/ses_does_not_exist")
    assert resp.status_code == 404
    assert resp.json()["detail"] == "Session ses_does_not_exist not found"


def test_get_session_returns_404_for_other_users_session(monkeypatch):
    """Session exists but belongs to another user → 404, NOT 403.

    The DB query is scoped by `created_by` — so when caller=user_b queries
    a session owned by user_a, the row is filtered out at the DB level and
    `find_first` returns None. The handler returns 404 to avoid leaking
    existence (contract §5).
    """
    # Simulate: row exists for user_a, but find_first(where={created_by=user_b})
    # returns None (Prisma scoping behavior).
    prisma = _make_prisma(find_first_return=None)

    import litellm.proxy.proxy_server as ps

    monkeypatch.setattr(ps, "prisma_client", prisma)

    # Caller is user_b but the session row is owned by user_a.
    app = _build_app(auth=_user_auth("user_b"))
    client = TestClient(app)

    resp = client.get("/v2/sessions/ses_owned_by_user_a")
    assert resp.status_code == 404, "Must 404 (not 403) to avoid leaking existence"
    assert resp.json()["detail"] == "Session ses_owned_by_user_a not found"

    # Verify the scoping: query passed user_b, not user_a.
    prisma.db.litellm_managedagentsession.find_first.assert_awaited_once_with(
        where={"id": "ses_owned_by_user_a", "created_by": "user_b"}
    )


# ---------------------------------------------------------------------------
# Auth + DB-not-connected
# ---------------------------------------------------------------------------


def test_get_session_no_auth_returns_401():
    """When `user_api_key_auth` rejects (no/invalid bearer token) → 401.

    We can't run the real `user_api_key_auth` in a unit test (it needs a
    full proxy DB + master-key bootstrap), so we install a dependency
    override that simulates the auth-failure path: raise HTTPException(401).
    The handler should never be reached — FastAPI short-circuits at the
    auth dep.
    """
    from fastapi import HTTPException

    def _reject_auth() -> UserAPIKeyAuth:
        raise HTTPException(
            status_code=401, detail="Authentication Error, Invalid proxy server token"
        )

    app = FastAPI()
    app.include_router(sessions_module.router)
    app.dependency_overrides[user_api_key_auth] = _reject_auth

    client = TestClient(app)
    resp = client.get("/v2/sessions/ses_anything")
    assert resp.status_code == 401, resp.text


def test_get_session_returns_500_when_db_not_connected(monkeypatch):
    """If `prisma_client` is None, return 500 with the canonical error."""
    import litellm.proxy.proxy_server as ps

    monkeypatch.setattr(ps, "prisma_client", None)

    app = _build_app(auth=_user_auth("user_a"))
    client = TestClient(app)

    resp = client.get("/v2/sessions/ses_anything")
    assert resp.status_code == 500
    detail = resp.json()["detail"]
    assert detail["error"] == CommonProxyErrors.db_not_connected_error.value


# ---------------------------------------------------------------------------
# Sanity: status passthrough
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("status", ["provisioning", "ready", "terminated", "error"])
def test_get_session_passes_through_all_statuses(monkeypatch, status):
    """The handler does NOT filter by status — all four enum values pass through.

    (Per contract §6.3 the read endpoint just returns the row; status-based
    fail-closed behavior lives on the messaging endpoints, not here.)
    """
    row = _make_session_row(
        session_id="ses_status_test", created_by="user_a", status=status
    )
    prisma = _make_prisma(find_first_return=row)

    import litellm.proxy.proxy_server as ps

    monkeypatch.setattr(ps, "prisma_client", prisma)

    app = _build_app(auth=_user_auth("user_a"))
    client = TestClient(app)

    resp = client.get("/v2/sessions/ses_status_test")
    assert resp.status_code == 200, resp.text
    assert resp.json()["status"] == status
