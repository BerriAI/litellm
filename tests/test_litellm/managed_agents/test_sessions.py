"""Unit tests for `GET /v2/sessions/:id` (LIT-2923) and
`GET /v2/agents/:agent_id/sessions`.

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
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath("../../.."))

from litellm.proxy.managed_agents_endpoints import sessions as sessions_module
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


# ---------------------------------------------------------------------------
# GET /v2/agents/:agent_id/sessions — list sessions for an agent
# ---------------------------------------------------------------------------


def _make_session_dict(
    *,
    session_id: str,
    agent_id: str,
    created_by: str,
    created_at: datetime,
    status: str = "ready",
) -> Dict[str, Any]:
    """Plain-dict session row, suitable for the in-memory `_FakeSessionTable`.

    Mirrors the columns the handler reads via `_row_to_session_response`.
    """
    return {
        "id": session_id,
        "agent_id": agent_id,
        "sandbox_type": "opencode",
        "sandbox_size": "small",
        "sandbox_timeout_minutes": 60,
        "sandbox_idle_timeout_minutes": 10,
        "sandbox_image": "litellm/opencode:latest",
        "sandbox_url": "http://127.0.0.1:1234",
        "sandbox_metadata": {"opencode_session_id": "oc_sid_xxx"},
        "status": status,
        "repos": [],
        "env_vars": {},
        "created_by": created_by,
        "created_at": created_at,
        "updated_at": created_at,
        "terminated_at": None,
    }


def _make_agent_dict(
    *,
    agent_id: str,
    created_by: str,
) -> Dict[str, Any]:
    now = datetime.now(timezone.utc)
    return {
        "id": agent_id,
        "name": f"agent-for-{agent_id}",
        "config": {
            "model": "anthropic/claude-opus-4",
            "system_prompt": "test",
            "tools": ["read"],
            "litellm_api_key": "sk-test",
            "litellm_base_url": "http://localhost:4000",
        },
        "created_by": created_by,
        "created_at": now,
        "updated_at": now,
    }


class _FakeAgentSessionDB:
    """In-memory stub for both `litellm_managedagent` and
    `litellm_managedagentsession` with the methods the handlers call.

    Built ad-hoc per test so we can seed agents and sessions independently.
    """

    def __init__(self) -> None:
        self.agents: Dict[str, Dict[str, Any]] = {}
        self.sessions: Dict[str, Dict[str, Any]] = {}

    # --- prisma-shaped access ---
    def as_prisma(self) -> MagicMock:
        agent_table = MagicMock()
        agent_table.find_first = AsyncMock(side_effect=self._agent_find_first)

        session_table = MagicMock()
        session_table.find_first = AsyncMock(side_effect=self._session_find_first)
        session_table.find_many = AsyncMock(side_effect=self._session_find_many)

        prisma = MagicMock()
        prisma.db = MagicMock()
        prisma.db.litellm_managedagent = agent_table
        prisma.db.litellm_managedagentsession = session_table
        return prisma

    # --- agent ---
    async def _agent_find_first(self, *, where: Dict[str, Any]) -> Optional[MagicMock]:
        for row in self.agents.values():
            if all(row.get(k) == v for k, v in where.items()):
                m = MagicMock()
                m.model_dump.return_value = dict(row)
                return m
        return None

    # --- session ---
    async def _session_find_first(
        self, *, where: Dict[str, Any]
    ) -> Optional[MagicMock]:
        for row in self.sessions.values():
            if all(row.get(k) == v for k, v in where.items()):
                m = MagicMock()
                m.model_dump.return_value = dict(row)
                return m
        return None

    async def _session_find_many(
        self,
        *,
        where: Dict[str, Any],
        take: int,
        skip: int,
        order: Dict[str, str],
    ) -> List[MagicMock]:
        matched = [
            row
            for row in self.sessions.values()
            if all(row.get(k) == v for k, v in where.items())
        ]
        order_key, order_dir = next(iter(order.items()))
        matched.sort(
            key=lambda r: r.get(order_key) or datetime.min,
            reverse=(order_dir == "desc"),
        )
        page = matched[skip : skip + take]
        out = []
        for row in page:
            m = MagicMock()
            m.model_dump.return_value = dict(row)
            out.append(m)
        return out


def test_list_sessions_for_agent_returns_only_that_agent(monkeypatch):
    """Sessions belonging to another agent must NOT leak into the response."""
    db = _FakeAgentSessionDB()
    db.agents["agt_a"] = _make_agent_dict(agent_id="agt_a", created_by="user_a")
    db.agents["agt_b"] = _make_agent_dict(agent_id="agt_b", created_by="user_a")

    base = datetime(2026, 5, 1, tzinfo=timezone.utc)
    db.sessions["ses_a1"] = _make_session_dict(
        session_id="ses_a1",
        agent_id="agt_a",
        created_by="user_a",
        created_at=base + timedelta(seconds=1),
    )
    db.sessions["ses_a2"] = _make_session_dict(
        session_id="ses_a2",
        agent_id="agt_a",
        created_by="user_a",
        created_at=base + timedelta(seconds=2),
    )
    db.sessions["ses_b1"] = _make_session_dict(
        session_id="ses_b1",
        agent_id="agt_b",
        created_by="user_a",
        created_at=base + timedelta(seconds=3),
    )

    import litellm.proxy.proxy_server as ps

    monkeypatch.setattr(ps, "prisma_client", db.as_prisma())

    app = _build_app(auth=_user_auth("user_a"))
    client = TestClient(app)

    resp = client.get("/v2/agents/agt_a/sessions")
    assert resp.status_code == 200, resp.text
    body = resp.json()

    assert len(body["data"]) == 2
    returned_ids = {row["id"] for row in body["data"]}
    assert returned_ids == {"ses_a1", "ses_a2"}
    assert "ses_b1" not in returned_ids
    # Every row's agent_id is the queried agent.
    for row in body["data"]:
        assert row["agent_id"] == "agt_a"


def test_list_sessions_for_agent_unknown_agent_returns_404(monkeypatch):
    """Unknown agent (not present for this caller) → 404 before any session lookup."""
    db = _FakeAgentSessionDB()
    # No agents seeded.

    import litellm.proxy.proxy_server as ps

    monkeypatch.setattr(ps, "prisma_client", db.as_prisma())

    app = _build_app(auth=_user_auth("user_a"))
    client = TestClient(app)

    resp = client.get("/v2/agents/agt_does_not_exist/sessions")
    assert resp.status_code == 404
    assert resp.json()["detail"] == "Agent agt_does_not_exist not found"


def test_list_sessions_for_agent_other_users_agent_returns_404(monkeypatch):
    """Agent belongs to another caller → 404, not 403."""
    db = _FakeAgentSessionDB()
    db.agents["agt_a"] = _make_agent_dict(agent_id="agt_a", created_by="user_a")

    import litellm.proxy.proxy_server as ps

    monkeypatch.setattr(ps, "prisma_client", db.as_prisma())

    app = _build_app(auth=_user_auth("user_b"))
    client = TestClient(app)

    resp = client.get("/v2/agents/agt_a/sessions")
    assert resp.status_code == 404
    assert resp.json()["detail"] == "Agent agt_a not found"


def test_list_sessions_for_agent_scopes_by_caller(monkeypatch):
    """Caller must not see another user's sessions even for an agent they own.

    (Defense-in-depth: the agent ownership check already blocks cross-user
    access, but the session query is independently scoped by created_by.)
    """
    db = _FakeAgentSessionDB()
    db.agents["agt_shared"] = _make_agent_dict(
        agent_id="agt_shared", created_by="user_a"
    )

    base = datetime(2026, 5, 1, tzinfo=timezone.utc)
    # Two sessions on agt_shared, owned by user_a (the caller).
    db.sessions["ses_a1"] = _make_session_dict(
        session_id="ses_a1",
        agent_id="agt_shared",
        created_by="user_a",
        created_at=base + timedelta(seconds=1),
    )
    # A session on the same agent_id but with created_by=user_other —
    # should never leak (bookkeeping anomaly that scoping must catch).
    db.sessions["ses_other"] = _make_session_dict(
        session_id="ses_other",
        agent_id="agt_shared",
        created_by="user_other",
        created_at=base + timedelta(seconds=99),
    )

    import litellm.proxy.proxy_server as ps

    monkeypatch.setattr(ps, "prisma_client", db.as_prisma())

    app = _build_app(auth=_user_auth("user_a"))
    client = TestClient(app)

    resp = client.get("/v2/agents/agt_shared/sessions")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    returned_ids = {row["id"] for row in body["data"]}
    assert returned_ids == {"ses_a1"}
    assert "ses_other" not in returned_ids


def test_list_sessions_for_agent_strips_internal_fields(monkeypatch):
    """`sandbox_url` and `sandbox_metadata` MUST NOT appear in the listing."""
    db = _FakeAgentSessionDB()
    db.agents["agt_a"] = _make_agent_dict(agent_id="agt_a", created_by="user_a")
    db.sessions["ses_a1"] = _make_session_dict(
        session_id="ses_a1",
        agent_id="agt_a",
        created_by="user_a",
        created_at=datetime(2026, 5, 1, tzinfo=timezone.utc),
    )

    import litellm.proxy.proxy_server as ps

    monkeypatch.setattr(ps, "prisma_client", db.as_prisma())

    app = _build_app(auth=_user_auth("user_a"))
    client = TestClient(app)

    resp = client.get("/v2/agents/agt_a/sessions")
    assert resp.status_code == 200, resp.text
    # Internal-only state must be absent from the entire response body.
    assert "sandbox_url" not in resp.text
    assert "sandbox_metadata" not in resp.text
    assert "opencode_session_id" not in resp.text


def test_list_sessions_for_agent_pagination_walks_pages(monkeypatch):
    """Seed 4 sessions, page through with limit=2."""
    db = _FakeAgentSessionDB()
    db.agents["agt_a"] = _make_agent_dict(agent_id="agt_a", created_by="user_a")

    base = datetime(2026, 5, 1, tzinfo=timezone.utc)
    for i in range(4):
        db.sessions[f"ses_{i}"] = _make_session_dict(
            session_id=f"ses_{i}",
            agent_id="agt_a",
            created_by="user_a",
            created_at=base + timedelta(seconds=i),
        )

    import litellm.proxy.proxy_server as ps

    monkeypatch.setattr(ps, "prisma_client", db.as_prisma())

    app = _build_app(auth=_user_auth("user_a"))
    client = TestClient(app)

    resp1 = client.get("/v2/agents/agt_a/sessions?limit=2")
    assert resp1.status_code == 200, resp1.text
    page1 = resp1.json()
    assert len(page1["data"]) == 2
    assert page1["has_more"] is True
    assert page1["next_cursor"] == "2"

    resp2 = client.get(
        f"/v2/agents/agt_a/sessions?limit=2&cursor={page1['next_cursor']}"
    )
    assert resp2.status_code == 200, resp2.text
    page2 = resp2.json()
    assert len(page2["data"]) == 2
    assert page2["has_more"] is False
    assert page2["next_cursor"] is None

    # Pages cover all 4 records and don't overlap.
    page1_ids = [r["id"] for r in page1["data"]]
    page2_ids = [r["id"] for r in page2["data"]]
    all_seen = set(page1_ids) | set(page2_ids)
    assert len(all_seen) == 4
    assert set(page1_ids).isdisjoint(set(page2_ids))


def test_list_sessions_for_agent_filters_by_status(monkeypatch):
    db = _FakeAgentSessionDB()
    db.agents["agt_a"] = _make_agent_dict(agent_id="agt_a", created_by="user_a")

    base = datetime(2026, 5, 1, tzinfo=timezone.utc)
    db.sessions["ses_ready"] = _make_session_dict(
        session_id="ses_ready",
        agent_id="agt_a",
        created_by="user_a",
        created_at=base + timedelta(seconds=1),
        status="ready",
    )
    db.sessions["ses_term"] = _make_session_dict(
        session_id="ses_term",
        agent_id="agt_a",
        created_by="user_a",
        created_at=base + timedelta(seconds=2),
        status="terminated",
    )

    import litellm.proxy.proxy_server as ps

    monkeypatch.setattr(ps, "prisma_client", db.as_prisma())

    app = _build_app(auth=_user_auth("user_a"))
    client = TestClient(app)

    resp = client.get("/v2/agents/agt_a/sessions?status=ready")
    assert resp.status_code == 200, resp.text
    ids = {r["id"] for r in resp.json()["data"]}
    assert ids == {"ses_ready"}


def test_list_sessions_for_agent_invalid_cursor_returns_422(monkeypatch):
    db = _FakeAgentSessionDB()
    db.agents["agt_a"] = _make_agent_dict(agent_id="agt_a", created_by="user_a")

    import litellm.proxy.proxy_server as ps

    monkeypatch.setattr(ps, "prisma_client", db.as_prisma())

    app = _build_app(auth=_user_auth("user_a"))
    client = TestClient(app)

    resp = client.get("/v2/agents/agt_a/sessions?cursor=not-a-number")
    assert resp.status_code == 422, resp.text


def test_list_sessions_for_agent_db_not_connected_returns_500(monkeypatch):
    import litellm.proxy.proxy_server as ps

    monkeypatch.setattr(ps, "prisma_client", None)

    app = _build_app(auth=_user_auth("user_a"))
    client = TestClient(app)

    resp = client.get("/v2/agents/agt_a/sessions")
    assert resp.status_code == 500, resp.text
    detail = resp.json()["detail"]
    assert detail["error"] == CommonProxyErrors.db_not_connected_error.value


# ---------------------------------------------------------------------------
# GET /v2/sessions — global list across all caller's agents
# ---------------------------------------------------------------------------


def test_list_sessions_returns_all_caller_sessions(monkeypatch):
    """Without filters, GET /v2/sessions returns sessions across every agent
    the caller owns.
    """
    db = _FakeAgentSessionDB()
    db.agents["agt_a"] = _make_agent_dict(agent_id="agt_a", created_by="user_a")
    db.agents["agt_b"] = _make_agent_dict(agent_id="agt_b", created_by="user_a")

    base = datetime(2026, 5, 1, tzinfo=timezone.utc)
    db.sessions["ses_a1"] = _make_session_dict(
        session_id="ses_a1",
        agent_id="agt_a",
        created_by="user_a",
        created_at=base + timedelta(seconds=1),
    )
    db.sessions["ses_a2"] = _make_session_dict(
        session_id="ses_a2",
        agent_id="agt_a",
        created_by="user_a",
        created_at=base + timedelta(seconds=2),
    )
    db.sessions["ses_b1"] = _make_session_dict(
        session_id="ses_b1",
        agent_id="agt_b",
        created_by="user_a",
        created_at=base + timedelta(seconds=3),
    )

    import litellm.proxy.proxy_server as ps

    monkeypatch.setattr(ps, "prisma_client", db.as_prisma())

    app = _build_app(auth=_user_auth("user_a"))
    client = TestClient(app)

    resp = client.get("/v2/sessions")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    returned_ids = {row["id"] for row in body["data"]}
    assert returned_ids == {"ses_a1", "ses_a2", "ses_b1"}


def test_list_sessions_scopes_by_caller(monkeypatch):
    """Sessions belonging to other callers must NEVER leak into the response."""
    db = _FakeAgentSessionDB()

    base = datetime(2026, 5, 1, tzinfo=timezone.utc)
    db.sessions["ses_mine"] = _make_session_dict(
        session_id="ses_mine",
        agent_id="agt_a",
        created_by="user_a",
        created_at=base + timedelta(seconds=1),
    )
    db.sessions["ses_other"] = _make_session_dict(
        session_id="ses_other",
        agent_id="agt_a",
        created_by="user_other",
        created_at=base + timedelta(seconds=2),
    )

    import litellm.proxy.proxy_server as ps

    monkeypatch.setattr(ps, "prisma_client", db.as_prisma())

    app = _build_app(auth=_user_auth("user_a"))
    client = TestClient(app)

    resp = client.get("/v2/sessions")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    returned_ids = {row["id"] for row in body["data"]}
    assert returned_ids == {"ses_mine"}
    assert "ses_other" not in returned_ids


def test_list_sessions_filters_by_agent_id(monkeypatch):
    """`agent_id=` query param narrows results to that agent only."""
    db = _FakeAgentSessionDB()

    base = datetime(2026, 5, 1, tzinfo=timezone.utc)
    db.sessions["ses_a1"] = _make_session_dict(
        session_id="ses_a1",
        agent_id="agt_a",
        created_by="user_a",
        created_at=base + timedelta(seconds=1),
    )
    db.sessions["ses_b1"] = _make_session_dict(
        session_id="ses_b1",
        agent_id="agt_b",
        created_by="user_a",
        created_at=base + timedelta(seconds=2),
    )

    import litellm.proxy.proxy_server as ps

    monkeypatch.setattr(ps, "prisma_client", db.as_prisma())

    app = _build_app(auth=_user_auth("user_a"))
    client = TestClient(app)

    resp = client.get("/v2/sessions?agent_id=agt_a")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    returned_ids = {row["id"] for row in body["data"]}
    assert returned_ids == {"ses_a1"}


def test_list_sessions_filters_by_status(monkeypatch):
    """`status=` query param narrows results to that status."""
    db = _FakeAgentSessionDB()

    base = datetime(2026, 5, 1, tzinfo=timezone.utc)
    db.sessions["ses_ready"] = _make_session_dict(
        session_id="ses_ready",
        agent_id="agt_a",
        created_by="user_a",
        created_at=base + timedelta(seconds=1),
        status="ready",
    )
    db.sessions["ses_term"] = _make_session_dict(
        session_id="ses_term",
        agent_id="agt_a",
        created_by="user_a",
        created_at=base + timedelta(seconds=2),
        status="terminated",
    )

    import litellm.proxy.proxy_server as ps

    monkeypatch.setattr(ps, "prisma_client", db.as_prisma())

    app = _build_app(auth=_user_auth("user_a"))
    client = TestClient(app)

    resp = client.get("/v2/sessions?status=ready")
    assert resp.status_code == 200, resp.text
    ids = {r["id"] for r in resp.json()["data"]}
    assert ids == {"ses_ready"}


def test_list_sessions_pagination_walks_pages(monkeypatch):
    """Seed 4 sessions, page through with limit=2."""
    db = _FakeAgentSessionDB()

    base = datetime(2026, 5, 1, tzinfo=timezone.utc)
    for i in range(4):
        db.sessions[f"ses_{i}"] = _make_session_dict(
            session_id=f"ses_{i}",
            agent_id="agt_a",
            created_by="user_a",
            created_at=base + timedelta(seconds=i),
        )

    import litellm.proxy.proxy_server as ps

    monkeypatch.setattr(ps, "prisma_client", db.as_prisma())

    app = _build_app(auth=_user_auth("user_a"))
    client = TestClient(app)

    resp1 = client.get("/v2/sessions?limit=2")
    assert resp1.status_code == 200, resp1.text
    page1 = resp1.json()
    assert len(page1["data"]) == 2
    assert page1["has_more"] is True
    assert page1["next_cursor"] == "2"

    resp2 = client.get(f"/v2/sessions?limit=2&cursor={page1['next_cursor']}")
    assert resp2.status_code == 200, resp2.text
    page2 = resp2.json()
    assert len(page2["data"]) == 2
    assert page2["has_more"] is False
    assert page2["next_cursor"] is None

    page1_ids = [r["id"] for r in page1["data"]]
    page2_ids = [r["id"] for r in page2["data"]]
    all_seen = set(page1_ids) | set(page2_ids)
    assert len(all_seen) == 4
    assert set(page1_ids).isdisjoint(set(page2_ids))


def test_list_sessions_strips_internal_fields(monkeypatch):
    """`sandbox_url`, `sandbox_metadata`, `opencode_session_id` MUST be stripped."""
    db = _FakeAgentSessionDB()
    db.sessions["ses_a1"] = _make_session_dict(
        session_id="ses_a1",
        agent_id="agt_a",
        created_by="user_a",
        created_at=datetime(2026, 5, 1, tzinfo=timezone.utc),
    )

    import litellm.proxy.proxy_server as ps

    monkeypatch.setattr(ps, "prisma_client", db.as_prisma())

    app = _build_app(auth=_user_auth("user_a"))
    client = TestClient(app)

    resp = client.get("/v2/sessions")
    assert resp.status_code == 200, resp.text
    assert "sandbox_url" not in resp.text
    assert "sandbox_metadata" not in resp.text
    assert "opencode_session_id" not in resp.text


def test_list_sessions_invalid_cursor_returns_422(monkeypatch):
    db = _FakeAgentSessionDB()

    import litellm.proxy.proxy_server as ps

    monkeypatch.setattr(ps, "prisma_client", db.as_prisma())

    app = _build_app(auth=_user_auth("user_a"))
    client = TestClient(app)

    resp = client.get("/v2/sessions?cursor=not-a-number")
    assert resp.status_code == 422, resp.text


def test_list_sessions_db_not_connected_returns_500(monkeypatch):
    import litellm.proxy.proxy_server as ps

    monkeypatch.setattr(ps, "prisma_client", None)

    app = _build_app(auth=_user_auth("user_a"))
    client = TestClient(app)

    resp = client.get("/v2/sessions")
    assert resp.status_code == 500, resp.text
    detail = resp.json()["detail"]
    assert detail["error"] == CommonProxyErrors.db_not_connected_error.value


def test_list_sessions_empty_when_no_sessions(monkeypatch):
    db = _FakeAgentSessionDB()

    import litellm.proxy.proxy_server as ps

    monkeypatch.setattr(ps, "prisma_client", db.as_prisma())

    app = _build_app(auth=_user_auth("user_a"))
    client = TestClient(app)

    resp = client.get("/v2/sessions")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["data"] == []
    assert body["has_more"] is False
    assert body["next_cursor"] is None
