"""Tests for managed_agents_endpoints/endpoints_agents.py.

Covers GET /v1/managed_agents/agents (list) and GET /v1/managed_agents/agents/{id},
including the ownership / authorization gate. The POST /agents create flow is
exercised end-to-end via the session tests, so this file focuses on the read
endpoints.
"""

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.managed_agents_endpoints.endpoints import router

# Importing the module registers /agents and /agents/{id} routes onto `router`.
import litellm.proxy.managed_agents_endpoints.endpoints_agents  # noqa: F401


@pytest.fixture
def user():
    return UserAPIKeyAuth(
        api_key="sk-user", user_id="u1", user_role=LitellmUserRoles.INTERNAL_USER
    )


@pytest.fixture
def other_user():
    return UserAPIKeyAuth(
        api_key="sk-other", user_id="u2", user_role=LitellmUserRoles.INTERNAL_USER
    )


@pytest.fixture
def admin():
    return UserAPIKeyAuth(
        api_key="sk-admin", user_id="a1", user_role=LitellmUserRoles.PROXY_ADMIN
    )


@pytest.fixture
def app_factory():
    def make(auth_user):
        app = FastAPI()
        app.include_router(router)
        app.dependency_overrides[user_api_key_auth] = lambda: auth_user
        return TestClient(app)

    return make


def _make_agent(agent_id="agt-1", created_by="u1", **kw):
    base = dict(
        agent_id=agent_id,
        agent_name="a",
        model="anthropic/claude-sonnet-4-6",
        prompt="be concise",
        tools=[],
        template_id="tmpl-1",
        branch="main",
        metadata={},
        created_at=datetime(2026, 5, 7, tzinfo=timezone.utc),
        created_by=created_by,
    )
    base.update(kw)
    return SimpleNamespace(**base)


def _make_prisma(agents=None, agent=None):
    p = MagicMock()
    agent_t = MagicMock()
    agent_t.find_unique = AsyncMock(return_value=agent)
    agent_t.find_many = AsyncMock(return_value=list(agents) if agents else [])
    p.db.litellm_managedagenttable = agent_t
    return p


# ---------------------------------------------------------------------------
# list_agents
# ---------------------------------------------------------------------------


def test_list_agents_returns_rows(app_factory, user):
    client = app_factory(user)
    rows = [
        _make_agent(agent_id="a1", agent_name="alpha"),
        _make_agent(agent_id="a2", agent_name="beta"),
    ]
    prisma = _make_prisma(agents=rows)
    with patch("litellm.proxy.proxy_server.prisma_client", prisma):
        resp = client.get("/v1/managed_agents/agents")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert [r["id"] for r in body] == ["a1", "a2"]
    assert body[0]["name"] == "alpha"
    assert body[0]["model"] == "anthropic/claude-sonnet-4-6"
    assert body[0]["template_id"] == "tmpl-1"
    assert body[0]["branch"] == "main"
    # created_at is serialized as ISO string
    assert body[0]["created_at"].startswith("2026-05-07")
    # Ordered by created_at desc
    _, kwargs = prisma.db.litellm_managedagenttable.find_many.call_args
    assert kwargs["order"] == {"created_at": "desc"}


def test_list_agents_empty(app_factory, user):
    client = app_factory(user)
    prisma = _make_prisma(agents=[])
    with patch("litellm.proxy.proxy_server.prisma_client", prisma):
        resp = client.get("/v1/managed_agents/agents")
    assert resp.status_code == 200
    assert resp.json() == []


def test_list_agents_500_when_prisma_unavailable(app_factory, user):
    client = app_factory(user)
    with patch("litellm.proxy.proxy_server.prisma_client", None):
        resp = client.get("/v1/managed_agents/agents")
    assert resp.status_code == 500
    assert "prisma" in resp.json()["detail"].lower()


def test_list_agents_filters_by_owner_for_non_admin(app_factory, user):
    client = app_factory(user)
    prisma = _make_prisma(agents=[])
    with patch("litellm.proxy.proxy_server.prisma_client", prisma):
        resp = client.get("/v1/managed_agents/agents")
    assert resp.status_code == 200
    _, kwargs = prisma.db.litellm_managedagenttable.find_many.call_args
    assert kwargs["where"] == {"created_by": "u1"}


def test_list_agents_admin_no_owner_filter(app_factory, admin):
    client = app_factory(admin)
    prisma = _make_prisma(agents=[])
    with patch("litellm.proxy.proxy_server.prisma_client", prisma):
        resp = client.get("/v1/managed_agents/agents")
    assert resp.status_code == 200
    _, kwargs = prisma.db.litellm_managedagenttable.find_many.call_args
    assert kwargs["where"] == {}


# ---------------------------------------------------------------------------
# get_agent
# ---------------------------------------------------------------------------


def test_get_agent_happy(app_factory, user):
    client = app_factory(user)
    prisma = _make_prisma(agent=_make_agent(agent_id="agt-9", created_by="u1"))
    with patch("litellm.proxy.proxy_server.prisma_client", prisma):
        resp = client.get("/v1/managed_agents/agents/agt-9")
    assert resp.status_code == 200
    body = resp.json()
    assert body["id"] == "agt-9"
    assert body["template_id"] == "tmpl-1"
    assert body["branch"] == "main"


def test_get_agent_404(app_factory, user):
    client = app_factory(user)
    prisma = _make_prisma(agent=None)
    with patch("litellm.proxy.proxy_server.prisma_client", prisma):
        resp = client.get("/v1/managed_agents/agents/missing")
    assert resp.status_code == 404
    assert "missing" in resp.json()["detail"]


def test_get_agent_returns_404_for_non_owner(app_factory, other_user):
    client = app_factory(other_user)
    prisma = _make_prisma(agent=_make_agent(agent_id="agt-9", created_by="u1"))
    with patch("litellm.proxy.proxy_server.prisma_client", prisma):
        resp = client.get("/v1/managed_agents/agents/agt-9")
    assert resp.status_code == 404


def test_get_agent_visible_to_admin(app_factory, admin):
    client = app_factory(admin)
    prisma = _make_prisma(agent=_make_agent(agent_id="agt-9", created_by="u1"))
    with patch("litellm.proxy.proxy_server.prisma_client", prisma):
        resp = client.get("/v1/managed_agents/agents/agt-9")
    assert resp.status_code == 200
