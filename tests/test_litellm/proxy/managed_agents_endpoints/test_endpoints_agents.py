"""Tests for managed_agents_endpoints/endpoints_agents.py.

Covers GET /v1/managed_agents/agents (list) and GET /v1/managed_agents/agents/{id},
including the ownership / authorization gate. The POST /agents create flow is
exercised end-to-end via the session tests, so this file focuses on the read
endpoints.
"""

import json
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


def _make_prisma(agents=None, agent=None, updated=None):
    p = MagicMock()
    agent_t = MagicMock()
    agent_t.find_unique = AsyncMock(return_value=agent)
    agent_t.find_many = AsyncMock(return_value=list(agents) if agents else [])
    agent_t.update = AsyncMock(return_value=updated if updated is not None else agent)
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


# ---------------------------------------------------------------------------
# create_agent — template visibility enforcement
# ---------------------------------------------------------------------------


def _make_template(template_id="tmpl-1", visibility="public", created_by="u1"):
    return SimpleNamespace(
        template_id=template_id,
        template_name="t",
        dockerfile_id="opencode",
        container_port=4096,
        repo_url="https://github.com/x/y",
        default_branch="main",
        visibility=visibility,
        git_credential_id=None,
        created_by=created_by,
    )


def test_list_agents_returns_empty_when_user_id_none(app_factory):
    """Non-admin caller with no user_id must not see other users' rows."""
    user_no_id = UserAPIKeyAuth(
        api_key="sk", user_id=None, user_role=LitellmUserRoles.INTERNAL_USER
    )
    client = app_factory(user_no_id)
    prisma = _make_prisma(agents=[_make_agent(agent_id="x", created_by="someone-else")])
    with patch("litellm.proxy.proxy_server.prisma_client", prisma):
        resp = client.get("/v1/managed_agents/agents")
    assert resp.status_code == 200
    assert resp.json() == []
    prisma.db.litellm_managedagenttable.find_many.assert_not_called()


def test_create_agent_encrypts_litellm_api_key(app_factory, user):
    """The plaintext API key must NOT be persisted in metadata; the
    encrypted form goes under litellm_api_key_encrypted."""
    client = app_factory(user)
    template = _make_template(visibility="public", created_by="u1")
    p = MagicMock()
    template_t = MagicMock()
    template_t.find_unique = AsyncMock(return_value=template)
    p.db.litellm_managedagentsandboxtemplatetable = template_t
    agent_t = MagicMock()
    agent_t.create = AsyncMock(return_value=_make_agent(agent_id="agt-2"))
    p.db.litellm_managedagenttable = agent_t

    body = {
        "name": "agt",
        "model": "anthropic/claude-sonnet-4-6",
        "template_id": "tmpl-1",
        "litellm_api_key": "sk-secret",
    }

    with (
        patch("litellm.proxy.proxy_server.prisma_client", p),
        patch(
            "litellm.proxy.managed_agents_endpoints.endpoints_agents.encrypt_value_helper",
            return_value="ENCRYPTED",
        ),
        patch(
            "litellm.proxy.managed_agents_endpoints.endpoints_agents.decrypt_git_token",
            new=AsyncMock(return_value=None),
        ),
        patch(
            "litellm.proxy.managed_agents_endpoints.endpoints_agents.validate_repo_branch"
        ),
    ):
        resp = client.post("/v1/managed_agents/agents", json=body)

    import json as _json

    assert resp.status_code == 200, resp.text
    raw_meta = agent_t.create.call_args.kwargs["data"]["metadata"]
    metadata = _json.loads(raw_meta) if isinstance(raw_meta, str) else raw_meta
    assert metadata["litellm_api_key_encrypted"] == "ENCRYPTED"
    assert "litellm_api_key" not in metadata
    assert "sk-secret" not in str(metadata)


def test_create_agent_rejects_private_template_for_non_owner(app_factory, other_user):
    client = app_factory(other_user)
    template = _make_template(visibility="private", created_by="u1")
    p = MagicMock()
    template_t = MagicMock()
    template_t.find_unique = AsyncMock(return_value=template)
    p.db.litellm_managedagentsandboxtemplatetable = template_t
    agent_t = MagicMock()
    agent_t.create = AsyncMock()
    p.db.litellm_managedagenttable = agent_t

    body = {
        "name": "agt",
        "model": "anthropic/claude-sonnet-4-6",
        "template_id": "tmpl-1",
    }

    with patch("litellm.proxy.proxy_server.prisma_client", p):
        resp = client.post("/v1/managed_agents/agents", json=body)

    assert resp.status_code == 404
    agent_t.create.assert_not_called()


# ---------------------------------------------------------------------------
# pfp_url surfaces from metadata
# ---------------------------------------------------------------------------


def test_get_agent_surfaces_pfp_url_from_metadata(app_factory, user):
    client = app_factory(user)
    a = _make_agent(metadata={"pfp_url": "data:image/jpeg;base64,AAAA"})
    prisma = _make_prisma(agent=a)
    with patch("litellm.proxy.proxy_server.prisma_client", prisma):
        resp = client.get("/v1/managed_agents/agents/agt-1")
    assert resp.status_code == 200
    assert resp.json()["pfp_url"] == "data:image/jpeg;base64,AAAA"


def test_get_agent_pfp_url_null_when_absent(app_factory, user):
    client = app_factory(user)
    a = _make_agent(metadata={})
    prisma = _make_prisma(agent=a)
    with patch("litellm.proxy.proxy_server.prisma_client", prisma):
        resp = client.get("/v1/managed_agents/agents/agt-1")
    assert resp.json()["pfp_url"] is None


def test_get_agent_surfaces_prompt(app_factory, user):
    client = app_factory(user)
    a = _make_agent(prompt="You review PRs for clarity, correctness, security.")
    prisma = _make_prisma(agent=a)
    with patch("litellm.proxy.proxy_server.prisma_client", prisma):
        resp = client.get("/v1/managed_agents/agents/agt-1")
    assert resp.status_code == 200
    assert resp.json()["prompt"] == "You review PRs for clarity, correctness, security."


def test_get_agent_surfaces_mcp_servers_from_metadata(app_factory, user):
    client = app_factory(user)
    a = _make_agent(metadata={"mcp_servers": ["mcp-1", "mcp-2"]})
    prisma = _make_prisma(agent=a)
    with patch("litellm.proxy.proxy_server.prisma_client", prisma):
        resp = client.get("/v1/managed_agents/agents/agt-1")
    assert resp.json()["mcp_servers"] == ["mcp-1", "mcp-2"]


def test_get_agent_mcp_servers_empty_when_absent(app_factory, user):
    client = app_factory(user)
    a = _make_agent(metadata={})
    prisma = _make_prisma(agent=a)
    with patch("litellm.proxy.proxy_server.prisma_client", prisma):
        resp = client.get("/v1/managed_agents/agents/agt-1")
    assert resp.json()["mcp_servers"] == []


def test_get_agent_mcp_servers_drops_non_strings(app_factory, user):
    """If the column somehow ends up with junk, the read endpoint should
    silently coerce rather than 500."""
    client = app_factory(user)
    a = _make_agent(metadata={"mcp_servers": ["good", 123, None, "ok"]})
    prisma = _make_prisma(agent=a)
    with patch("litellm.proxy.proxy_server.prisma_client", prisma):
        resp = client.get("/v1/managed_agents/agents/agt-1")
    assert resp.json()["mcp_servers"] == ["good", "ok"]


def test_update_agent_sets_mcp_servers(app_factory, user):
    client = app_factory(user)
    existing = _make_agent(metadata={"litellm_api_key": "sk-x"})
    updated = _make_agent(
        metadata={"litellm_api_key": "sk-x", "mcp_servers": ["mcp-1"]}
    )
    prisma = _make_prisma(agent=existing, updated=updated)
    with patch("litellm.proxy.proxy_server.prisma_client", prisma):
        resp = client.patch(
            "/v1/managed_agents/agents/agt-1", json={"mcp_servers": ["mcp-1"]}
        )
    assert resp.status_code == 200
    _, kwargs = prisma.db.litellm_managedagenttable.update.call_args
    new_metadata = json.loads(kwargs["data"]["metadata"])
    assert new_metadata["mcp_servers"] == ["mcp-1"]
    assert new_metadata["litellm_api_key"] == "sk-x"


def test_update_agent_clears_mcp_servers_with_empty_list(app_factory, user):
    client = app_factory(user)
    existing = _make_agent(metadata={"mcp_servers": ["old"]})
    updated = _make_agent(metadata={})
    prisma = _make_prisma(agent=existing, updated=updated)
    with patch("litellm.proxy.proxy_server.prisma_client", prisma):
        resp = client.patch("/v1/managed_agents/agents/agt-1", json={"mcp_servers": []})
    assert resp.status_code == 200
    _, kwargs = prisma.db.litellm_managedagenttable.update.call_args
    new_metadata = json.loads(kwargs["data"]["metadata"])
    assert "mcp_servers" not in new_metadata


def test_update_agent_pfp_and_mcp_in_one_call(app_factory, user):
    """Both edits should land in a single metadata write, not two races."""
    client = app_factory(user)
    existing = _make_agent(metadata={"litellm_api_key": "sk-x"})
    updated = _make_agent(metadata={})
    prisma = _make_prisma(agent=existing, updated=updated)
    with patch("litellm.proxy.proxy_server.prisma_client", prisma):
        resp = client.patch(
            "/v1/managed_agents/agents/agt-1",
            json={"pfp_url": "data:image/png;base64,XXX", "mcp_servers": ["m1"]},
        )
    assert resp.status_code == 200
    # Single update call, with both keys present in the new metadata.
    assert prisma.db.litellm_managedagenttable.update.call_count == 1
    _, kwargs = prisma.db.litellm_managedagenttable.update.call_args
    new_metadata = json.loads(kwargs["data"]["metadata"])
    assert new_metadata["pfp_url"] == "data:image/png;base64,XXX"
    assert new_metadata["mcp_servers"] == ["m1"]
    assert new_metadata["litellm_api_key"] == "sk-x"


def test_get_agent_metadata_as_json_string(app_factory, user):
    """Prisma sometimes returns Json columns as string. _coerce_metadata
    should parse it transparently."""
    client = app_factory(user)
    a = _make_agent(metadata='{"pfp_url": "https://x/y.png"}')
    prisma = _make_prisma(agent=a)
    with patch("litellm.proxy.proxy_server.prisma_client", prisma):
        resp = client.get("/v1/managed_agents/agents/agt-1")
    assert resp.json()["pfp_url"] == "https://x/y.png"


# ---------------------------------------------------------------------------
# update_agent (PATCH)
# ---------------------------------------------------------------------------


def test_update_agent_sets_pfp_url(app_factory, user):
    client = app_factory(user)
    existing = _make_agent(metadata={"litellm_api_key": "sk-x"})
    updated = _make_agent(
        metadata={"litellm_api_key": "sk-x", "pfp_url": "data:image/jpeg;base64,AAAA"}
    )
    prisma = _make_prisma(agent=existing, updated=updated)
    with patch("litellm.proxy.proxy_server.prisma_client", prisma):
        resp = client.patch(
            "/v1/managed_agents/agents/agt-1",
            json={"pfp_url": "data:image/jpeg;base64,AAAA"},
        )
    assert resp.status_code == 200, resp.text
    assert resp.json()["pfp_url"] == "data:image/jpeg;base64,AAAA"

    # Confirm we didn't clobber existing metadata. jsonify_object serializes
    # the metadata dict to a JSON string before it hits Prisma.
    _, kwargs = prisma.db.litellm_managedagenttable.update.call_args
    new_metadata = json.loads(kwargs["data"]["metadata"])
    assert new_metadata["litellm_api_key"] == "sk-x"
    assert new_metadata["pfp_url"] == "data:image/jpeg;base64,AAAA"


def test_update_agent_clears_pfp_url_with_empty_string(app_factory, user):
    client = app_factory(user)
    existing = _make_agent(metadata={"pfp_url": "old", "litellm_api_key": "sk-x"})
    updated = _make_agent(metadata={"litellm_api_key": "sk-x"})
    prisma = _make_prisma(agent=existing, updated=updated)
    with patch("litellm.proxy.proxy_server.prisma_client", prisma):
        resp = client.patch("/v1/managed_agents/agents/agt-1", json={"pfp_url": ""})
    assert resp.status_code == 200
    _, kwargs = prisma.db.litellm_managedagenttable.update.call_args
    new_metadata = json.loads(kwargs["data"]["metadata"])
    assert "pfp_url" not in new_metadata
    # other metadata is preserved
    assert new_metadata["litellm_api_key"] == "sk-x"


def test_update_agent_renames_via_name_field(app_factory, user):
    client = app_factory(user)
    existing = _make_agent(agent_name="old")
    updated = _make_agent(agent_name="renamed")
    prisma = _make_prisma(agent=existing, updated=updated)
    with patch("litellm.proxy.proxy_server.prisma_client", prisma):
        resp = client.patch("/v1/managed_agents/agents/agt-1", json={"name": "renamed"})
    assert resp.status_code == 200
    _, kwargs = prisma.db.litellm_managedagenttable.update.call_args
    assert kwargs["data"]["agent_name"] == "renamed"


def test_update_agent_404(app_factory, user):
    client = app_factory(user)
    prisma = _make_prisma(agent=None)
    with patch("litellm.proxy.proxy_server.prisma_client", prisma):
        resp = client.patch(
            "/v1/managed_agents/agents/missing",
            json={"pfp_url": "x"},
        )
    assert resp.status_code == 404


def test_update_agent_no_op_when_body_empty(app_factory, user):
    client = app_factory(user)
    existing = _make_agent()
    prisma = _make_prisma(agent=existing)
    with patch("litellm.proxy.proxy_server.prisma_client", prisma):
        resp = client.patch("/v1/managed_agents/agents/agt-1", json={})
    assert resp.status_code == 200
    # update was not called when nothing to update
    prisma.db.litellm_managedagenttable.update.assert_not_called()


def test_update_agent_rejects_unknown_field(app_factory, user):
    client = app_factory(user)
    existing = _make_agent()
    prisma = _make_prisma(agent=existing)
    with patch("litellm.proxy.proxy_server.prisma_client", prisma):
        resp = client.patch(
            "/v1/managed_agents/agents/agt-1",
            json={"model": "gpt-4o-mini"},
        )
    # AgentUpdate has extra="forbid"
    assert resp.status_code == 422
