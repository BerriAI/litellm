"""Tests for managed_agents_endpoints/endpoints_sessions.py.

Covers POST /v1/managed_agents/agents/{agent_id}/session,
GET /v1/managed_agents/sessions/{session_id},
DELETE /v1/managed_agents/sessions/{session_id}.

AWS lifecycle (bootstrap_shared_infra, run_task_sync, wait_running_get_ip_sync,
wait_http_ready, stop_task_sync, stop_session_task) is fully mocked.
Harness HTTP (harness_create_session, harness_send_message) is also mocked —
this file is a unit test, not the smoke test that hits a live container.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.managed_agents_endpoints.endpoints import router

# Importing the module registers /agents/{agent_id}/session, /sessions/{id}
# routes onto `router`. Without this, the routes don't exist on the test app.
import litellm.proxy.managed_agents_endpoints.endpoints_sessions  # noqa: F401


@pytest.fixture
def user():
    return UserAPIKeyAuth(
        api_key="sk-user", user_id="u1", user_role=LitellmUserRoles.INTERNAL_USER
    )


@pytest.fixture
def app_factory():
    def make(auth_user):
        app = FastAPI()
        app.include_router(router)
        app.dependency_overrides[user_api_key_auth] = lambda: auth_user
        return TestClient(app)

    return make


def _make_template(build_status="ready"):
    return SimpleNamespace(
        template_id="tmpl-1",
        template_name="t",
        dockerfile_id="opencode",
        container_port=4096,
        repo_url="https://github.com/x/y",
        default_branch="main",
        visibility="public",
        git_credential_id=None,
        image_uri="img:abc",
        task_def_arn="arn:td",
        image_hash="abc",
        build_status=build_status,
        build_error=None,
    )


def _make_agent(template):
    a = SimpleNamespace(
        agent_id="agt-1",
        agent_name="a",
        model="anthropic/claude-sonnet-4-6",
        prompt="be concise",
        tools=[],
        template_id=template.template_id,
        branch="main",
        metadata={"litellm_api_key": "sk-x", "litellm_api_base": "http://x"},
    )
    a.template = template
    return a


def _make_session(session_id="sess-1", **kw):
    base = dict(
        session_id=session_id,
        agent_id="agt-1",
        status="creating",
        task_arn=None,
        sandbox_url=None,
        harness_session_id=None,
        fargate_cluster="litellm-agents",
        fargate_task_def_arn="arn:td",
        failure_reason=None,
        stopped_at=None,
        last_seen_at=None,
        created_by="u1",
        team_id=None,
    )
    base.update(kw)
    return SimpleNamespace(**base)


def _make_prisma(agent=None, session=None, sessions=None):
    p = MagicMock()
    agent_t = MagicMock()
    agent_t.find_unique = AsyncMock(return_value=agent)
    p.db.litellm_managedagenttable = agent_t

    sess_t = MagicMock()
    if session is not None:
        sess_t.create = AsyncMock(return_value=session)
        sess_t.find_unique = AsyncMock(return_value=session)
    else:
        sess_t.create = AsyncMock()
        sess_t.find_unique = AsyncMock(return_value=None)
    sess_t.update = AsyncMock()
    sess_t.find_many = AsyncMock(return_value=list(sessions) if sessions else [])
    p.db.litellm_managedagentsessiontable = sess_t
    return p


# ---------------------------------------------------------------------------
# create_session
# ---------------------------------------------------------------------------


def test_create_session_404_when_agent_missing(app_factory, user):
    client = app_factory(user)
    prisma = _make_prisma(agent=None)
    with patch("litellm.proxy.proxy_server.prisma_client", prisma):
        resp = client.post("/v1/managed_agents/agents/missing/session", json={})
    assert resp.status_code == 404
    assert "missing" in resp.json()["detail"]


def test_create_session_409_when_template_not_ready(app_factory, user):
    client = app_factory(user)
    template = _make_template(build_status="pending")
    agent = _make_agent(template)
    prisma = _make_prisma(agent=agent)
    with patch("litellm.proxy.proxy_server.prisma_client", prisma):
        resp = client.post("/v1/managed_agents/agents/agt-1/session", json={})
    assert resp.status_code == 409
    assert "not ready" in resp.json()["detail"]


def test_create_session_happy_path_with_initial_prompt(app_factory, user):
    client = app_factory(user)
    template = _make_template()
    agent = _make_agent(template)
    created_session = _make_session()
    prisma = _make_prisma(agent=agent, session=created_session)

    infra = SimpleNamespace(
        cluster_arn="arn:cluster",
        task_exec_role_arn="arn:role",
        security_group_id="sg-1",
        log_group_name="/ecs/x",
        vpc_id="vpc-1",
        subnet_ids=["subnet-1"],
    )

    with (
        patch("litellm.proxy.proxy_server.prisma_client", prisma),
        patch(
            "litellm.proxy.managed_agents_endpoints.endpoints_sessions.bootstrap_shared_infra",
            return_value=infra,
        ) as mock_boot,
        patch(
            "litellm.proxy.managed_agents_endpoints.endpoints_sessions.run_task_sync",
            return_value="arn:task/abc",
        ) as mock_run,
        patch(
            "litellm.proxy.managed_agents_endpoints.endpoints_sessions.wait_running_get_ip_sync",
            return_value="1.2.3.4",
        ),
        patch(
            "litellm.proxy.managed_agents_endpoints.endpoints_sessions.wait_http_ready",
            new=AsyncMock(return_value=None),
        ),
        patch(
            "litellm.proxy.managed_agents_endpoints.endpoints_sessions.harness_create_session",
            new=AsyncMock(return_value="harness-sess-1"),
        ),
        patch(
            "litellm.proxy.managed_agents_endpoints.endpoints_sessions.harness_send_message",
            new=AsyncMock(return_value={"parts": [{"type": "text", "text": "hi"}]}),
        ),
        patch(
            "litellm.proxy.managed_agents_endpoints.endpoints_sessions.decrypt_git_token",
            new=AsyncMock(return_value=None),
        ),
        patch(
            "litellm.proxy.managed_agents_endpoints.config_loader.MANAGED_AGENTS_CONFIG",
            SimpleNamespace(
                aws_region="us-west-2",
                aws=SimpleNamespace(cluster=None),
            ),
        ),
    ):
        resp = client.post(
            "/v1/managed_agents/agents/agt-1/session",
            json={"title": "smoke", "initial_prompt": "what is this repo?"},
        )

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["status"] == "ready"
    assert body["sandbox_url"] == "http://1.2.3.4:4096"
    assert body["task_arn"] == "arn:task/abc"
    assert body["response"]["parts"][0]["text"] == "hi"

    # Bootstrap was called with config region + container port
    mock_boot.assert_called_once()
    args, _ = mock_boot.call_args
    assert args[0] == "us-west-2"
    assert args[2] == 4096

    # run_task_sync got the env vars from agent.metadata + template
    _, run_kwargs = mock_run.call_args
    env = run_kwargs["env"]
    assert env["LITELLM_API_KEY"] == "sk-x"
    assert env["LITELLM_API_BASE"] == "http://x"
    assert env["LITELLM_DEFAULT_MODEL"] == "anthropic/claude-sonnet-4-6"
    assert env["REPO_URL"] == "https://github.com/x/y"
    assert env["BRANCH"] == "main"
    assert env["AGENT_PROMPT"] == "be concise"
    assert "GIT_TOKEN" not in env  # template has no credential


def test_create_session_happy_path_no_initial_prompt(app_factory, user):
    client = app_factory(user)
    template = _make_template()
    agent = _make_agent(template)
    prisma = _make_prisma(agent=agent, session=_make_session())
    infra = SimpleNamespace(
        cluster_arn="arn:cluster",
        task_exec_role_arn="arn:role",
        security_group_id="sg-1",
        log_group_name="/ecs/x",
        vpc_id="vpc-1",
        subnet_ids=["subnet-1"],
    )
    send_mock = AsyncMock(return_value={"parts": []})

    with (
        patch("litellm.proxy.proxy_server.prisma_client", prisma),
        patch(
            "litellm.proxy.managed_agents_endpoints.endpoints_sessions.bootstrap_shared_infra",
            return_value=infra,
        ),
        patch(
            "litellm.proxy.managed_agents_endpoints.endpoints_sessions.run_task_sync",
            return_value="arn:task/abc",
        ),
        patch(
            "litellm.proxy.managed_agents_endpoints.endpoints_sessions.wait_running_get_ip_sync",
            return_value="1.2.3.4",
        ),
        patch(
            "litellm.proxy.managed_agents_endpoints.endpoints_sessions.wait_http_ready",
            new=AsyncMock(return_value=None),
        ),
        patch(
            "litellm.proxy.managed_agents_endpoints.endpoints_sessions.harness_create_session",
            new=AsyncMock(return_value="harness-sess-1"),
        ),
        patch(
            "litellm.proxy.managed_agents_endpoints.endpoints_sessions.harness_send_message",
            new=send_mock,
        ),
        patch(
            "litellm.proxy.managed_agents_endpoints.endpoints_sessions.decrypt_git_token",
            new=AsyncMock(return_value=None),
        ),
        patch(
            "litellm.proxy.managed_agents_endpoints.config_loader.MANAGED_AGENTS_CONFIG",
            SimpleNamespace(
                aws_region="us-west-2",
                aws=SimpleNamespace(cluster=None),
            ),
        ),
    ):
        resp = client.post("/v1/managed_agents/agents/agt-1/session", json={})

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["status"] == "ready"
    # No initial_prompt → harness_send_message NOT called
    send_mock.assert_not_called()
    assert body["response"] is None


def test_create_session_marks_failed_on_exception(app_factory, user):
    client = app_factory(user)
    template = _make_template()
    agent = _make_agent(template)
    prisma = _make_prisma(agent=agent, session=_make_session())
    infra = SimpleNamespace(
        cluster_arn="arn:cluster",
        task_exec_role_arn="arn:role",
        security_group_id="sg-1",
        log_group_name="/ecs/x",
        vpc_id="vpc-1",
        subnet_ids=["subnet-1"],
    )

    with (
        patch("litellm.proxy.proxy_server.prisma_client", prisma),
        patch(
            "litellm.proxy.managed_agents_endpoints.endpoints_sessions.bootstrap_shared_infra",
            return_value=infra,
        ),
        patch(
            "litellm.proxy.managed_agents_endpoints.endpoints_sessions.run_task_sync",
            return_value="arn:task/abc",
        ),
        patch(
            "litellm.proxy.managed_agents_endpoints.endpoints_sessions.wait_running_get_ip_sync",
            side_effect=TimeoutError("never reached RUNNING"),
        ),
        patch(
            "litellm.proxy.managed_agents_endpoints.endpoints_sessions.stop_task_sync",
            return_value=None,
        ) as mock_stop,
        patch(
            "litellm.proxy.managed_agents_endpoints.endpoints_sessions.decrypt_git_token",
            new=AsyncMock(return_value=None),
        ),
        patch(
            "litellm.proxy.managed_agents_endpoints.config_loader.MANAGED_AGENTS_CONFIG",
            SimpleNamespace(
                aws_region="us-west-2",
                aws=SimpleNamespace(cluster=None),
            ),
        ),
    ):
        resp = client.post("/v1/managed_agents/agents/agt-1/session", json={})

    assert resp.status_code == 500
    assert "session create failed" in resp.json()["detail"]

    # Session row was marked failed
    update_calls = prisma.db.litellm_managedagentsessiontable.update.call_args_list
    failed_calls = [
        c for c in update_calls if c.kwargs.get("data", {}).get("status") == "failed"
    ]
    assert failed_calls, f"expected status=failed update, got {update_calls}"

    # Best-effort stop_task_sync was attempted
    mock_stop.assert_called_once()


# ---------------------------------------------------------------------------
# get_session
# ---------------------------------------------------------------------------


def test_get_session_happy(app_factory, user):
    client = app_factory(user)
    sess = _make_session(
        session_id="sess-9",
        status="ready",
        task_arn="arn:task/9",
        sandbox_url="http://1.2.3.4:4096",
    )
    prisma = _make_prisma(session=sess)
    with patch("litellm.proxy.proxy_server.prisma_client", prisma):
        resp = client.get("/v1/managed_agents/sessions/sess-9")
    assert resp.status_code == 200
    body = resp.json()
    assert body["id"] == "sess-9"
    assert body["status"] == "ready"
    assert body["sandbox_url"] == "http://1.2.3.4:4096"
    assert body["task_arn"] == "arn:task/9"


def test_get_session_404(app_factory, user):
    client = app_factory(user)
    prisma = _make_prisma()
    with patch("litellm.proxy.proxy_server.prisma_client", prisma):
        resp = client.get("/v1/managed_agents/sessions/nope")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# delete_session
# ---------------------------------------------------------------------------


def test_delete_session_404(app_factory, user):
    client = app_factory(user)
    prisma = _make_prisma()
    with patch("litellm.proxy.proxy_server.prisma_client", prisma):
        resp = client.delete("/v1/managed_agents/sessions/nope")
    assert resp.status_code == 404


def test_delete_session_stops_task_and_marks_dead(app_factory, user):
    client = app_factory(user)
    sess = _make_session(
        session_id="sess-9",
        status="ready",
        task_arn="arn:task/9",
        fargate_cluster="my-cluster",
    )
    prisma = _make_prisma(session=sess)
    stop_mock = AsyncMock(return_value=None)
    with (
        patch("litellm.proxy.proxy_server.prisma_client", prisma),
        patch(
            "litellm.proxy.managed_agents_endpoints.endpoints_sessions.stop_session_task",
            new=stop_mock,
        ),
        patch(
            "litellm.proxy.managed_agents_endpoints.config_loader.MANAGED_AGENTS_CONFIG",
            SimpleNamespace(
                aws_region="us-west-2",
                aws=SimpleNamespace(cluster=None),
            ),
        ),
    ):
        resp = client.delete("/v1/managed_agents/sessions/sess-9")
    assert resp.status_code == 200
    assert resp.json() == {"id": "sess-9", "status": "dead"}

    stop_mock.assert_awaited_once()
    _, kwargs = stop_mock.call_args
    # Cluster comes from row.fargate_cluster, not config default
    assert kwargs["cluster"] == "my-cluster"
    assert kwargs["task_arn"] == "arn:task/9"
    assert kwargs["session_id"] == "sess-9"

    # Row was updated to dead
    update_calls = prisma.db.litellm_managedagentsessiontable.update.call_args_list
    assert any(
        c.kwargs.get("data", {}).get("status") == "dead" for c in update_calls
    ), f"expected status=dead update, got {update_calls}"


def test_delete_session_no_task_arn_skips_stop(app_factory, user):
    client = app_factory(user)
    sess = _make_session(session_id="sess-9", status="creating", task_arn=None)
    prisma = _make_prisma(session=sess)
    stop_mock = AsyncMock(return_value=None)
    with (
        patch("litellm.proxy.proxy_server.prisma_client", prisma),
        patch(
            "litellm.proxy.managed_agents_endpoints.endpoints_sessions.stop_session_task",
            new=stop_mock,
        ),
        patch(
            "litellm.proxy.managed_agents_endpoints.config_loader.MANAGED_AGENTS_CONFIG",
            SimpleNamespace(
                aws_region="us-west-2",
                aws=SimpleNamespace(cluster=None),
            ),
        ),
    ):
        resp = client.delete("/v1/managed_agents/sessions/sess-9")
    assert resp.status_code == 200
    stop_mock.assert_not_called()


# ---------------------------------------------------------------------------
# list_sessions
# ---------------------------------------------------------------------------


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


def test_list_sessions_returns_all_when_no_filter(app_factory, admin):
    client = app_factory(admin)
    rows = [
        _make_session(session_id="s1", agent_id="agt-1", status="ready"),
        _make_session(session_id="s2", agent_id="agt-2", status="dead"),
    ]
    prisma = _make_prisma(sessions=rows)
    with patch("litellm.proxy.proxy_server.prisma_client", prisma):
        resp = client.get("/v1/managed_agents/sessions")
    assert resp.status_code == 200
    body = resp.json()
    assert [r["id"] for r in body] == ["s1", "s2"]
    # Admin caller: no owner filter
    _, kwargs = prisma.db.litellm_managedagentsessiontable.find_many.call_args
    assert kwargs["where"] == {}
    assert kwargs["order"] == {"created_at": "desc"}


def test_list_sessions_filters_by_agent_id(app_factory, admin):
    client = app_factory(admin)
    rows = [_make_session(session_id="s1", agent_id="agt-1", status="ready")]
    prisma = _make_prisma(sessions=rows)
    with patch("litellm.proxy.proxy_server.prisma_client", prisma):
        resp = client.get("/v1/managed_agents/sessions?agent_id=agt-1")
    assert resp.status_code == 200
    assert [r["agent_id"] for r in resp.json()] == ["agt-1"]
    _, kwargs = prisma.db.litellm_managedagentsessiontable.find_many.call_args
    assert kwargs["where"] == {"agent_id": "agt-1"}


def test_list_sessions_empty(app_factory, user):
    client = app_factory(user)
    prisma = _make_prisma(sessions=[])
    with patch("litellm.proxy.proxy_server.prisma_client", prisma):
        resp = client.get("/v1/managed_agents/sessions")
    assert resp.status_code == 200
    assert resp.json() == []


def test_list_sessions_filters_by_owner_for_non_admin(app_factory, user):
    client = app_factory(user)
    prisma = _make_prisma(sessions=[])
    with patch("litellm.proxy.proxy_server.prisma_client", prisma):
        resp = client.get("/v1/managed_agents/sessions")
    assert resp.status_code == 200
    _, kwargs = prisma.db.litellm_managedagentsessiontable.find_many.call_args
    assert kwargs["where"] == {"created_by": "u1"}


# ---------------------------------------------------------------------------
# Ownership / authorization on get and delete
# ---------------------------------------------------------------------------


def test_get_session_returns_404_for_non_owner(app_factory, other_user):
    client = app_factory(other_user)
    sess = _make_session(
        session_id="sess-9",
        status="ready",
        sandbox_url="http://1.2.3.4:4096",
        created_by="u1",
    )
    prisma = _make_prisma(session=sess)
    with patch("litellm.proxy.proxy_server.prisma_client", prisma):
        resp = client.get("/v1/managed_agents/sessions/sess-9")
    assert resp.status_code == 404


def test_get_session_visible_to_admin(app_factory, admin):
    client = app_factory(admin)
    sess = _make_session(
        session_id="sess-9",
        status="ready",
        sandbox_url="http://1.2.3.4:4096",
        created_by="u1",
    )
    prisma = _make_prisma(session=sess)
    with patch("litellm.proxy.proxy_server.prisma_client", prisma):
        resp = client.get("/v1/managed_agents/sessions/sess-9")
    assert resp.status_code == 200


def test_delete_session_returns_404_for_non_owner(app_factory, other_user):
    client = app_factory(other_user)
    sess = _make_session(
        session_id="sess-9",
        status="ready",
        task_arn="arn:task/9",
        created_by="u1",
    )
    prisma = _make_prisma(session=sess)
    stop_mock = AsyncMock(return_value=None)
    with (
        patch("litellm.proxy.proxy_server.prisma_client", prisma),
        patch(
            "litellm.proxy.managed_agents_endpoints.endpoints_sessions.stop_session_task",
            new=stop_mock,
        ),
        patch(
            "litellm.proxy.managed_agents_endpoints.config_loader.MANAGED_AGENTS_CONFIG",
            SimpleNamespace(
                aws_region="us-west-2",
                aws=SimpleNamespace(cluster=None),
            ),
        ),
    ):
        resp = client.delete("/v1/managed_agents/sessions/sess-9")
    assert resp.status_code == 404
    stop_mock.assert_not_called()
