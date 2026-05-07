"""Unit tests for the messaging endpoints (LIT-2920).

Covers ``POST /v2/sessions/:id/messages`` and ``GET /v2/sessions/:id/messages``.

Auth pattern: FastAPI ``app.dependency_overrides`` for ``user_api_key_auth``.
Adapter pattern: monkeypatch ``get_adapter`` so the handler resolves to a
mock without touching the network.
DB pattern: stub ``prisma_client`` with in-memory tables (mirrors the
Wave-2b agent test scaffolding).

The router is mounted onto the proxy ``app`` inside a fixture so this test
file does not require Wave 3's wiring in ``proxy_server.py`` to be present.
"""

import os
import sys
import types
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

import litellm.proxy.proxy_server as ps
from litellm.managed_agents.adapters.base import (
    SandboxBadGatewayError,
    SandboxUnreachableError,
)
from litellm.proxy.managed_agents_endpoints.messages import router as messages_router
from litellm.managed_agents.types import MessageRow
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.proxy_server import app

sys.path.insert(0, os.path.abspath("../../../"))


# ---------------------------------------------------------------------------
# Shared fixtures and helpers
# ---------------------------------------------------------------------------

USER_ID = "user_xyz"
SESSION_ID = "ses_test"
AGENT_ID = "agt_test"
SANDBOX_URL = "http://127.0.0.1:1234"
OC_SID = "oc_sid_xxx"
DEFAULT_AGENT_MODEL = "anthropic/claude-opus-4"


def _ensure_router_mounted() -> None:
    """Mount the messages router onto ``app`` exactly once.

    Wave 3 owns the real registration in ``proxy_server.py``. For tests we
    mount it here so we can hit the endpoints via ``TestClient``.
    """
    paths = {getattr(r, "path", None) for r in app.router.routes}
    if "/v2/sessions/{session_id}/messages" not in paths:
        app.include_router(messages_router)


def _make_session_row(
    *,
    session_id: str = SESSION_ID,
    agent_id: str = AGENT_ID,
    status: str = "ready",
    sandbox_url: Optional[str] = SANDBOX_URL,
    opencode_session_id: Optional[str] = OC_SID,
    sandbox_type: str = "opencode",
    created_by: str = USER_ID,
) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {}
    if opencode_session_id is not None:
        metadata["opencode_session_id"] = opencode_session_id
    return {
        "id": session_id,
        "agent_id": agent_id,
        "sandbox_type": sandbox_type,
        "sandbox_size": "small",
        "sandbox_timeout_minutes": 60,
        "sandbox_idle_timeout_minutes": 10,
        "sandbox_image": None,
        "sandbox_url": sandbox_url,
        "sandbox_metadata": metadata,
        "status": status,
        "repos": [],
        "env_vars": {},
        "created_by": created_by,
        "created_at": datetime.now(timezone.utc),
        "updated_at": datetime.now(timezone.utc),
        "terminated_at": None,
    }


def _make_agent_row(
    *,
    agent_id: str = AGENT_ID,
    model: Optional[str] = DEFAULT_AGENT_MODEL,
    created_by: str = USER_ID,
) -> Dict[str, Any]:
    config: Dict[str, Any] = {
        "system_prompt": "You are a code reviewer.",
        "tools": ["read"],
        "litellm_api_key": "sk-test",
        "litellm_base_url": "http://localhost:4000",
    }
    if model is not None:
        config["model"] = model
    return {
        "id": agent_id,
        "name": "code-reviewer",
        "config": config,
        "created_by": created_by,
        "created_at": datetime.now(timezone.utc),
        "updated_at": datetime.now(timezone.utc),
    }


class _FakeSessionTable:
    def __init__(self, rows: Optional[List[Dict[str, Any]]] = None) -> None:
        self.rows: List[Dict[str, Any]] = list(rows or [])
        self.find_first = AsyncMock(side_effect=self._find_first)

    async def _find_first(
        self, *, where: Dict[str, Any]
    ) -> Optional[types.SimpleNamespace]:
        for row in self.rows:
            if all(row.get(k) == v for k, v in where.items()):
                return types.SimpleNamespace(model_dump=lambda r=row: dict(r))
        return None


class _FakeAgentTable:
    def __init__(self, rows: Optional[List[Dict[str, Any]]] = None) -> None:
        self.rows: List[Dict[str, Any]] = list(rows or [])
        self.find_first = AsyncMock(side_effect=self._find_first)

    async def _find_first(
        self, *, where: Dict[str, Any]
    ) -> Optional[types.SimpleNamespace]:
        for row in self.rows:
            if all(row.get(k) == v for k, v in where.items()):
                return types.SimpleNamespace(model_dump=lambda r=row: dict(r))
        return None


def _build_fake_prisma(
    sessions: Optional[List[Dict[str, Any]]] = None,
    agents: Optional[List[Dict[str, Any]]] = None,
) -> MagicMock:
    fake_db = types.SimpleNamespace(
        litellm_managedagentsession=_FakeSessionTable(sessions),
        litellm_managedagent=_FakeAgentTable(agents),
    )
    fake_prisma = MagicMock()
    fake_prisma.db = fake_db
    return fake_prisma


def _override_auth(user_id: str = USER_ID) -> UserAPIKeyAuth:
    user_key = UserAPIKeyAuth(api_key="sk-test", user_id=user_id)
    app.dependency_overrides[user_api_key_auth] = lambda: user_key
    return user_key


@pytest.fixture(autouse=True)
def _mount_router_and_clear_overrides():
    _ensure_router_mounted()
    yield
    app.dependency_overrides.clear()


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture
def mock_adapter(monkeypatch):
    """Monkeypatch ``get_adapter`` to return a configurable AsyncMock.

    The same adapter instance is returned by ``get_adapter`` for every
    call inside the handler — ``send_message``, ``list_messages``, and
    ``abort`` are exposed as AsyncMocks so each test can configure side
    effects / return values independently.
    """
    adapter = MagicMock()
    adapter.send_message = AsyncMock(return_value=None)
    adapter.list_messages = AsyncMock(return_value=[])
    adapter.abort = AsyncMock(return_value=None)
    monkeypatch.setattr(
        "litellm.proxy.managed_agents_endpoints.messages.get_adapter",
        lambda sandbox_type: adapter,
    )
    return adapter


# ---------------------------------------------------------------------------
# POST /v2/sessions/:id/messages
# ---------------------------------------------------------------------------


class TestSendMessage:
    def test_ready_session_returns_202_with_user_message_row(
        self, monkeypatch, client, mock_adapter
    ):
        """Happy path: ready session → 202, adapter.send_message called with
        the right args, and the response is a synthesized user MessageRow.
        """
        fake_prisma = _build_fake_prisma(
            sessions=[_make_session_row()],
            agents=[_make_agent_row()],
        )
        monkeypatch.setattr(ps, "prisma_client", fake_prisma)
        _override_auth()

        resp = client.post(
            f"/v2/sessions/{SESSION_ID}/messages",
            json={"content": "hello", "model": "anthropic/claude-opus-4"},
            headers={"Authorization": "Bearer sk-test"},
        )

        assert resp.status_code == 202
        body = resp.json()
        assert body["session_id"] == SESSION_ID
        assert body["role"] == "user"
        assert body["content"] == "hello"
        assert body["status"] == "in_progress"
        assert body["model"] == "anthropic/claude-opus-4"
        assert body["completed_at"] is None
        assert body["id"].startswith("msg_")

        mock_adapter.send_message.assert_awaited_once_with(
            SANDBOX_URL,
            OC_SID,
            "hello",
            "anthropic/claude-opus-4",
        )

    def test_provisioning_session_returns_503(self, monkeypatch, client, mock_adapter):
        fake_prisma = _build_fake_prisma(
            sessions=[_make_session_row(status="provisioning")],
        )
        monkeypatch.setattr(ps, "prisma_client", fake_prisma)
        _override_auth()

        resp = client.post(
            f"/v2/sessions/{SESSION_ID}/messages",
            json={"content": "hello"},
            headers={"Authorization": "Bearer sk-test"},
        )

        assert resp.status_code == 503
        assert resp.json()["detail"] == "Session not ready"
        # contract §7: provisioning carries Retry-After: 5
        assert resp.headers.get("retry-after") == "5"
        mock_adapter.send_message.assert_not_awaited()

    def test_terminated_session_returns_404(self, monkeypatch, client, mock_adapter):
        fake_prisma = _build_fake_prisma(
            sessions=[_make_session_row(status="terminated")],
        )
        monkeypatch.setattr(ps, "prisma_client", fake_prisma)
        _override_auth()

        resp = client.post(
            f"/v2/sessions/{SESSION_ID}/messages",
            json={"content": "hello"},
            headers={"Authorization": "Bearer sk-test"},
        )

        assert resp.status_code == 404
        assert SESSION_ID in resp.json()["detail"]
        mock_adapter.send_message.assert_not_awaited()

    def test_missing_session_returns_404(self, monkeypatch, client, mock_adapter):
        fake_prisma = _build_fake_prisma(sessions=[])
        monkeypatch.setattr(ps, "prisma_client", fake_prisma)
        _override_auth()

        resp = client.post(
            f"/v2/sessions/{SESSION_ID}/messages",
            json={"content": "hello"},
            headers={"Authorization": "Bearer sk-test"},
        )

        assert resp.status_code == 404
        mock_adapter.send_message.assert_not_awaited()

    def test_sandbox_unreachable_returns_504(self, monkeypatch, client, mock_adapter):
        fake_prisma = _build_fake_prisma(
            sessions=[_make_session_row()],
            agents=[_make_agent_row()],
        )
        monkeypatch.setattr(ps, "prisma_client", fake_prisma)
        _override_auth()

        mock_adapter.send_message.side_effect = SandboxUnreachableError("boom")

        resp = client.post(
            f"/v2/sessions/{SESSION_ID}/messages",
            json={"content": "hello"},
            headers={"Authorization": "Bearer sk-test"},
        )

        assert resp.status_code == 504
        assert resp.json()["detail"] == {"error": "Sandbox unreachable"}

    def test_sandbox_bad_gateway_returns_502(self, monkeypatch, client, mock_adapter):
        fake_prisma = _build_fake_prisma(
            sessions=[_make_session_row()],
            agents=[_make_agent_row()],
        )
        monkeypatch.setattr(ps, "prisma_client", fake_prisma)
        _override_auth()

        mock_adapter.send_message.side_effect = SandboxBadGatewayError("malformed")

        resp = client.post(
            f"/v2/sessions/{SESSION_ID}/messages",
            json={"content": "hello"},
            headers={"Authorization": "Bearer sk-test"},
        )

        assert resp.status_code == 502
        assert resp.json()["detail"] == {"error": "Bad gateway"}

    def test_missing_content_returns_422(self, monkeypatch, client, mock_adapter):
        fake_prisma = _build_fake_prisma(
            sessions=[_make_session_row()],
            agents=[_make_agent_row()],
        )
        monkeypatch.setattr(ps, "prisma_client", fake_prisma)
        _override_auth()

        resp = client.post(
            f"/v2/sessions/{SESSION_ID}/messages",
            json={},
            headers={"Authorization": "Bearer sk-test"},
        )

        assert resp.status_code == 422
        mock_adapter.send_message.assert_not_awaited()

    def test_optional_model_falls_back_to_agent_config(
        self, monkeypatch, client, mock_adapter
    ):
        """Omitting `model` in the request → handler resolves it from
        agent.config["model"] before calling the adapter.
        """
        fake_prisma = _build_fake_prisma(
            sessions=[_make_session_row()],
            agents=[_make_agent_row(model="anthropic/claude-sonnet-4")],
        )
        monkeypatch.setattr(ps, "prisma_client", fake_prisma)
        _override_auth()

        resp = client.post(
            f"/v2/sessions/{SESSION_ID}/messages",
            json={"content": "hello"},
            headers={"Authorization": "Bearer sk-test"},
        )

        assert resp.status_code == 202
        assert resp.json()["model"] == "anthropic/claude-sonnet-4"
        mock_adapter.send_message.assert_awaited_once_with(
            SANDBOX_URL,
            OC_SID,
            "hello",
            "anthropic/claude-sonnet-4",
        )

    def test_missing_auth_returns_401(self, monkeypatch, client, mock_adapter):
        """Without an ``Authorization`` header and a master key set, real auth
        rejects. The proxy's ``user_api_key_auth`` is permissive when
        ``master_key is None`` — set a master key + drop the override so the
        real chain runs.
        """
        fake_prisma = _build_fake_prisma(
            sessions=[_make_session_row()],
            agents=[_make_agent_row()],
        )
        monkeypatch.setattr(ps, "prisma_client", fake_prisma)
        monkeypatch.setattr(ps, "master_key", "sk-test-master")
        # Do NOT override auth — exercise the real dependency.
        app.dependency_overrides.pop(user_api_key_auth, None)

        resp = client.post(
            f"/v2/sessions/{SESSION_ID}/messages",
            json={"content": "hello"},
        )

        # The real auth chain rejects unauthenticated calls. Proxy maps the
        # raised exception to ProxyException → 401 in most versions, but some
        # 4xx is fine — what matters is that it's an auth failure, NOT a
        # successful 2xx, and the adapter was never invoked.
        assert resp.status_code in (400, 401, 403), resp.text
        mock_adapter.send_message.assert_not_awaited()


# ---------------------------------------------------------------------------
# GET /v2/sessions/:id/messages
# ---------------------------------------------------------------------------


def _make_message_row(
    *,
    msg_id: str,
    role: str,
    content: str,
    status: str = "completed",
) -> MessageRow:
    return MessageRow(
        id=msg_id,
        session_id=SESSION_ID,
        role=role,  # type: ignore[arg-type]
        content=content,
        model=DEFAULT_AGENT_MODEL,
        status=status,  # type: ignore[arg-type]
        created_at=datetime.now(timezone.utc),
        completed_at=datetime.now(timezone.utc) if status == "completed" else None,
    )


class TestListMessages:
    def test_returns_messages_wrapped_in_message_list_shape(
        self, monkeypatch, client, mock_adapter
    ):
        fake_prisma = _build_fake_prisma(sessions=[_make_session_row()])
        monkeypatch.setattr(ps, "prisma_client", fake_prisma)
        _override_auth()

        msgs = [
            _make_message_row(msg_id="msg_1", role="user", content="hi"),
            _make_message_row(msg_id="msg_2", role="assistant", content="hello!"),
        ]
        mock_adapter.list_messages.return_value = msgs

        resp = client.get(
            f"/v2/sessions/{SESSION_ID}/messages",
            headers={"Authorization": "Bearer sk-test"},
        )

        assert resp.status_code == 200
        body = resp.json()
        assert "data" in body
        assert len(body["data"]) == 2
        assert body["data"][0]["id"] == "msg_1"
        assert body["data"][1]["id"] == "msg_2"
        assert body["next_cursor"] is None
        assert body["has_more"] is False

        mock_adapter.list_messages.assert_awaited_once_with(
            SANDBOX_URL,
            OC_SID,
            SESSION_ID,
            50,
        )

    def test_role_filter_narrows_results(self, monkeypatch, client, mock_adapter):
        fake_prisma = _build_fake_prisma(sessions=[_make_session_row()])
        monkeypatch.setattr(ps, "prisma_client", fake_prisma)
        _override_auth()

        mock_adapter.list_messages.return_value = [
            _make_message_row(msg_id="msg_u1", role="user", content="hi"),
            _make_message_row(msg_id="msg_a1", role="assistant", content="hello!"),
            _make_message_row(msg_id="msg_u2", role="user", content="more"),
        ]

        resp = client.get(
            f"/v2/sessions/{SESSION_ID}/messages?role=user",
            headers={"Authorization": "Bearer sk-test"},
        )

        assert resp.status_code == 200
        body = resp.json()
        assert len(body["data"]) == 2
        assert {m["id"] for m in body["data"]} == {"msg_u1", "msg_u2"}
        assert all(m["role"] == "user" for m in body["data"])

    def test_empty_session_returns_empty_data_array(
        self, monkeypatch, client, mock_adapter
    ):
        fake_prisma = _build_fake_prisma(sessions=[_make_session_row()])
        monkeypatch.setattr(ps, "prisma_client", fake_prisma)
        _override_auth()

        mock_adapter.list_messages.return_value = []

        resp = client.get(
            f"/v2/sessions/{SESSION_ID}/messages",
            headers={"Authorization": "Bearer sk-test"},
        )

        assert resp.status_code == 200
        body = resp.json()
        assert body["data"] == []
        assert body["next_cursor"] is None
        assert body["has_more"] is False

    def test_provisioning_session_returns_503(self, monkeypatch, client, mock_adapter):
        fake_prisma = _build_fake_prisma(
            sessions=[_make_session_row(status="provisioning")],
        )
        monkeypatch.setattr(ps, "prisma_client", fake_prisma)
        _override_auth()

        resp = client.get(
            f"/v2/sessions/{SESSION_ID}/messages",
            headers={"Authorization": "Bearer sk-test"},
        )

        assert resp.status_code == 503
        assert resp.headers.get("retry-after") == "5"
        mock_adapter.list_messages.assert_not_awaited()

    def test_missing_session_returns_404(self, monkeypatch, client, mock_adapter):
        fake_prisma = _build_fake_prisma(sessions=[])
        monkeypatch.setattr(ps, "prisma_client", fake_prisma)
        _override_auth()

        resp = client.get(
            f"/v2/sessions/{SESSION_ID}/messages",
            headers={"Authorization": "Bearer sk-test"},
        )

        assert resp.status_code == 404
        mock_adapter.list_messages.assert_not_awaited()

    def test_sandbox_unreachable_returns_504(self, monkeypatch, client, mock_adapter):
        fake_prisma = _build_fake_prisma(sessions=[_make_session_row()])
        monkeypatch.setattr(ps, "prisma_client", fake_prisma)
        _override_auth()

        mock_adapter.list_messages.side_effect = SandboxUnreachableError("boom")

        resp = client.get(
            f"/v2/sessions/{SESSION_ID}/messages",
            headers={"Authorization": "Bearer sk-test"},
        )

        assert resp.status_code == 504
        assert resp.json()["detail"] == {"error": "Sandbox unreachable"}

    def test_missing_auth_returns_401(self, monkeypatch, client, mock_adapter):
        """Without an ``Authorization`` header and a master key set, real auth
        rejects. See note on the corresponding ``TestSendMessage`` test.
        """
        fake_prisma = _build_fake_prisma(sessions=[_make_session_row()])
        monkeypatch.setattr(ps, "prisma_client", fake_prisma)
        monkeypatch.setattr(ps, "master_key", "sk-test-master")
        # Do NOT override auth — exercise the real dependency.
        app.dependency_overrides.pop(user_api_key_auth, None)

        resp = client.get(f"/v2/sessions/{SESSION_ID}/messages")

        assert resp.status_code in (400, 401, 403), resp.text
        mock_adapter.list_messages.assert_not_awaited()


# ---------------------------------------------------------------------------
# POST /v2/sessions/:id/abort
# ---------------------------------------------------------------------------


class TestAbortSession:
    def test_ready_session_aborts_and_returns_ok(
        self, monkeypatch, client, mock_adapter
    ):
        """Happy path: ready session → 200, adapter.abort called with the
        session's sandbox_url + opencode_session_id, returns
        ``{"id": session_id, "aborted": True}``.
        """
        fake_prisma = _build_fake_prisma(sessions=[_make_session_row()])
        monkeypatch.setattr(ps, "prisma_client", fake_prisma)
        _override_auth()

        resp = client.post(
            f"/v2/sessions/{SESSION_ID}/abort",
            headers={"Authorization": "Bearer sk-test"},
        )

        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body == {"id": SESSION_ID, "aborted": True}

        mock_adapter.abort.assert_awaited_once_with(SANDBOX_URL, OC_SID)

    def test_missing_session_returns_404(self, monkeypatch, client, mock_adapter):
        fake_prisma = _build_fake_prisma(sessions=[])
        monkeypatch.setattr(ps, "prisma_client", fake_prisma)
        _override_auth()

        resp = client.post(
            f"/v2/sessions/{SESSION_ID}/abort",
            headers={"Authorization": "Bearer sk-test"},
        )

        assert resp.status_code == 404
        assert SESSION_ID in resp.json()["detail"]
        mock_adapter.abort.assert_not_awaited()

    def test_provisioning_session_returns_503(self, monkeypatch, client, mock_adapter):
        fake_prisma = _build_fake_prisma(
            sessions=[_make_session_row(status="provisioning")],
        )
        monkeypatch.setattr(ps, "prisma_client", fake_prisma)
        _override_auth()

        resp = client.post(
            f"/v2/sessions/{SESSION_ID}/abort",
            headers={"Authorization": "Bearer sk-test"},
        )

        assert resp.status_code == 503
        assert resp.json()["detail"] == "Session not ready"
        assert resp.headers.get("retry-after") == "5"
        mock_adapter.abort.assert_not_awaited()

    def test_terminated_session_returns_404(self, monkeypatch, client, mock_adapter):
        fake_prisma = _build_fake_prisma(
            sessions=[_make_session_row(status="terminated")],
        )
        monkeypatch.setattr(ps, "prisma_client", fake_prisma)
        _override_auth()

        resp = client.post(
            f"/v2/sessions/{SESSION_ID}/abort",
            headers={"Authorization": "Bearer sk-test"},
        )

        assert resp.status_code == 404
        mock_adapter.abort.assert_not_awaited()

    def test_sandbox_unreachable_returns_504(self, monkeypatch, client, mock_adapter):
        fake_prisma = _build_fake_prisma(sessions=[_make_session_row()])
        monkeypatch.setattr(ps, "prisma_client", fake_prisma)
        _override_auth()

        mock_adapter.abort.side_effect = SandboxUnreachableError("boom")

        resp = client.post(
            f"/v2/sessions/{SESSION_ID}/abort",
            headers={"Authorization": "Bearer sk-test"},
        )

        assert resp.status_code == 504
        assert resp.json()["detail"] == {"error": "Sandbox unreachable"}

    def test_sandbox_bad_gateway_returns_502(self, monkeypatch, client, mock_adapter):
        fake_prisma = _build_fake_prisma(sessions=[_make_session_row()])
        monkeypatch.setattr(ps, "prisma_client", fake_prisma)
        _override_auth()

        mock_adapter.abort.side_effect = SandboxBadGatewayError("malformed")

        resp = client.post(
            f"/v2/sessions/{SESSION_ID}/abort",
            headers={"Authorization": "Bearer sk-test"},
        )

        assert resp.status_code == 502
        assert resp.json()["detail"] == {"error": "Bad gateway"}
