"""
Unit tests for workflow management endpoints (/v1/workflows/runs/*).
Uses FastAPI TestClient with a mocked prisma_client.
"""

import os
import sys
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
from prisma.errors import UniqueViolationError

sys.path.insert(0, os.path.abspath("../../.."))

from litellm.proxy.management_endpoints.workflow_management_endpoints import router


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_run(
    run_id: str = "run-1",
    session_id: str = "sess-1",
    workflow_type: str = "shin-builder",
    status: str = "pending",
    created_by: Any = "tok-test",
) -> MagicMock:
    obj = MagicMock()
    obj.run_id = run_id
    obj.session_id = session_id
    obj.workflow_type = workflow_type
    obj.status = status
    obj.created_by = created_by
    obj.created_at = datetime.now(timezone.utc)
    obj.updated_at = datetime.now(timezone.utc)
    obj.input = None
    obj.output = None
    obj.metadata = None
    return obj


def _make_event(
    event_id: str = "evt-1",
    run_id: str = "run-1",
    event_type: str = "step.started",
    step_name: str = "grill",
    sequence_number: int = 0,
) -> MagicMock:
    obj = MagicMock()
    obj.event_id = event_id
    obj.run_id = run_id
    obj.event_type = event_type
    obj.step_name = step_name
    obj.sequence_number = sequence_number
    obj.data = None
    obj.created_at = datetime.now(timezone.utc)
    return obj


def _make_message(
    message_id: str = "msg-1",
    run_id: str = "run-1",
    role: str = "user",
    content: str = "hello",
    sequence_number: int = 0,
) -> MagicMock:
    obj = MagicMock()
    obj.message_id = message_id
    obj.run_id = run_id
    obj.role = role
    obj.content = content
    obj.sequence_number = sequence_number
    obj.session_id = None
    obj.created_at = datetime.now(timezone.utc)
    return obj


def _make_tx(event_return=None, run_return=None, msg_return=None) -> MagicMock:
    """Build an async context-manager mock for prisma_client.db.tx()."""
    tx = MagicMock()
    tx.litellm_workflowevent = MagicMock()
    tx.litellm_workflowevent.create = AsyncMock(
        return_value=event_return or _make_event()
    )
    tx.litellm_workflowrun = MagicMock()
    tx.litellm_workflowrun.update = AsyncMock(return_value=run_return or _make_run())
    tx.litellm_workflowmessage = MagicMock()
    tx.litellm_workflowmessage.create = AsyncMock(
        return_value=msg_return or _make_message()
    )
    tx.__aenter__ = AsyncMock(return_value=tx)
    tx.__aexit__ = AsyncMock(return_value=False)
    return tx


def _make_prisma_client() -> MagicMock:
    client = MagicMock()
    client.db = MagicMock()
    client.db.litellm_workflowrun = MagicMock()
    client.db.litellm_workflowevent = MagicMock()
    client.db.litellm_workflowmessage = MagicMock()
    # default tx() returns a no-op transaction
    client.db.tx = MagicMock(return_value=_make_tx())
    return client


def _make_app() -> FastAPI:
    app = FastAPI()
    app.include_router(router)
    return app


def _override_auth() -> Any:
    from litellm.proxy._types import UserAPIKeyAuth

    auth = UserAPIKeyAuth(api_key="sk-test", user_id="admin")
    auth.token = "tok-test"
    return auth


def _override_auth_admin() -> Any:
    from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth

    auth = UserAPIKeyAuth(api_key="sk-master")
    auth.user_role = LitellmUserRoles.PROXY_ADMIN  # type: ignore[assignment]
    return auth


def _override_auth_user_with_token(token: str = "tok-abc") -> Any:
    """Return a non-admin caller whose hashed token equals `token`."""
    from litellm.proxy._types import UserAPIKeyAuth

    auth = UserAPIKeyAuth(api_key="sk-user", user_id="user-1")
    auth.token = token  # override the computed hash with a predictable value
    return auth


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCreateWorkflowRun:
    def setup_method(self):
        from litellm.proxy.auth.user_api_key_auth import user_api_key_auth

        self._prisma = _make_prisma_client()
        app = _make_app()
        app.dependency_overrides[user_api_key_auth] = _override_auth
        self.client = TestClient(app, raise_server_exceptions=True)

    @patch("litellm.proxy.proxy_server.prisma_client")
    def test_create_returns_run(self, mock_pc):
        mock_pc.db = self._prisma.db
        self._prisma.db.litellm_workflowrun.create = AsyncMock(return_value=_make_run())

        resp = self.client.post(
            "/v1/workflows/runs",
            json={"workflow_type": "shin-builder"},
        )
        assert resp.status_code == 200
        self._prisma.db.litellm_workflowrun.create.assert_awaited_once()

    @patch("litellm.proxy.proxy_server.prisma_client", None)
    def test_create_500_when_no_db(self):
        resp = self.client.post(
            "/v1/workflows/runs",
            json={"workflow_type": "shin-builder"},
        )
        assert resp.status_code == 500


class TestListWorkflowRuns:
    def setup_method(self):
        from litellm.proxy.auth.user_api_key_auth import user_api_key_auth

        self._prisma = _make_prisma_client()
        app = _make_app()
        app.dependency_overrides[user_api_key_auth] = _override_auth
        self.client = TestClient(app, raise_server_exceptions=True)

    @patch("litellm.proxy.proxy_server.prisma_client")
    def test_list_returns_runs(self, mock_pc):
        mock_pc.db = self._prisma.db
        self._prisma.db.litellm_workflowrun.find_many = AsyncMock(
            return_value=[_make_run()]
        )

        resp = self.client.get("/v1/workflows/runs")
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 1

    @patch("litellm.proxy.proxy_server.prisma_client")
    def test_list_filters_by_status(self, mock_pc):
        mock_pc.db = self._prisma.db
        self._prisma.db.litellm_workflowrun.find_many = AsyncMock(return_value=[])

        resp = self.client.get("/v1/workflows/runs?status=running")
        assert resp.status_code == 200
        call_kwargs = self._prisma.db.litellm_workflowrun.find_many.call_args[1]
        assert call_kwargs["where"]["status"] == "running"

    @patch("litellm.proxy.proxy_server.prisma_client")
    def test_list_filters_by_multiple_statuses(self, mock_pc):
        mock_pc.db = self._prisma.db
        self._prisma.db.litellm_workflowrun.find_many = AsyncMock(return_value=[])

        resp = self.client.get("/v1/workflows/runs?status=running,paused")
        assert resp.status_code == 200
        call_kwargs = self._prisma.db.litellm_workflowrun.find_many.call_args[1]
        assert call_kwargs["where"]["status"] == {"in": ["running", "paused"]}


class TestGetWorkflowRun:
    def setup_method(self):
        from litellm.proxy.auth.user_api_key_auth import user_api_key_auth

        self._prisma = _make_prisma_client()
        app = _make_app()
        app.dependency_overrides[user_api_key_auth] = _override_auth
        self.client = TestClient(app, raise_server_exceptions=True)

    @patch("litellm.proxy.proxy_server.prisma_client")
    def test_get_existing_run(self, mock_pc):
        mock_pc.db = self._prisma.db
        self._prisma.db.litellm_workflowrun.find_unique = AsyncMock(
            return_value=_make_run()
        )

        resp = self.client.get("/v1/workflows/runs/run-1")
        assert resp.status_code == 200

    @patch("litellm.proxy.proxy_server.prisma_client")
    def test_get_missing_run_returns_404(self, mock_pc):
        mock_pc.db = self._prisma.db
        self._prisma.db.litellm_workflowrun.find_unique = AsyncMock(return_value=None)

        resp = self.client.get("/v1/workflows/runs/nonexistent")
        assert resp.status_code == 404


class TestUpdateWorkflowRun:
    def setup_method(self):
        from litellm.proxy.auth.user_api_key_auth import user_api_key_auth

        self._prisma = _make_prisma_client()
        app = _make_app()
        app.dependency_overrides[user_api_key_auth] = _override_auth
        self.client = TestClient(app, raise_server_exceptions=True)

    @patch("litellm.proxy.proxy_server.prisma_client")
    def test_update_status(self, mock_pc):
        mock_pc.db = self._prisma.db
        self._prisma.db.litellm_workflowrun.find_unique = AsyncMock(
            return_value=_make_run()
        )
        updated = _make_run(status="completed")
        self._prisma.db.litellm_workflowrun.update = AsyncMock(return_value=updated)

        resp = self.client.patch(
            "/v1/workflows/runs/run-1", json={"status": "completed"}
        )
        assert resp.status_code == 200
        self._prisma.db.litellm_workflowrun.update.assert_awaited_once()

    @patch("litellm.proxy.proxy_server.prisma_client")
    def test_update_no_fields_returns_400(self, mock_pc):
        mock_pc.db = self._prisma.db
        resp = self.client.patch("/v1/workflows/runs/run-1", json={})
        assert resp.status_code == 400


class TestAppendWorkflowEvent:
    def setup_method(self):
        from litellm.proxy.auth.user_api_key_auth import user_api_key_auth

        self._prisma = _make_prisma_client()
        app = _make_app()
        app.dependency_overrides[user_api_key_auth] = _override_auth
        self.client = TestClient(app, raise_server_exceptions=True)

    @patch("litellm.proxy.proxy_server.prisma_client")
    def test_append_event_updates_run_status(self, mock_pc):
        mock_pc.db = self._prisma.db
        # _require_run check
        self._prisma.db.litellm_workflowrun.find_unique = AsyncMock(
            return_value=_make_run()
        )
        self._prisma.db.litellm_workflowevent.find_many = AsyncMock(return_value=[])
        tx = _make_tx(
            event_return=_make_event(), run_return=_make_run(status="running")
        )
        self._prisma.db.tx = MagicMock(return_value=tx)

        resp = self.client.post(
            "/v1/workflows/runs/run-1/events",
            json={"event_type": "step.started", "step_name": "grill"},
        )
        assert resp.status_code == 200
        # run status updated inside tx
        tx.litellm_workflowrun.update.assert_awaited_once()
        update_call = tx.litellm_workflowrun.update.call_args[1]
        assert update_call["data"]["status"] == "running"

    @patch("litellm.proxy.proxy_server.prisma_client")
    def test_append_event_no_status_update_for_unknown_type(self, mock_pc):
        mock_pc.db = self._prisma.db
        self._prisma.db.litellm_workflowrun.find_unique = AsyncMock(
            return_value=_make_run()
        )
        self._prisma.db.litellm_workflowevent.find_many = AsyncMock(return_value=[])
        tx = _make_tx(event_return=_make_event(event_type="custom.event"))
        self._prisma.db.tx = MagicMock(return_value=tx)

        resp = self.client.post(
            "/v1/workflows/runs/run-1/events",
            json={"event_type": "custom.event", "step_name": "grill"},
        )
        assert resp.status_code == 200
        # no status update inside tx for unknown event_type
        tx.litellm_workflowrun.update.assert_not_awaited()

    @patch("litellm.proxy.proxy_server.prisma_client")
    def test_sequence_number_increments(self, mock_pc):
        mock_pc.db = self._prisma.db
        self._prisma.db.litellm_workflowrun.find_unique = AsyncMock(
            return_value=_make_run()
        )
        existing = _make_event(sequence_number=4)
        self._prisma.db.litellm_workflowevent.find_many = AsyncMock(
            return_value=[existing]
        )
        tx = _make_tx(event_return=_make_event(sequence_number=5))
        self._prisma.db.tx = MagicMock(return_value=tx)

        self.client.post(
            "/v1/workflows/runs/run-1/events",
            json={"event_type": "step.started", "step_name": "plan"},
        )
        create_call = tx.litellm_workflowevent.create.call_args[1]
        assert create_call["data"]["sequence_number"] == 5

    @patch("litellm.proxy.proxy_server.prisma_client")
    def test_unknown_run_id_returns_404(self, mock_pc):
        mock_pc.db = self._prisma.db
        self._prisma.db.litellm_workflowrun.find_unique = AsyncMock(return_value=None)

        resp = self.client.post(
            "/v1/workflows/runs/nonexistent/events",
            json={"event_type": "step.started", "step_name": "grill"},
        )
        assert resp.status_code == 404

    @patch("litellm.proxy.proxy_server.prisma_client")
    def test_sequence_collision_retries_and_succeeds(self, mock_pc):
        """UniqueViolationError on first attempt triggers retry; second attempt succeeds."""
        mock_pc.db = self._prisma.db
        self._prisma.db.litellm_workflowrun.find_unique = AsyncMock(
            return_value=_make_run()
        )
        self._prisma.db.litellm_workflowevent.find_many = AsyncMock(return_value=[])

        # First tx raises UniqueViolationError; second succeeds.
        tx_fail = _make_tx()
        tx_fail.__aenter__ = AsyncMock(return_value=tx_fail)
        tx_fail.litellm_workflowevent.create = AsyncMock(
            side_effect=UniqueViolationError(
                {"user_facing_error": {"message": "unique"}}
            )
        )
        tx_fail.__aexit__ = AsyncMock(return_value=False)

        tx_ok = _make_tx(event_return=_make_event(sequence_number=1))

        self._prisma.db.tx = MagicMock(side_effect=[tx_fail, tx_ok])

        resp = self.client.post(
            "/v1/workflows/runs/run-1/events",
            json={"event_type": "step.started", "step_name": "grill"},
        )
        assert resp.status_code == 200


class TestWorkflowMessages:
    def setup_method(self):
        from litellm.proxy.auth.user_api_key_auth import user_api_key_auth

        self._prisma = _make_prisma_client()
        app = _make_app()
        app.dependency_overrides[user_api_key_auth] = _override_auth
        self.client = TestClient(app, raise_server_exceptions=True)

    @patch("litellm.proxy.proxy_server.prisma_client")
    def test_append_message(self, mock_pc):
        mock_pc.db = self._prisma.db
        self._prisma.db.litellm_workflowrun.find_unique = AsyncMock(
            return_value=_make_run()
        )
        self._prisma.db.litellm_workflowmessage.find_many = AsyncMock(return_value=[])
        self._prisma.db.litellm_workflowmessage.create = AsyncMock(
            return_value=_make_message()
        )

        resp = self.client.post(
            "/v1/workflows/runs/run-1/messages",
            json={"role": "user", "content": "fix the bug"},
        )
        assert resp.status_code == 200

    @patch("litellm.proxy.proxy_server.prisma_client")
    def test_append_message_unknown_run_returns_404(self, mock_pc):
        mock_pc.db = self._prisma.db
        self._prisma.db.litellm_workflowrun.find_unique = AsyncMock(return_value=None)

        resp = self.client.post(
            "/v1/workflows/runs/nonexistent/messages",
            json={"role": "user", "content": "hello"},
        )
        assert resp.status_code == 404

    @patch("litellm.proxy.proxy_server.prisma_client")
    def test_list_messages_ordered(self, mock_pc):
        mock_pc.db = self._prisma.db
        self._prisma.db.litellm_workflowrun.find_unique = AsyncMock(
            return_value=_make_run()
        )
        self._prisma.db.litellm_workflowmessage.find_many = AsyncMock(
            return_value=[
                _make_message(sequence_number=0),
                _make_message(sequence_number=1, role="assistant"),
            ]
        )

        resp = self.client.get("/v1/workflows/runs/run-1/messages")
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 2
        call_kwargs = self._prisma.db.litellm_workflowmessage.find_many.call_args[1]
        assert call_kwargs["order"] == {"sequence_number": "asc"}

    @patch("litellm.proxy.proxy_server.prisma_client")
    def test_list_messages_respects_limit(self, mock_pc):
        mock_pc.db = self._prisma.db
        self._prisma.db.litellm_workflowrun.find_unique = AsyncMock(
            return_value=_make_run()
        )
        self._prisma.db.litellm_workflowmessage.find_many = AsyncMock(return_value=[])

        resp = self.client.get("/v1/workflows/runs/run-1/messages?limit=25")
        assert resp.status_code == 200
        call_kwargs = self._prisma.db.litellm_workflowmessage.find_many.call_args[1]
        assert call_kwargs["take"] == 25


class TestListWorkflowEvents:
    def setup_method(self):
        from litellm.proxy.auth.user_api_key_auth import user_api_key_auth

        self._prisma = _make_prisma_client()
        app = _make_app()
        app.dependency_overrides[user_api_key_auth] = _override_auth
        self.client = TestClient(app, raise_server_exceptions=True)

    @patch("litellm.proxy.proxy_server.prisma_client")
    def test_list_events_ordered(self, mock_pc):
        mock_pc.db = self._prisma.db
        self._prisma.db.litellm_workflowrun.find_unique = AsyncMock(
            return_value=_make_run()
        )
        self._prisma.db.litellm_workflowevent.find_many = AsyncMock(
            return_value=[
                _make_event(sequence_number=0),
                _make_event(sequence_number=1),
            ]
        )

        resp = self.client.get("/v1/workflows/runs/run-1/events")
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 2
        call_kwargs = self._prisma.db.litellm_workflowevent.find_many.call_args[1]
        assert call_kwargs["order"] == {"sequence_number": "asc"}

    @patch("litellm.proxy.proxy_server.prisma_client")
    def test_list_events_respects_limit(self, mock_pc):
        mock_pc.db = self._prisma.db
        self._prisma.db.litellm_workflowrun.find_unique = AsyncMock(
            return_value=_make_run()
        )
        self._prisma.db.litellm_workflowevent.find_many = AsyncMock(return_value=[])

        resp = self.client.get("/v1/workflows/runs/run-1/events?limit=10")
        assert resp.status_code == 200
        call_kwargs = self._prisma.db.litellm_workflowevent.find_many.call_args[1]
        assert call_kwargs["take"] == 10

    @patch("litellm.proxy.proxy_server.prisma_client")
    def test_list_events_unknown_run_returns_404(self, mock_pc):
        mock_pc.db = self._prisma.db
        self._prisma.db.litellm_workflowrun.find_unique = AsyncMock(return_value=None)

        resp = self.client.get("/v1/workflows/runs/nonexistent/events")
        assert resp.status_code == 404


class TestTenantIsolation:
    """Ownership enforcement: non-admin callers only see their own runs."""

    def _make_app_with_auth(self, auth_fn):
        from litellm.proxy.auth.user_api_key_auth import user_api_key_auth

        self._prisma = _make_prisma_client()
        app = _make_app()
        app.dependency_overrides[user_api_key_auth] = auth_fn
        return TestClient(app, raise_server_exceptions=True)

    @patch("litellm.proxy.proxy_server.prisma_client")
    def test_create_stores_caller_token(self, mock_pc):
        token = "tok-owner"
        client = self._make_app_with_auth(lambda: _override_auth_user_with_token(token))
        mock_pc.db = self._prisma.db
        self._prisma.db.litellm_workflowrun.create = AsyncMock(
            return_value=_make_run(created_by=token)
        )

        resp = client.post("/v1/workflows/runs", json={"workflow_type": "test"})
        assert resp.status_code == 200
        create_call = self._prisma.db.litellm_workflowrun.create.call_args[1]
        assert create_call["data"]["created_by"] == token

    @patch("litellm.proxy.proxy_server.prisma_client")
    def test_non_admin_list_scoped_to_caller_token(self, mock_pc):
        token = "tok-owner"
        client = self._make_app_with_auth(lambda: _override_auth_user_with_token(token))
        mock_pc.db = self._prisma.db
        self._prisma.db.litellm_workflowrun.find_many = AsyncMock(return_value=[])

        resp = client.get("/v1/workflows/runs")
        assert resp.status_code == 200
        call_kwargs = self._prisma.db.litellm_workflowrun.find_many.call_args[1]
        assert call_kwargs["where"].get("created_by") == token

    @patch("litellm.proxy.proxy_server.prisma_client")
    def test_admin_list_not_scoped(self, mock_pc):
        client = self._make_app_with_auth(_override_auth_admin)
        mock_pc.db = self._prisma.db
        self._prisma.db.litellm_workflowrun.find_many = AsyncMock(return_value=[])

        resp = client.get("/v1/workflows/runs")
        assert resp.status_code == 200
        call_kwargs = self._prisma.db.litellm_workflowrun.find_many.call_args[1]
        assert "created_by" not in call_kwargs["where"]

    @patch("litellm.proxy.proxy_server.prisma_client")
    def test_non_admin_get_other_users_run_returns_404(self, mock_pc):
        token = "tok-caller"
        client = self._make_app_with_auth(lambda: _override_auth_user_with_token(token))
        mock_pc.db = self._prisma.db
        # Run owned by a different key
        self._prisma.db.litellm_workflowrun.find_unique = AsyncMock(
            return_value=_make_run(created_by="tok-other-owner")
        )

        resp = client.get("/v1/workflows/runs/run-1")
        assert resp.status_code == 404

    @patch("litellm.proxy.proxy_server.prisma_client")
    def test_non_admin_get_null_owner_run_returns_404(self, mock_pc):
        token = "tok-caller"
        client = self._make_app_with_auth(lambda: _override_auth_user_with_token(token))
        mock_pc.db = self._prisma.db
        self._prisma.db.litellm_workflowrun.find_unique = AsyncMock(
            return_value=_make_run(created_by=None)
        )

        resp = client.get("/v1/workflows/runs/run-1")
        assert resp.status_code == 404

    @patch("litellm.proxy.proxy_server.prisma_client")
    def test_non_admin_update_null_owner_run_returns_404(self, mock_pc):
        token = "tok-caller"
        client = self._make_app_with_auth(lambda: _override_auth_user_with_token(token))
        mock_pc.db = self._prisma.db
        self._prisma.db.litellm_workflowrun.find_unique = AsyncMock(
            return_value=_make_run(created_by=None)
        )
        self._prisma.db.litellm_workflowrun.update = AsyncMock(
            return_value=_make_run(status="completed")
        )

        resp = client.patch("/v1/workflows/runs/run-1", json={"status": "completed"})
        assert resp.status_code == 404
        self._prisma.db.litellm_workflowrun.update.assert_not_awaited()

    @patch("litellm.proxy.proxy_server.prisma_client")
    def test_non_admin_get_own_run_succeeds(self, mock_pc):
        token = "tok-caller"
        client = self._make_app_with_auth(lambda: _override_auth_user_with_token(token))
        mock_pc.db = self._prisma.db
        self._prisma.db.litellm_workflowrun.find_unique = AsyncMock(
            return_value=_make_run(created_by=token)
        )

        resp = client.get("/v1/workflows/runs/run-1")
        assert resp.status_code == 200
