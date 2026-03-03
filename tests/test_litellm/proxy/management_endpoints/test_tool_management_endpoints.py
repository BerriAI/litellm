"""
Unit tests for tool management endpoints (/v1/tool/*).
Uses FastAPI TestClient with mocked DB functions.

Patches target the source modules (litellm.proxy.db.tool_registry_writer.*
and litellm.proxy.proxy_server.prisma_client) because the endpoint code
imports these inside function bodies to avoid circular imports.
"""

import os
import sys
from datetime import datetime, timezone
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath("../../.."))

from litellm.proxy.management_endpoints.tool_management_endpoints import router
from litellm.types.tool_management import LiteLLM_ToolTableRow

# --- helpers ---


def _make_tool_row(
    tool_name: str = "my_tool",
    call_policy: str = "untrusted",
    origin: Optional[str] = None,
) -> LiteLLM_ToolTableRow:
    now = datetime.now(timezone.utc)
    return LiteLLM_ToolTableRow(
        tool_id="uuid-1",
        tool_name=tool_name,
        origin=origin,
        call_policy=call_policy,  # type: ignore[arg-type]
        assignments={},
        created_at=now,
        updated_at=now,
    )


def _make_app() -> FastAPI:
    """Build a minimal FastAPI app with the tool management router."""
    app = FastAPI()
    app.include_router(router)
    return app


# Stub the auth dependency so we don't need a real proxy running.
def _override_auth():
    from litellm.proxy._types import UserAPIKeyAuth

    return UserAPIKeyAuth(api_key="sk-test", user_id="admin")


# A real (non-None) prisma stub for truthiness checks.
_MOCK_PRISMA = MagicMock()


# --- test class ---


class TestToolManagementEndpoints:
    def setup_method(self):
        from litellm.proxy.auth.user_api_key_auth import user_api_key_auth

        app = _make_app()
        app.dependency_overrides[user_api_key_auth] = _override_auth
        self.client = TestClient(app, raise_server_exceptions=True)

    @patch(
        "litellm.proxy.db.tool_registry_writer.list_tools",
        new_callable=AsyncMock,
    )
    @patch("litellm.proxy.proxy_server.prisma_client", _MOCK_PRISMA)
    def test_list_tools_returns_200(self, mock_db_list):
        mock_db_list.return_value = [_make_tool_row()]

        resp = self.client.get("/v1/tool/list")
        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] == 1
        assert body["tools"][0]["tool_name"] == "my_tool"

    @patch(
        "litellm.proxy.db.tool_registry_writer.list_tools",
        new_callable=AsyncMock,
    )
    @patch("litellm.proxy.proxy_server.prisma_client", _MOCK_PRISMA)
    def test_list_tools_with_policy_filter(self, mock_db_list):
        mock_db_list.return_value = [_make_tool_row(call_policy="blocked")]

        resp = self.client.get("/v1/tool/list?call_policy=blocked")
        assert resp.status_code == 200
        assert resp.json()["tools"][0]["call_policy"] == "blocked"

    @patch(
        "litellm.proxy.db.tool_registry_writer.get_tool",
        new_callable=AsyncMock,
    )
    @patch("litellm.proxy.proxy_server.prisma_client", _MOCK_PRISMA)
    def test_get_tool_found(self, mock_db_get):
        mock_db_get.return_value = _make_tool_row(tool_name="tool_a")

        resp = self.client.get("/v1/tool/tool_a")
        assert resp.status_code == 200
        assert resp.json()["tool_name"] == "tool_a"

    @patch(
        "litellm.proxy.db.tool_registry_writer.get_tool",
        new_callable=AsyncMock,
    )
    @patch("litellm.proxy.proxy_server.prisma_client", _MOCK_PRISMA)
    def test_get_tool_not_found_returns_404(self, mock_db_get):
        mock_db_get.return_value = None

        resp = self.client.get("/v1/tool/nonexistent", follow_redirects=True)
        assert resp.status_code == 404

    @patch(
        "litellm.proxy.db.tool_registry_writer.update_tool_policy",
        new_callable=AsyncMock,
    )
    @patch("litellm.proxy.proxy_server.prisma_client", _MOCK_PRISMA)
    def test_update_tool_policy_blocked(self, mock_db_update):
        mock_db_update.return_value = _make_tool_row(call_policy="blocked")

        resp = self.client.post(
            "/v1/tool/policy",
            json={"tool_name": "my_tool", "call_policy": "blocked"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["call_policy"] == "blocked"
        assert body["updated"] is True

    @patch("litellm.proxy.proxy_server.prisma_client", None)
    def test_list_tools_no_db_returns_500(self):
        resp = self.client.get("/v1/tool/list")
        assert resp.status_code == 500

    def test_update_tool_policy_invalid_policy_returns_422(self):
        resp = self.client.post(
            "/v1/tool/policy",
            json={"tool_name": "my_tool", "call_policy": "invalid_value"},
        )
        assert resp.status_code == 422
