"""
Tests for Google Cloud authentication in MCP client.

Covers the MCPGoogleAuth httpx.Auth subclass used for GCP-managed MCP servers
(e.g. https://bigquery.googleapis.com/mcp), plus config loading, client wiring,
and credential encryption for the gcp_service_account auth type.
"""

import json
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from litellm.experimental_mcp_client.client import MCPClient, MCPGoogleAuth
from litellm.types.mcp import MCPAuth, MCPTransport
from litellm.types.mcp_server.mcp_server_manager import MCPServer

SERVICE_ACCOUNT_JSON = json.dumps({"type": "service_account", "project_id": "my-project"})


def _vertex_base_stub(token: str = "ya29.test-token", project_id: str = "my-project") -> MagicMock:
    stub = MagicMock()
    stub.get_access_token.return_value = (token, project_id)
    stub.get_access_token_async = AsyncMock(return_value=(token, project_id))
    return stub


def _request() -> httpx.Request:
    return httpx.Request("POST", "https://bigquery.googleapis.com/mcp", json={"jsonrpc": "2.0"})


class TestMCPGoogleAuth:
    def test_auth_flow_sets_bearer_token(self):
        vertex_base = _vertex_base_stub()
        auth = MCPGoogleAuth(gcp_credentials=SERVICE_ACCOUNT_JSON, vertex_base=vertex_base)

        request = next(auth.auth_flow(_request()))

        assert request.headers["Authorization"] == "Bearer ya29.test-token"
        assert "x-goog-user-project" not in request.headers
        vertex_base.get_access_token.assert_called_once_with(
            credentials=SERVICE_ACCOUNT_JSON,
            project_id=None,
        )

    def test_auth_flow_sets_quota_project_header_when_configured(self):
        auth = MCPGoogleAuth(gcp_project_id="billing-project", vertex_base=_vertex_base_stub())

        request = next(auth.auth_flow(_request()))

        assert request.headers["x-goog-user-project"] == "billing-project"

    def test_auth_flow_uses_application_default_credentials_when_unset(self):
        vertex_base = _vertex_base_stub(token="ya29.workload-identity")
        auth = MCPGoogleAuth(vertex_base=vertex_base)

        request = next(auth.auth_flow(_request()))

        assert request.headers["Authorization"] == "Bearer ya29.workload-identity"
        vertex_base.get_access_token.assert_called_once_with(credentials=None, project_id=None)

    def test_auth_flow_refreshes_token_per_request(self):
        vertex_base = _vertex_base_stub()
        vertex_base.get_access_token.side_effect = [("first", "p"), ("second", "p")]
        auth = MCPGoogleAuth(vertex_base=vertex_base)

        first = next(auth.auth_flow(_request()))
        second = next(auth.auth_flow(_request()))

        assert first.headers["Authorization"] == "Bearer first"
        assert second.headers["Authorization"] == "Bearer second"

    @pytest.mark.asyncio
    async def test_async_auth_flow_uses_async_token_path(self):
        vertex_base = _vertex_base_stub(token="ya29.async-token")
        auth = MCPGoogleAuth(gcp_credentials=SERVICE_ACCOUNT_JSON, vertex_base=vertex_base)

        request = await auth.async_auth_flow(_request()).__anext__()

        assert request.headers["Authorization"] == "Bearer ya29.async-token"
        vertex_base.get_access_token.assert_not_called()
        vertex_base.get_access_token_async.assert_awaited_once_with(
            credentials=SERVICE_ACCOUNT_JSON,
            project_id=None,
        )


class TestMCPClientGoogleAuth:
    def test_factory_wires_google_auth_into_httpx_client(self):
        google_auth = MCPGoogleAuth(vertex_base=_vertex_base_stub())
        client = MCPClient(
            server_url="https://bigquery.googleapis.com/mcp",
            transport_type=MCPTransport.http,
            auth_type=MCPAuth.gcp_service_account,
            google_auth=google_auth,
        )

        httpx_client = client._create_httpx_client_factory()(
            headers={"Content-Type": "application/json"},
            timeout=httpx.Timeout(30.0),
        )

        assert httpx_client._auth is google_auth

    def test_google_auth_does_not_add_static_auth_headers(self):
        client = MCPClient(
            server_url="https://bigquery.googleapis.com/mcp",
            transport_type=MCPTransport.http,
            auth_type=MCPAuth.gcp_service_account,
            google_auth=MCPGoogleAuth(vertex_base=_vertex_base_stub()),
        )

        assert "Authorization" not in client._get_auth_headers()


class TestMCPServerManagerGoogleAuth:
    @pytest.mark.asyncio
    async def test_load_config_parses_gcp_fields(self):
        from litellm.proxy._experimental.mcp_server.mcp_server_manager import MCPServerManager

        manager = MCPServerManager()
        await manager.load_servers_from_config(
            {
                "bigquery": {
                    "url": "https://bigquery.googleapis.com/mcp",
                    "transport": "http",
                    "auth_type": "gcp_service_account",
                    "gcp_credentials": SERVICE_ACCOUNT_JSON,
                    "gcp_project_id": "my-project",
                }
            }
        )

        server = next(iter(manager.config_mcp_servers.values()))
        assert server.auth_type == MCPAuth.gcp_service_account
        assert server.gcp_credentials == SERVICE_ACCOUNT_JSON
        assert server.gcp_project_id == "my-project"

    @pytest.mark.asyncio
    async def test_create_mcp_client_builds_google_auth(self):
        from litellm.proxy._experimental.mcp_server.mcp_server_manager import MCPServerManager

        server = MCPServer(
            server_id="test-gcp",
            name="bigquery",
            server_name="bigquery",
            url="https://bigquery.googleapis.com/mcp",
            transport=MCPTransport.http,
            auth_type=MCPAuth.gcp_service_account,
            gcp_credentials=SERVICE_ACCOUNT_JSON,
            gcp_project_id="my-project",
        )

        client = await MCPServerManager()._create_mcp_client(server=server)

        assert isinstance(client._google_auth, MCPGoogleAuth)
        assert client._google_auth.gcp_credentials == SERVICE_ACCOUNT_JSON
        assert client._google_auth.gcp_project_id == "my-project"

    @pytest.mark.asyncio
    async def test_create_mcp_client_without_gcp_auth(self):
        from litellm.proxy._experimental.mcp_server.mcp_server_manager import MCPServerManager

        server = MCPServer(
            server_id="test-bearer",
            name="bearer",
            server_name="bearer",
            url="https://example.com/mcp",
            transport=MCPTransport.http,
            auth_type=MCPAuth.bearer_token,
            authentication_token="token",
        )

        client = await MCPServerManager()._create_mcp_client(server=server)

        assert client._google_auth is None

    def test_registry_dump_redacts_service_account_json(self):
        from litellm.proxy._experimental.mcp_server.mcp_server_manager import _redacted_registry_dump

        server = MCPServer(
            server_id="test-gcp",
            name="bigquery",
            url="https://bigquery.googleapis.com/mcp",
            transport=MCPTransport.http,
            auth_type=MCPAuth.gcp_service_account,
            gcp_credentials=SERVICE_ACCOUNT_JSON,
        )

        dump = _redacted_registry_dump({"test-gcp": server})["test-gcp"]

        assert dump["gcp_credentials"] == "**REDACTED**"

    def test_build_from_table_decrypts_gcp_credentials(self, monkeypatch):
        from litellm.proxy._experimental.mcp_server.db import decrypt_credentials, encrypt_credentials

        monkeypatch.setenv("LITELLM_SALT_KEY", "sk-test-salt-key")

        credentials = encrypt_credentials(
            {"gcp_credentials": SERVICE_ACCOUNT_JSON, "gcp_project_id": "my-project"},
            encryption_key=None,
        )
        assert credentials["gcp_credentials"] != SERVICE_ACCOUNT_JSON
        assert credentials["gcp_project_id"] == "my-project"

        assert decrypt_credentials(credentials)["gcp_credentials"] == SERVICE_ACCOUNT_JSON

    def test_extract_gcp_credentials_from_unencrypted_blob(self):
        from litellm.proxy._experimental.mcp_server.mcp_server_manager import MCPServerManager

        extracted = MCPServerManager()._extract_gcp_credentials(
            {"gcp_credentials": SERVICE_ACCOUNT_JSON, "gcp_project_id": "my-project"},
            credentials_are_encrypted=False,
        )

        assert extracted == {
            "gcp_credentials": SERVICE_ACCOUNT_JSON,
            "gcp_project_id": "my-project",
        }
