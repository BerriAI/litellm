"""
Tests for AWS SigV4 authentication in MCP client.

Tests the MCPSigV4Auth httpx.Auth subclass that enables per-request
SigV4 signing for Bedrock AgentCore MCP servers, plus DB/UI path
tests for credential encryption, merge-on-update, and build_from_table.
"""

import json

import pytest
from unittest.mock import patch, MagicMock, AsyncMock

import httpx

from litellm.experimental_mcp_client.client import MCPSigV4Auth, MCPClient
from litellm.types.mcp import MCPAuth, MCPTransport


class TestMCPSigV4Auth:
    """Unit tests for the MCPSigV4Auth class."""

    def test_init_with_explicit_credentials(self):
        """MCPSigV4Auth initializes with explicit AWS credentials."""
        auth = MCPSigV4Auth(
            aws_access_key_id="AKIAIOSFODNN7EXAMPLE",
            aws_secret_access_key="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
            aws_session_token="FwoGZXIvYXdzEBYaDH...",
            aws_region_name="us-east-1",
            aws_service_name="bedrock-agentcore",
        )
        assert auth.credentials is not None
        assert auth.credentials.access_key == "AKIAIOSFODNN7EXAMPLE"
        assert auth.credentials.secret_key == "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
        assert auth.credentials.token == "FwoGZXIvYXdzEBYaDH..."
        assert auth.region_name == "us-east-1"
        assert auth.service_name == "bedrock-agentcore"

    def test_requires_request_body_flag(self):
        """MCPSigV4Auth sets requires_request_body so httpx buffers the body before signing."""
        assert MCPSigV4Auth.requires_request_body is True

    def test_init_defaults(self):
        """MCPSigV4Auth uses correct defaults for region and service."""
        auth = MCPSigV4Auth(
            aws_access_key_id="AKIAIOSFODNN7EXAMPLE",
            aws_secret_access_key="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
        )
        assert auth.region_name == "us-east-1"
        assert auth.service_name == "bedrock-agentcore"

    def test_init_with_resolved_env_values(self):
        """MCPSigV4Auth works with pre-resolved values (os.environ/ is resolved at config load time)."""
        # Values arrive already resolved by ProxyConfig._check_for_os_environ_vars(),
        # so MCPSigV4Auth receives plain strings, not os.environ/ prefixed values.
        auth = MCPSigV4Auth(
            aws_access_key_id="RESOLVED_KEY_FROM_ENV",
            aws_secret_access_key="RESOLVED_SECRET_FROM_ENV",
            aws_region_name="us-west-2",
        )
        assert auth.credentials.access_key == "RESOLVED_KEY_FROM_ENV"
        assert auth.credentials.secret_key == "RESOLVED_SECRET_FROM_ENV"
        assert auth.region_name == "us-west-2"

    def test_init_falls_back_to_boto_session(self):
        """MCPSigV4Auth falls back to boto3 credential chain when no explicit creds."""
        mock_creds = MagicMock()
        mock_creds.access_key = "SESSION_KEY"
        mock_creds.secret_key = "SESSION_SECRET"

        mock_session = MagicMock()
        mock_session.get_credentials.return_value = mock_creds

        with patch("botocore.session.get_session", return_value=mock_session):
            auth = MCPSigV4Auth(
                aws_region_name="eu-west-1",
                aws_service_name="custom-service",
            )
            assert auth.credentials == mock_creds
            assert auth.region_name == "eu-west-1"
            assert auth.service_name == "custom-service"

    def test_init_raises_when_no_credentials(self):
        """MCPSigV4Auth raises ValueError when no credentials are available."""
        mock_session = MagicMock()
        mock_session.get_credentials.return_value = None

        with patch("botocore.session.get_session", return_value=mock_session):
            with pytest.raises(ValueError, match="No AWS credentials found"):
                MCPSigV4Auth()

    def test_auth_flow_signs_request(self):
        """MCPSigV4Auth.auth_flow adds SigV4 headers to the request."""
        auth = MCPSigV4Auth(
            aws_access_key_id="AKIAIOSFODNN7EXAMPLE",
            aws_secret_access_key="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
            aws_region_name="us-east-1",
            aws_service_name="bedrock-agentcore",
        )

        request = httpx.Request(
            method="POST",
            url="https://bedrock-agentcore.us-east-1.amazonaws.com/runtimes/test/invocations",
            headers={"Content-Type": "application/json"},
            content=b'{"jsonrpc":"2.0","method":"tools/list","id":1}',
        )

        # Execute auth_flow generator
        flow = auth.auth_flow(request)
        signed_request = next(flow)

        # Verify SigV4 headers were added
        assert "Authorization" in signed_request.headers
        assert "AWS4-HMAC-SHA256" in signed_request.headers["Authorization"]
        assert "x-amz-date" in signed_request.headers
        assert "bedrock-agentcore" in signed_request.headers["Authorization"]

    def test_auth_flow_different_bodies_produce_different_signatures(self):
        """Each request gets a unique signature based on its body."""
        auth = MCPSigV4Auth(
            aws_access_key_id="AKIAIOSFODNN7EXAMPLE",
            aws_secret_access_key="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
            aws_region_name="us-east-1",
        )

        request1 = httpx.Request(
            method="POST",
            url="https://example.com/mcp",
            headers={"Content-Type": "application/json"},
            content=b'{"jsonrpc":"2.0","method":"tools/list","id":1}',
        )
        request2 = httpx.Request(
            method="POST",
            url="https://example.com/mcp",
            headers={"Content-Type": "application/json"},
            content=b'{"jsonrpc":"2.0","method":"tools/call","id":2,"params":{"name":"search"}}',
        )

        signed1 = next(auth.auth_flow(request1))
        signed2 = next(auth.auth_flow(request2))

        # Signatures must differ because body content differs
        assert signed1.headers["Authorization"] != signed2.headers["Authorization"]

    def test_auth_flow_includes_security_token(self):
        """SigV4 signing includes X-Amz-Security-Token when session token is present."""
        auth = MCPSigV4Auth(
            aws_access_key_id="AKIAIOSFODNN7EXAMPLE",
            aws_secret_access_key="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
            aws_session_token="SESSION_TOKEN_EXAMPLE",
            aws_region_name="us-east-1",
        )

        request = httpx.Request(
            method="POST",
            url="https://example.com/mcp",
            headers={"Content-Type": "application/json"},
            content=b'{"jsonrpc":"2.0","method":"initialize","id":0}',
        )

        signed_request = next(auth.auth_flow(request))
        assert "x-amz-security-token" in signed_request.headers


class TestMCPClientSigV4Integration:
    """Tests for MCPClient with SigV4 auth wired through."""

    def test_mcp_client_stores_aws_auth(self):
        """MCPClient stores the aws_auth parameter."""
        mock_auth = MagicMock(spec=httpx.Auth)
        client = MCPClient(
            server_url="https://example.com/mcp",
            transport_type=MCPTransport.http,
            auth_type=MCPAuth.aws_sigv4,
            aws_auth=mock_auth,
        )
        assert client._aws_auth is mock_auth

    def test_mcp_client_factory_uses_aws_auth(self):
        """The httpx client factory uses aws_auth when no explicit auth is passed."""
        mock_auth = MCPSigV4Auth(
            aws_access_key_id="AKIAIOSFODNN7EXAMPLE",
            aws_secret_access_key="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
        )
        client = MCPClient(
            server_url="https://example.com/mcp",
            transport_type=MCPTransport.http,
            aws_auth=mock_auth,
        )

        factory = client._create_httpx_client_factory()
        httpx_client = factory(
            headers={"Content-Type": "application/json"},
            timeout=httpx.Timeout(30.0),
        )

        # Verify the auth object was actually wired into the httpx client
        assert httpx_client._auth is mock_auth

    def test_mcp_client_factory_explicit_auth_takes_precedence(self):
        """When explicit auth= is passed to the factory, it takes precedence over aws_auth."""
        aws_auth = MCPSigV4Auth(
            aws_access_key_id="AKIAIOSFODNN7EXAMPLE",
            aws_secret_access_key="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
        )
        explicit_auth = MagicMock(spec=httpx.Auth)

        client = MCPClient(
            server_url="https://example.com/mcp",
            transport_type=MCPTransport.http,
            aws_auth=aws_auth,
        )

        factory = client._create_httpx_client_factory()
        httpx_client = factory(
            headers={"Content-Type": "application/json"},
            timeout=httpx.Timeout(30.0),
            auth=explicit_auth,
        )

        # Explicit auth should win over aws_auth
        assert httpx_client._auth is explicit_auth

    def test_mcp_client_factory_no_aws_auth(self):
        """The httpx client factory works normally when no aws_auth is set."""
        client = MCPClient(
            server_url="https://example.com/mcp",
            transport_type=MCPTransport.http,
        )

        factory = client._create_httpx_client_factory()
        httpx_client = factory(
            headers={"Content-Type": "application/json"},
            timeout=httpx.Timeout(30.0),
        )
        # No auth should be set when aws_auth is not configured
        assert httpx_client._auth is None


class TestMCPServerManagerSigV4:
    """Tests for MCPServerManager config loading with SigV4."""

    @pytest.mark.asyncio
    async def test_load_config_with_aws_sigv4(self):
        """Config loading correctly parses aws_sigv4 auth type and AWS fields."""
        from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
            MCPServerManager,
        )

        config = {
            "agentcore_tools": {
                "url": "https://bedrock-agentcore.us-east-1.amazonaws.com/runtimes/test/invocations",
                "transport": "http",
                "auth_type": "aws_sigv4",
                "aws_access_key_id": "AKIAIOSFODNN7EXAMPLE",
                "aws_secret_access_key": "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
                "aws_region_name": "us-east-1",
                "aws_service_name": "bedrock-agentcore",
            }
        }

        manager = MCPServerManager()
        await manager.load_servers_from_config(config)

        server = next(iter(manager.config_mcp_servers.values()))
        assert server.auth_type == MCPAuth.aws_sigv4
        assert server.aws_access_key_id == "AKIAIOSFODNN7EXAMPLE"
        assert server.aws_secret_access_key == "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
        assert server.aws_region_name == "us-east-1"
        assert server.aws_service_name == "bedrock-agentcore"

    @pytest.mark.asyncio
    async def test_create_mcp_client_with_sigv4(self):
        """_create_mcp_client creates client with SigV4 auth when auth_type is aws_sigv4."""
        from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
            MCPServerManager,
        )
        from litellm.types.mcp_server.mcp_server_manager import MCPServer

        server = MCPServer(
            server_id="test-sigv4",
            name="test_sigv4_server",
            server_name="test_sigv4",
            url="https://bedrock-agentcore.us-east-1.amazonaws.com/runtimes/test/invocations",
            transport=MCPTransport.http,
            auth_type=MCPAuth.aws_sigv4,
            aws_access_key_id="AKIAIOSFODNN7EXAMPLE",
            aws_secret_access_key="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
            aws_region_name="us-east-1",
        )

        manager = MCPServerManager()
        client = await manager._create_mcp_client(server=server)

        assert client.auth_type == MCPAuth.aws_sigv4
        assert client._aws_auth is not None
        assert isinstance(client._aws_auth, MCPSigV4Auth)

    @pytest.mark.asyncio
    async def test_create_mcp_client_without_sigv4(self):
        """_create_mcp_client does not create SigV4 auth for non-SigV4 servers."""
        from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
            MCPServerManager,
        )
        from litellm.types.mcp_server.mcp_server_manager import MCPServer

        server = MCPServer(
            server_id="test-bearer",
            name="test_bearer_server",
            server_name="test_bearer",
            url="https://example.com/mcp",
            transport=MCPTransport.http,
            auth_type=MCPAuth.bearer_token,
            authentication_token="test-token",
        )

        manager = MCPServerManager()
        client = await manager._create_mcp_client(server=server)

        assert client._aws_auth is None


class TestSigV4CredentialEncryption:
    """Test encrypt/decrypt round-trip for AWS SigV4 credentials."""

    def test_encrypt_credentials_handles_aws_fields(self):
        """AWS credential fields are encrypted in the credentials dict."""
        from litellm.proxy._experimental.mcp_server.db import encrypt_credentials

        creds = {
            "aws_access_key_id": "AKIAIOSFODNN7EXAMPLE",
            "aws_secret_access_key": "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
            "aws_session_token": "FwoGZX...",
            "aws_region_name": "us-east-1",
            "aws_service_name": "bedrock-agentcore",
        }

        with patch(
            "litellm.proxy._experimental.mcp_server.db.encrypt_value_helper",
            side_effect=lambda value, new_encryption_key: f"enc:{value}",
        ):
            result = encrypt_credentials(credentials=creds, encryption_key="test-key")

        # Secrets should be encrypted
        assert result["aws_access_key_id"] == "enc:AKIAIOSFODNN7EXAMPLE"
        assert (
            result["aws_secret_access_key"]
            == "enc:wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
        )
        assert result["aws_session_token"] == "enc:FwoGZX..."
        # Non-secrets should be unchanged
        assert result["aws_region_name"] == "us-east-1"
        assert result["aws_service_name"] == "bedrock-agentcore"

    def test_encrypt_credentials_skips_absent_aws_fields(self):
        """encrypt_credentials does not fail when AWS fields are absent."""
        from litellm.proxy._experimental.mcp_server.db import encrypt_credentials

        creds = {"auth_value": "some-token"}

        with patch(
            "litellm.proxy._experimental.mcp_server.db.encrypt_value_helper",
            side_effect=lambda value, new_encryption_key: f"enc:{value}",
        ):
            result = encrypt_credentials(credentials=creds, encryption_key="test-key")

        assert result["auth_value"] == "enc:some-token"
        assert "aws_access_key_id" not in result


class TestCredentialMergeOnUpdate:
    """Test that partial credential updates preserve existing fields."""

    @pytest.mark.asyncio
    async def test_partial_update_preserves_existing_credentials(self):
        """Updating only aws_region_name should not wipe aws_secret_access_key."""
        from litellm.proxy._experimental.mcp_server.db import update_mcp_server
        from litellm.proxy._types import UpdateMCPServerRequest

        existing_record = MagicMock()
        existing_record.auth_type = "aws_sigv4"
        existing_record.credentials = json.dumps(
            {
                "aws_access_key_id": "enc:AKI",
                "aws_secret_access_key": "enc:SAK",
                "aws_region_name": "us-east-1",
            }
        )

        mock_prisma = MagicMock()
        mock_prisma.db.litellm_mcpservertable.find_unique = AsyncMock(
            return_value=existing_record
        )
        mock_prisma.db.litellm_mcpservertable.update = AsyncMock(
            return_value=MagicMock()
        )

        data = UpdateMCPServerRequest(
            server_id="test-server",
            auth_type="aws_sigv4",
            credentials={"aws_region_name": "eu-west-1"},
        )

        with patch(
            "litellm.proxy._experimental.mcp_server.db._get_salt_key",
            return_value=None,
        ), patch(
            "litellm.proxy._experimental.mcp_server.db.encrypt_value_helper",
            side_effect=lambda value, new_encryption_key: value,
        ):
            await update_mcp_server(mock_prisma, data, "test-user")

        # Grab the data dict passed to prisma update
        update_call = mock_prisma.db.litellm_mcpservertable.update
        assert update_call.called
        data_dict = update_call.call_args[1]["data"]
        merged_creds = json.loads(data_dict["credentials"])

        # Existing encrypted secrets should be preserved
        assert merged_creds["aws_access_key_id"] == "enc:AKI"
        assert merged_creds["aws_secret_access_key"] == "enc:SAK"
        # New region value should be updated
        assert merged_creds["aws_region_name"] == "eu-west-1"

    @pytest.mark.asyncio
    async def test_update_without_credentials_preserves_all(self):
        """Update with no credentials field should not touch existing credentials."""
        from litellm.proxy._experimental.mcp_server.db import update_mcp_server
        from litellm.proxy._types import UpdateMCPServerRequest

        mock_prisma = MagicMock()
        mock_prisma.db.litellm_mcpservertable.update = AsyncMock(
            return_value=MagicMock()
        )

        data = UpdateMCPServerRequest(
            server_id="test-server",
            description="Updated description",
        )

        with patch(
            "litellm.proxy._experimental.mcp_server.db._get_salt_key",
            return_value=None,
        ):
            await update_mcp_server(mock_prisma, data, "test-user")

        data_dict = mock_prisma.db.litellm_mcpservertable.update.call_args[1]["data"]
        assert "credentials" not in data_dict

    @pytest.mark.asyncio
    async def test_update_new_server_no_merge(self):
        """Update with credentials on a server that has no existing credentials."""
        from litellm.proxy._experimental.mcp_server.db import update_mcp_server
        from litellm.proxy._types import UpdateMCPServerRequest

        existing_record = MagicMock()
        existing_record.auth_type = "aws_sigv4"
        existing_record.credentials = None

        mock_prisma = MagicMock()
        mock_prisma.db.litellm_mcpservertable.find_unique = AsyncMock(
            return_value=existing_record
        )
        mock_prisma.db.litellm_mcpservertable.update = AsyncMock(
            return_value=MagicMock()
        )

        data = UpdateMCPServerRequest(
            server_id="test-server",
            auth_type="aws_sigv4",
            credentials={"aws_region_name": "us-east-1"},
        )

        with patch(
            "litellm.proxy._experimental.mcp_server.db._get_salt_key",
            return_value=None,
        ), patch(
            "litellm.proxy._experimental.mcp_server.db.encrypt_value_helper",
            side_effect=lambda value, new_encryption_key: value,
        ):
            await update_mcp_server(mock_prisma, data, "test-user")

        data_dict = mock_prisma.db.litellm_mcpservertable.update.call_args[1]["data"]
        stored_creds = json.loads(data_dict["credentials"])
        assert stored_creds == {"aws_region_name": "us-east-1"}

    @pytest.mark.asyncio
    async def test_auth_type_change_replaces_credentials_entirely(self):
        """Switching auth_type should replace credentials, not merge."""
        from litellm.proxy._experimental.mcp_server.db import update_mcp_server
        from litellm.proxy._types import UpdateMCPServerRequest

        existing_record = MagicMock()
        existing_record.auth_type = "aws_sigv4"
        existing_record.credentials = json.dumps(
            {
                "aws_access_key_id": "enc:AKI",
                "aws_secret_access_key": "enc:SAK",
                "aws_region_name": "us-east-1",
            }
        )

        mock_prisma = MagicMock()
        mock_prisma.db.litellm_mcpservertable.find_unique = AsyncMock(
            return_value=existing_record
        )
        mock_prisma.db.litellm_mcpservertable.update = AsyncMock(
            return_value=MagicMock()
        )

        data = UpdateMCPServerRequest(
            server_id="test-server",
            auth_type="api_key",
            credentials={"auth_value": "my-key"},
        )

        with patch(
            "litellm.proxy._experimental.mcp_server.db._get_salt_key",
            return_value=None,
        ), patch(
            "litellm.proxy._experimental.mcp_server.db.encrypt_value_helper",
            side_effect=lambda value, new_encryption_key: f"enc:{value}",
        ):
            await update_mcp_server(mock_prisma, data, "test-user")

        data_dict = mock_prisma.db.litellm_mcpservertable.update.call_args[1]["data"]
        stored_creds = json.loads(data_dict["credentials"])
        # Should only have the new api_key credential, no stale aws_* fields
        assert stored_creds == {"auth_value": "enc:my-key"}

    @pytest.mark.asyncio
    async def test_same_auth_type_merges_credentials(self):
        """Same auth_type should merge credentials (preserve untouched fields)."""
        from litellm.proxy._experimental.mcp_server.db import update_mcp_server
        from litellm.proxy._types import UpdateMCPServerRequest

        existing_record = MagicMock()
        existing_record.auth_type = "oauth2"
        existing_record.credentials = json.dumps(
            {
                "client_id": "enc:id",
                "client_secret": "enc:secret",
                "scopes": ["read"],
            }
        )

        mock_prisma = MagicMock()
        mock_prisma.db.litellm_mcpservertable.find_unique = AsyncMock(
            return_value=existing_record
        )
        mock_prisma.db.litellm_mcpservertable.update = AsyncMock(
            return_value=MagicMock()
        )

        data = UpdateMCPServerRequest(
            server_id="test-server",
            auth_type="oauth2",
            credentials={"scopes": ["read", "write"]},
        )

        with patch(
            "litellm.proxy._experimental.mcp_server.db._get_salt_key",
            return_value=None,
        ), patch(
            "litellm.proxy._experimental.mcp_server.db.encrypt_value_helper",
            side_effect=lambda value, new_encryption_key: value,
        ):
            await update_mcp_server(mock_prisma, data, "test-user")

        data_dict = mock_prisma.db.litellm_mcpservertable.update.call_args[1]["data"]
        merged_creds = json.loads(data_dict["credentials"])
        assert merged_creds["client_id"] == "enc:id"
        assert merged_creds["client_secret"] == "enc:secret"
        assert merged_creds["scopes"] == ["read", "write"]


class TestSigV4BuildFromTable:
    """Test build_mcp_server_from_table correctly loads AWS SigV4 credentials."""

    @pytest.mark.asyncio
    async def test_build_mcp_server_from_table_with_sigv4_credentials(self):
        """SigV4 credentials from DB are decrypted and mapped to MCPServer fields."""
        from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
            MCPServerManager,
        )

        table_record = MagicMock()
        table_record.server_id = "test-sigv4-server"
        table_record.server_name = "sigv4_server"
        table_record.alias = None
        table_record.description = None
        table_record.url = "https://bedrock-agentcore.us-east-1.amazonaws.com/invocations"
        table_record.spec_path = None
        table_record.transport = "http"
        table_record.auth_type = "aws_sigv4"
        table_record.mcp_info = {"server_name": "sigv4_server"}
        table_record.credentials = json.dumps(
            {
                "aws_access_key_id": "enc:AKIAEXAMPLE",
                "aws_secret_access_key": "enc:SECRET",
                "aws_session_token": "enc:TOKEN",
                "aws_region_name": "us-east-1",
                "aws_service_name": "bedrock-agentcore",
            }
        )
        table_record.extra_headers = None
        table_record.static_headers = None
        table_record.command = None
        table_record.args = []
        table_record.env = None
        table_record.mcp_access_groups = []
        table_record.allowed_tools = []
        table_record.disallowed_tools = None
        table_record.allow_all_keys = False
        table_record.available_on_public_internet = True
        table_record.authorization_url = None
        table_record.token_url = None
        table_record.registration_url = None
        table_record.created_at = None
        table_record.updated_at = None
        table_record.client_id = None
        table_record.client_secret = None
        table_record.tool_name_to_display_name = None
        table_record.tool_name_to_description = None
        table_record.byok_api_key_help_url = None
        table_record.oauth2_flow = None

        manager = MCPServerManager()

        with patch(
            "litellm.proxy._experimental.mcp_server.mcp_server_manager.decrypt_value_helper",
            side_effect=lambda value, key, exception_type, return_original_value: value.replace(
                "enc:", ""
            ),
        ):
            server = await manager.build_mcp_server_from_table(table_record)

        assert server.auth_type == "aws_sigv4"
        assert server.aws_access_key_id == "AKIAEXAMPLE"
        assert server.aws_secret_access_key == "SECRET"
        assert server.aws_session_token == "TOKEN"
        assert server.aws_region_name == "us-east-1"
        assert server.aws_service_name == "bedrock-agentcore"

    @pytest.mark.asyncio
    async def test_build_mcp_server_from_table_without_sigv4_credentials(self):
        """Non-SigV4 servers still work — AWS fields default to None."""
        from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
            MCPServerManager,
        )

        table_record = MagicMock()
        table_record.server_id = "test-bearer-server"
        table_record.server_name = "bearer_server"
        table_record.alias = None
        table_record.description = None
        table_record.url = "https://example.com/mcp"
        table_record.spec_path = None
        table_record.transport = "http"
        table_record.auth_type = "bearer_token"
        table_record.mcp_info = {"server_name": "bearer_server"}
        table_record.credentials = json.dumps({"auth_value": "enc:tok"})
        table_record.extra_headers = None
        table_record.static_headers = None
        table_record.command = None
        table_record.args = []
        table_record.env = None
        table_record.mcp_access_groups = []
        table_record.allowed_tools = []
        table_record.disallowed_tools = None
        table_record.allow_all_keys = False
        table_record.available_on_public_internet = True
        table_record.authorization_url = None
        table_record.token_url = None
        table_record.registration_url = None
        table_record.created_at = None
        table_record.updated_at = None
        table_record.client_id = None
        table_record.client_secret = None
        table_record.tool_name_to_display_name = None
        table_record.tool_name_to_description = None
        table_record.byok_api_key_help_url = None
        table_record.oauth2_flow = None

        manager = MCPServerManager()

        with patch(
            "litellm.proxy._experimental.mcp_server.mcp_server_manager.decrypt_value_helper",
            side_effect=lambda value, key, exception_type, return_original_value: value.replace(
                "enc:", ""
            ),
        ):
            server = await manager.build_mcp_server_from_table(table_record)

        assert server.auth_type == "bearer_token"
        assert server.aws_access_key_id is None
        assert server.aws_secret_access_key is None
        assert server.aws_session_token is None
        assert server.aws_region_name is None
        assert server.aws_service_name is None


class TestDecryptCredentials:
    """Test decrypt_credentials helper."""

    def test_decrypt_credentials_handles_all_secret_fields(self):
        """All secret fields are decrypted; non-secret fields are left as-is."""
        from litellm.proxy._experimental.mcp_server.db import decrypt_credentials

        creds = {
            "auth_value": "enc:tok",
            "client_id": "enc:cid",
            "client_secret": "enc:csec",
            "aws_access_key_id": "enc:AKI",
            "aws_secret_access_key": "enc:SAK",
            "aws_session_token": "enc:TOK",
            "aws_region_name": "us-east-1",
            "aws_service_name": "bedrock-agentcore",
        }

        with patch(
            "litellm.proxy._experimental.mcp_server.db.decrypt_value_helper",
            side_effect=lambda value, key, exception_type="error", return_original_value=False: value.replace("enc:", ""),
        ):
            result = decrypt_credentials(credentials=creds)

        assert result["auth_value"] == "tok"
        assert result["client_id"] == "cid"
        assert result["client_secret"] == "csec"
        assert result["aws_access_key_id"] == "AKI"
        assert result["aws_secret_access_key"] == "SAK"
        assert result["aws_session_token"] == "TOK"
        # Non-secrets untouched
        assert result["aws_region_name"] == "us-east-1"
        assert result["aws_service_name"] == "bedrock-agentcore"

    def test_decrypt_credentials_skips_absent_fields(self):
        """Absent fields are not touched."""
        from litellm.proxy._experimental.mcp_server.db import decrypt_credentials

        creds = {"aws_access_key_id": "enc:AKI"}

        with patch(
            "litellm.proxy._experimental.mcp_server.db.decrypt_value_helper",
            side_effect=lambda value, key, exception_type="error", return_original_value=False: value.replace("enc:", ""),
        ):
            result = decrypt_credentials(credentials=creds)

        assert result["aws_access_key_id"] == "AKI"
        assert "aws_secret_access_key" not in result


class TestRotateCredentials:
    """Test rotate_mcp_server_credentials_master_key decrypts before re-encrypting."""

    @pytest.mark.asyncio
    async def test_rotation_decrypts_then_reencrypts(self):
        """Key rotation should decrypt with old key then encrypt with new key."""
        from litellm.proxy._experimental.mcp_server.db import (
            rotate_mcp_server_credentials_master_key,
        )

        server = MagicMock()
        server.server_id = "srv-1"
        server.credentials = {
            "aws_access_key_id": "enc_old:AKI",
            "aws_secret_access_key": "enc_old:SAK",
            "aws_region_name": "us-east-1",
        }

        mock_prisma = MagicMock()
        mock_prisma.db.litellm_mcpservertable.find_many = AsyncMock(
            return_value=[server]
        )
        mock_prisma.db.litellm_mcpservertable.update = AsyncMock()

        with patch(
            "litellm.proxy._experimental.mcp_server.db._get_salt_key",
            return_value="old-key",
        ), patch(
            "litellm.proxy._experimental.mcp_server.db.decrypt_value_helper",
            side_effect=lambda value, key, exception_type="error", return_original_value=False: value.replace("enc_old:", ""),
        ), patch(
            "litellm.proxy._experimental.mcp_server.db.encrypt_value_helper",
            side_effect=lambda value, new_encryption_key: f"enc_new:{value}",
        ):
            await rotate_mcp_server_credentials_master_key(
                mock_prisma, "admin", "new-key"
            )

        update_call = mock_prisma.db.litellm_mcpservertable.update
        assert update_call.called
        stored_creds = json.loads(update_call.call_args[1]["data"]["credentials"])
        # Should be decrypted from old, then encrypted with new
        assert stored_creds["aws_access_key_id"] == "enc_new:AKI"
        assert stored_creds["aws_secret_access_key"] == "enc_new:SAK"
        # Non-secret fields should pass through unchanged
        assert stored_creds["aws_region_name"] == "us-east-1"


class TestAuthTypeSwitchClearsCredentials:
    """Test that switching auth_type without credentials clears stale secrets."""

    @pytest.mark.asyncio
    async def test_auth_type_change_without_credentials_clears_stale(self):
        """Changing auth_type without providing credentials should clear old ones."""
        from litellm.proxy._experimental.mcp_server.db import update_mcp_server
        from litellm.proxy._types import UpdateMCPServerRequest

        existing_record = MagicMock()
        existing_record.auth_type = "oauth2"
        existing_record.credentials = json.dumps(
            {"client_id": "enc:cid", "client_secret": "enc:csec"}
        )

        mock_prisma = MagicMock()
        mock_prisma.db.litellm_mcpservertable.find_unique = AsyncMock(
            return_value=existing_record
        )
        mock_prisma.db.litellm_mcpservertable.update = AsyncMock(
            return_value=MagicMock()
        )

        data = UpdateMCPServerRequest(
            server_id="test-server",
            auth_type="aws_sigv4",
            # No credentials provided
        )

        with patch(
            "litellm.proxy._experimental.mcp_server.db._get_salt_key",
            return_value=None,
        ):
            await update_mcp_server(mock_prisma, data, "test-user")

        data_dict = mock_prisma.db.litellm_mcpservertable.update.call_args[1]["data"]
        # Credentials should be cleared (set to None)
        assert data_dict.get("credentials") is None


class TestInheritCredentials:
    """Test _inherit_credentials_from_existing_server copies AWS fields."""

    def test_inherits_sigv4_credentials(self):
        """SigV4 fields are copied from existing server to inherited credentials."""
        from litellm.proxy.management_endpoints.mcp_management_endpoints import (
            _inherit_credentials_from_existing_server,
        )
        from litellm.proxy._types import NewMCPServerRequest
        from litellm.types.mcp_server.mcp_server_manager import MCPServer

        existing = MCPServer(
            server_id="existing-sigv4",
            name="sigv4_server",
            server_name="sigv4_server",
            url="https://bedrock-agentcore.us-east-1.amazonaws.com/mcp",
            transport=MCPTransport.http,
            auth_type=MCPAuth.aws_sigv4,
            aws_access_key_id="AKIAEXAMPLE",
            aws_secret_access_key="SECRET",
            aws_session_token="TOKEN",
            aws_region_name="us-east-1",
            aws_service_name="bedrock-agentcore",
        )

        payload = NewMCPServerRequest(
            server_id="existing-sigv4",
            server_name="sigv4_server",
            url="https://bedrock-agentcore.us-east-1.amazonaws.com/mcp",
            transport="http",
            auth_type="aws_sigv4",
        )

        with patch(
            "litellm.proxy.management_endpoints.mcp_management_endpoints.global_mcp_server_manager"
        ) as mock_manager:
            mock_manager.get_mcp_server_by_id.return_value = existing
            result = _inherit_credentials_from_existing_server(payload)

        assert result.credentials is not None
        assert result.credentials["aws_access_key_id"] == "AKIAEXAMPLE"
        assert result.credentials["aws_secret_access_key"] == "SECRET"
        assert result.credentials["aws_session_token"] == "TOKEN"
        assert result.credentials["aws_region_name"] == "us-east-1"
        assert result.credentials["aws_service_name"] == "bedrock-agentcore"
