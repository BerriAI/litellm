"""
Tests for AWS SigV4 authentication in MCP client.

Tests the MCPSigV4Auth httpx.Auth subclass that enables per-request
SigV4 signing for Bedrock AgentCore MCP servers.
"""

import pytest
from unittest.mock import patch, MagicMock

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
