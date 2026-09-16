import pytest

from litellm.proxy.management_endpoints.mcp_connector_import import (
    ConnectorConversionError,
    ConvertedConnector,
    MCPConnectorImportRequest,
    convert_connector_entries,
    sanitize_connector_name,
)
from litellm.types.mcp import MCPAuth, MCPTransport


def _single(payload: dict) -> ConvertedConnector | ConnectorConversionError:
    results = convert_connector_entries(MCPConnectorImportRequest.model_validate(payload))
    assert len(results) == 1
    return results[0]


class TestSanitizeConnectorName:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("my-server", "my_server"),
            ("  spaced name  ", "spaced_name"),
            ("already_ok", "already_ok"),
            ("a.b.c", "a_b_c"),
            ("---", ""),
        ],
    )
    def test_sanitizes_to_mcp_safe_names(self, raw, expected):
        assert sanitize_connector_name(raw) == expected


class TestConvertMcpServersMapping:
    def test_url_connector_with_authorization_token(self):
        result = _single(
            {
                "mcpServers": {
                    "github-mcp": {
                        "url": "https://api.example.com/mcp",
                        "authorization_token": "secret-token",
                        "headers": {"X-Env": "prod"},
                        "description": "GitHub connector",
                    }
                }
            }
        )
        assert isinstance(result, ConvertedConnector)
        assert result.request.server_name == "github_mcp"
        assert result.request.alias == "github_mcp"
        assert result.request.transport == MCPTransport.http
        assert result.request.url == "https://api.example.com/mcp"
        assert result.request.auth_type == MCPAuth.bearer_token
        assert result.request.credentials == {"auth_value": "secret-token"}
        assert result.request.static_headers == {"X-Env": "prod"}
        assert result.request.description == "GitHub connector"

    def test_authorization_header_becomes_bearer_credentials(self):
        result = _single(
            {
                "mcpServers": {
                    "srv": {
                        "url": "https://x.example/mcp",
                        "headers": {"Authorization": "Bearer header-token", "X-Env": "prod"},
                    }
                }
            }
        )
        assert isinstance(result, ConvertedConnector)
        assert result.request.auth_type == MCPAuth.bearer_token
        assert result.request.credentials == {"auth_value": "header-token"}
        assert result.request.static_headers == {"X-Env": "prod"}

    def test_authorization_header_without_bearer_prefix_is_sent_verbatim(self):
        result = _single(
            {"mcpServers": {"srv": {"url": "https://x.example/mcp", "headers": {"authorization": "raw-token"}}}}
        )
        assert isinstance(result, ConvertedConnector)
        assert result.request.auth_type == MCPAuth.authorization
        assert result.request.credentials == {"auth_value": "raw-token"}
        assert result.request.static_headers is None

    def test_basic_authorization_header_is_sent_verbatim(self):
        result = _single(
            {"mcpServers": {"srv": {"url": "https://x.example/mcp", "headers": {"Authorization": "Basic dXNlcjpwdw=="}}}}
        )
        assert isinstance(result, ConvertedConnector)
        assert result.request.auth_type == MCPAuth.authorization
        assert result.request.credentials == {"auth_value": "Basic dXNlcjpwdw=="}
        assert result.request.static_headers is None

    def test_authorization_token_wins_over_authorization_header(self):
        result = _single(
            {
                "mcpServers": {
                    "srv": {
                        "url": "https://x.example/mcp",
                        "authorization_token": "explicit-token",
                        "headers": {"Authorization": "Bearer header-token"},
                    }
                }
            }
        )
        assert isinstance(result, ConvertedConnector)
        assert result.request.auth_type == MCPAuth.bearer_token
        assert result.request.credentials == {"auth_value": "explicit-token"}
        assert result.request.static_headers is None

    def test_camel_case_authorization_token_alias(self):
        result = _single(
            {"mcpServers": {"srv": {"url": "https://x.example/mcp", "authorizationToken": "tok"}}}
        )
        assert isinstance(result, ConvertedConnector)
        assert result.request.credentials == {"auth_value": "tok"}

    def test_url_connector_without_token_uses_no_auth(self):
        result = _single({"mcpServers": {"open": {"url": "https://open.example/mcp"}}})
        assert isinstance(result, ConvertedConnector)
        assert result.request.auth_type == MCPAuth.none
        assert result.request.credentials is None

    def test_sse_type_maps_to_sse_transport(self):
        result = _single({"mcpServers": {"legacy": {"type": "sse", "url": "https://sse.example/mcp"}}})
        assert isinstance(result, ConvertedConnector)
        assert result.request.transport == MCPTransport.sse

    def test_stdio_connector(self):
        result = _single(
            {
                "mcpServers": {
                    "local": {
                        "command": "npx",
                        "args": ["-y", "@example/mcp-server"],
                        "env": {"API_KEY": "value"},
                    }
                }
            }
        )
        assert isinstance(result, ConvertedConnector)
        assert result.request.transport == MCPTransport.stdio
        assert result.request.command == "npx"
        assert result.request.args == ["-y", "@example/mcp-server"]
        assert result.request.env == {"API_KEY": "value"}

    def test_disallowed_stdio_command_returns_error(self):
        result = _single({"mcpServers": {"evil": {"command": "rm", "args": ["-rf", "/"]}}})
        assert isinstance(result, ConnectorConversionError)
        assert "not in the allowed commands list" in result.error

    def test_unsupported_type_returns_error(self):
        result = _single({"mcpServers": {"ws": {"type": "websocket", "url": "wss://x.example"}}})
        assert isinstance(result, ConnectorConversionError)
        assert "Unsupported connector type" in result.error

    def test_missing_url_and_command_returns_error(self):
        result = _single({"mcpServers": {"empty": {}}})
        assert isinstance(result, ConnectorConversionError)
        assert "either a url or a command" in result.error

    def test_url_and_command_together_returns_error(self):
        result = _single({"mcpServers": {"both": {"url": "https://x.example/mcp", "command": "npx"}}})
        assert isinstance(result, ConnectorConversionError)
        assert "both a url and a command" in result.error

    def test_name_empty_after_sanitization_returns_error(self):
        result = _single({"mcpServers": {"---": {"url": "https://x.example/mcp"}}})
        assert isinstance(result, ConnectorConversionError)
        assert "empty after sanitization" in result.error


class TestConvertMcpServersList:
    def test_anthropic_messages_api_list_shape(self):
        result = _single(
            {
                "mcp_servers": [
                    {
                        "type": "url",
                        "url": "https://mcp.example.com/sse",
                        "name": "deepwiki",
                        "authorization_token": "tok",
                    }
                ]
            }
        )
        assert isinstance(result, ConvertedConnector)
        assert result.request.server_name == "deepwiki"
        assert result.request.transport == MCPTransport.http
        assert result.request.credentials == {"auth_value": "tok"}

    def test_list_entry_without_name_returns_error(self):
        result = _single({"mcp_servers": [{"type": "url", "url": "https://x.example/mcp"}]})
        assert isinstance(result, ConnectorConversionError)
        assert "must have a name" in result.error

    def test_partial_conversion_preserves_per_entry_results(self):
        results = convert_connector_entries(
            MCPConnectorImportRequest.model_validate(
                {
                    "mcpServers": {
                        "good": {"url": "https://good.example/mcp"},
                        "bad": {"type": "websocket", "url": "wss://bad.example"},
                    }
                }
            )
        )
        assert len(results) == 2
        assert isinstance(results[0], ConvertedConnector)
        assert isinstance(results[1], ConnectorConversionError)
