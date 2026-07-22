"""
Tests for Amazon Bedrock AgentCore Web Search integration.
"""

import json
import os
import sys
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

sys.path.insert(0, os.path.abspath("../.."))

import litellm
from litellm.llms.bedrock.search.transformation import AgentCoreSearchConfig

GATEWAY_URL = "https://testgateway-abc123.gateway.bedrock-agentcore.us-east-1.amazonaws.com/mcp"

MCP_RESULTS = [
    {
        "title": "Test Result 1",
        "url": "https://example.com/1",
        "text": "Snippet for result 1",
        "publishedDate": "2026-06-16",
    },
    {
        "title": "Test Result 2",
        "url": "https://example.com/2",
        "text": "Snippet for result 2",
    },
]


def _mcp_response_body() -> dict:
    return {
        "jsonrpc": "2.0",
        "id": 1,
        "result": {"content": [{"type": "text", "text": json.dumps(MCP_RESULTS)}]},
    }


def _make_mock_response(json_body: dict = None, text: str = None) -> MagicMock:
    mock_response = MagicMock()
    mock_response.status_code = 200
    if text is not None:
        mock_response.text = text
    else:
        mock_response.text = json.dumps(json_body)
        mock_response.json.return_value = json_body
    return mock_response


class TestAgentCoreSearch:
    """
    Tests for AgentCore Web Search functionality with mocked network/signing.
    """

    @pytest.mark.asyncio
    async def test_agentcore_search_request_payload(self):
        """Validates the MCP tools/call payload and SigV4 signing without real AWS calls."""
        os.environ["AGENTCORE_GATEWAY_URL"] = GATEWAY_URL

        mock_response = _make_mock_response(_mcp_response_body())

        with (
            patch(
                "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post",
                new_callable=AsyncMock,
            ) as mock_post,
            patch.object(
                AgentCoreSearchConfig,
                "_sign_request",
                return_value=(
                    {"Authorization": "AWS4-HMAC-SHA256 test", "Content-Type": "application/json"},
                    json.dumps({"signed": True}).encode(),
                ),
            ) as mock_sign,
        ):
            mock_post.return_value = mock_response

            response = await litellm.asearch(
                query="latest developments in AI",
                search_provider="agentcore",
                max_results=5,
            )

            mock_post.assert_called_once()
            call_kwargs = mock_post.call_args.kwargs
            assert call_kwargs["url"] == GATEWAY_URL
            # Signed body must be sent verbatim
            assert call_kwargs["data"] == json.dumps({"signed": True}).encode()
            assert "json" not in call_kwargs

            # Signing was invoked with the MCP request
            mock_sign.assert_called_once()
            sign_kwargs = mock_sign.call_args.kwargs
            request_data = sign_kwargs["request_data"]
            assert request_data["method"] == "tools/call"
            assert request_data["params"]["name"] == "web-search-tool___WebSearch"
            assert request_data["params"]["arguments"]["query"] == "latest developments in AI"
            assert request_data["params"]["arguments"]["maxResults"] == 5
            assert sign_kwargs["service_name"] == "bedrock-agentcore"

            assert len(response.results) == 2
            assert response.results[0].title == "Test Result 1"
            assert response.results[0].url == "https://example.com/1"
            assert response.results[0].snippet == "Snippet for result 1"
            assert response.results[0].date == "2026-06-16"

    def test_transform_search_request_query_truncation(self):
        """AgentCore rejects queries > 200 chars; the request must truncate."""
        config = AgentCoreSearchConfig()
        long_query = "a" * 300
        data = config.transform_search_request(query=long_query, optional_params={})
        assert len(data["params"]["arguments"]["query"]) == 200

    def test_transform_search_request_joins_list_queries(self):
        config = AgentCoreSearchConfig()
        data = config.transform_search_request(query=["foo", "bar"], optional_params={})
        assert data["params"]["arguments"]["query"] == "foo bar"

    def test_transform_search_request_custom_tool_name(self):
        config = AgentCoreSearchConfig()
        data = config.transform_search_request(query="q", optional_params={"tool_name": "my-target___WebSearch"})
        assert data["params"]["name"] == "my-target___WebSearch"

    def test_transform_search_request_rejects_non_websearch_tool_name(self):
        """A caller-supplied tool_name must not reach other tools on the gateway."""
        config = AgentCoreSearchConfig()
        with pytest.raises(ValueError, match="must end with"):
            config.transform_search_request(query="q", optional_params={"tool_name": "admin-target___DeleteUser"})

    def test_transform_search_request_sends_documented_default_max_results(self):
        """The documented default of 10 is sent explicitly, not left to the gateway."""
        config = AgentCoreSearchConfig()
        data = config.transform_search_request(query="q", optional_params={})
        assert data["params"]["arguments"]["maxResults"] == 10

    def test_get_complete_url_requires_gateway_url(self):
        config = AgentCoreSearchConfig()
        os.environ.pop("AGENTCORE_GATEWAY_URL", None)
        with pytest.raises(ValueError, match="AGENTCORE_GATEWAY_URL"):
            config.get_complete_url(api_base=None, optional_params={})

    def test_get_complete_url_prefers_api_base(self):
        config = AgentCoreSearchConfig()
        assert config.get_complete_url(api_base=GATEWAY_URL, optional_params={}) == GATEWAY_URL

    def test_validate_environment_sets_mcp_headers(self):
        """MCP Streamable HTTP requires accepting both JSON and SSE."""
        config = AgentCoreSearchConfig()
        headers = config.validate_environment(headers={})
        assert headers["Accept"] == "application/json, text/event-stream"
        assert headers["Content-Type"] == "application/json"

    def test_transform_search_response_parses_sse_frame(self):
        """Gateway may answer with an SSE-framed JSON-RPC message."""
        config = AgentCoreSearchConfig()
        body = _mcp_response_body()
        sse_text = f"event: message\ndata: {json.dumps(body)}\n\n"
        mock_response = _make_mock_response(text=sse_text)

        response = config.transform_search_response(raw_response=mock_response, logging_obj=MagicMock())
        assert len(response.results) == 2
        assert response.results[1].url == "https://example.com/2"

    def test_transform_search_response_parses_multiline_sse_data(self):
        """SSE data may be split across several data: lines (joined per spec)."""
        config = AgentCoreSearchConfig()
        pretty = json.dumps(_mcp_response_body(), indent=2)
        sse_text = "event: message\n" + "\n".join(f"data: {line}" for line in pretty.splitlines()) + "\n\n"
        mock_response = _make_mock_response(text=sse_text)

        response = config.transform_search_response(raw_response=mock_response, logging_obj=MagicMock())
        assert len(response.results) == 2

    def test_transform_search_response_skips_progress_events(self):
        """A progress notification before the JSON-RPC result must not shadow it."""
        config = AgentCoreSearchConfig()
        progress = {"jsonrpc": "2.0", "method": "notifications/progress", "params": {"progress": 1}}
        sse_text = (
            f"event: message\ndata: {json.dumps(progress)}\n\n"
            f"event: message\ndata: {json.dumps(_mcp_response_body())}\n\n"
        )
        mock_response = _make_mock_response(text=sse_text)

        response = config.transform_search_response(raw_response=mock_response, logging_obj=MagicMock())
        assert len(response.results) == 2

    def test_transform_search_response_raises_on_mcp_error(self):
        config = AgentCoreSearchConfig()
        mock_response = _make_mock_response(
            {"jsonrpc": "2.0", "id": 1, "error": {"code": -32601, "message": "tool not found"}}
        )
        with pytest.raises(Exception, match="tool not found"):
            config.transform_search_response(raw_response=mock_response, logging_obj=MagicMock())

    def test_sign_request_uses_bearer_token_when_api_key_set(self):
        """CUSTOM_JWT gateways: api_key is sent as a bearer token, no SigV4."""
        config = AgentCoreSearchConfig()
        request_data = {"jsonrpc": "2.0", "id": 1}

        headers, signed_body = config.sign_request(
            headers={"Content-Type": "application/json"},
            optional_params={},
            request_data=request_data,
            api_base=GATEWAY_URL,
            api_key="test-jwt-token",
        )
        assert headers["Authorization"] == "Bearer test-jwt-token"
        assert signed_body == json.dumps(request_data).encode()

    def test_sign_request_uses_bearer_token_from_env(self):
        """Server token is attached when the request targets the configured gateway host."""
        config = AgentCoreSearchConfig()
        os.environ["AGENTCORE_GATEWAY_TOKEN"] = "env-jwt-token"
        os.environ["AGENTCORE_GATEWAY_URL"] = GATEWAY_URL
        try:
            headers, _ = config.sign_request(
                headers={},
                optional_params={},
                request_data={"jsonrpc": "2.0"},
                api_base=GATEWAY_URL,
            )
            assert headers["Authorization"] == "Bearer env-jwt-token"
        finally:
            os.environ.pop("AGENTCORE_GATEWAY_TOKEN", None)
            os.environ.pop("AGENTCORE_GATEWAY_URL", None)

    def test_sign_request_refuses_server_token_to_untrusted_host(self):
        """Server-managed token must not be sent to a caller-chosen api_base."""
        config = AgentCoreSearchConfig()
        os.environ["AGENTCORE_GATEWAY_TOKEN"] = "env-jwt-token"
        os.environ["AGENTCORE_GATEWAY_URL"] = GATEWAY_URL
        try:
            with pytest.raises(ValueError, match="Refusing to send"):
                config.sign_request(
                    headers={},
                    optional_params={},
                    request_data={"jsonrpc": "2.0"},
                    api_base="https://attacker.example.com/mcp",
                )
        finally:
            os.environ.pop("AGENTCORE_GATEWAY_TOKEN", None)
            os.environ.pop("AGENTCORE_GATEWAY_URL", None)

    def test_sign_request_does_not_leak_bedrock_bearer_token(self):
        """AWS_BEARER_TOKEN_BEDROCK is a Bedrock Runtime credential — it must not
        replace SigV4 on requests to an AgentCore gateway."""
        config = AgentCoreSearchConfig()

        with patch.object(
            AgentCoreSearchConfig.__mro__[2],  # BaseAWSLLM
            "_sign_request",
            return_value=({}, b"{}"),
        ) as mock_base_sign:
            config.sign_request(
                headers={},
                optional_params={},
                request_data={"jsonrpc": "2.0"},
                api_base=GATEWAY_URL,
            )
            # api_key="" (falsy, not None) disables the base class's
            # AWS_BEARER_TOKEN_BEDROCK env fallback.
            assert mock_base_sign.call_args.kwargs["api_key"] == ""

    def test_sign_request_custom_hostname_requires_region(self):
        """Non-standard hostnames can't yield a signing region — require it explicitly."""
        config = AgentCoreSearchConfig()
        saved = {var: os.environ.pop(var, None) for var in ("AWS_REGION", "AWS_REGION_NAME", "AWS_DEFAULT_REGION")}
        try:
            with pytest.raises(ValueError, match="signing region"):
                config.sign_request(
                    headers={},
                    optional_params={},
                    request_data={"jsonrpc": "2.0"},
                    api_base="https://gateway.internal.example.com/mcp",
                )
        finally:
            for var, val in saved.items():
                if val is not None:
                    os.environ[var] = val

    def test_sign_request_passes_explicit_aws_credentials(self):
        """Explicit aws_* params (e.g. from a proxy search_tools entry) reach the signer."""
        config = AgentCoreSearchConfig()

        with patch.object(
            AgentCoreSearchConfig.__mro__[2],  # BaseAWSLLM
            "_sign_request",
            return_value=({}, b"{}"),
        ) as mock_base_sign:
            config.sign_request(
                headers={},
                optional_params={
                    "aws_access_key_id": "AKIATEST",
                    "aws_secret_access_key": "secret",
                    "aws_session_token": "token",
                },
                request_data={"jsonrpc": "2.0"},
                api_base=GATEWAY_URL,
            )
            passed = mock_base_sign.call_args.kwargs["optional_params"]
            assert passed["aws_access_key_id"] == "AKIATEST"
            assert passed["aws_secret_access_key"] == "secret"
            assert passed["aws_session_token"] == "token"

    def test_sign_request_derives_region_from_gateway_url(self):
        """Signing region must come from the gateway URL, not the caller's default region."""
        config = AgentCoreSearchConfig()
        eu_url = "https://gw-x.gateway.bedrock-agentcore.eu-central-1.amazonaws.com/mcp"

        with patch.object(
            AgentCoreSearchConfig.__mro__[2],  # BaseAWSLLM
            "_sign_request",
            return_value=({}, b"{}"),
        ) as mock_base_sign:
            config.sign_request(
                headers={},
                optional_params={},
                request_data={"jsonrpc": "2.0"},
                api_base=eu_url,
            )
            assert mock_base_sign.call_args.kwargs["optional_params"]["aws_region_name"] == "eu-central-1"
