"""
Tests for Amazon Bedrock AgentCore Web Search integration.

Mirror of tests/search_tests/test_agentcore_search.py placed in the
test_litellm tree so the AgentCoreSearchConfig transformation is exercised by
the sharded CI (coverage collection runs against this tree).
"""

import json
import os

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

import litellm
from litellm.llms.bedrock.search.transformation import (
    AGENTCORE_DEFAULT_MCP_PROTOCOL_VERSION,
    AgentCoreSearchConfig,
)

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
    async def test_agentcore_search_request_payload(self, monkeypatch):
        """Validates the MCP tools/call payload and SigV4 signing without real AWS calls."""
        monkeypatch.setenv("AGENTCORE_GATEWAY_URL", GATEWAY_URL)

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
            assert call_kwargs["json"] is None

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
        """MCP Streamable HTTP requires accepting both JSON and SSE, and declaring
        the protocol revision the client speaks."""
        config = AgentCoreSearchConfig()
        headers = config.validate_environment(headers={})
        assert headers["Accept"] == "application/json, text/event-stream"
        assert headers["Content-Type"] == "application/json"
        assert headers["MCP-Protocol-Version"] == AGENTCORE_DEFAULT_MCP_PROTOCOL_VERSION

    def test_default_protocol_version_is_the_agentcore_gateway_default(self):
        """A default AgentCore gateway supports only 2025-03-26 and answers
        -32600 to anything newer, so that exact revision must be the default."""
        assert AGENTCORE_DEFAULT_MCP_PROTOCOL_VERSION == "2025-03-26"

    def test_protocol_version_env_override_wins(self):
        """A gateway pinned to a newer supportedVersions list needs the header
        to match, so AGENTCORE_MCP_PROTOCOL_VERSION must override the default."""
        config = AgentCoreSearchConfig()
        with patch.dict(os.environ, {"AGENTCORE_MCP_PROTOCOL_VERSION": "2025-06-18"}):
            headers = config.validate_environment(headers={})
        assert headers["MCP-Protocol-Version"] == "2025-06-18"

    def test_protocol_version_header_survives_signing(self):
        """Both auth paths must keep the MCP-Protocol-Version header on the wire."""
        config = AgentCoreSearchConfig()
        headers = config.validate_environment(headers={})

        bearer_headers, _ = config.sign_request(
            headers=headers,
            optional_params={},
            request_data={"jsonrpc": "2.0"},
            api_base=GATEWAY_URL,
            api_key="test-jwt-token",
        )
        assert bearer_headers["MCP-Protocol-Version"] == AGENTCORE_DEFAULT_MCP_PROTOCOL_VERSION

        with patch.dict(
            os.environ,
            {
                "AWS_ACCESS_KEY_ID": "AKIAIOSFODNN7EXAMPLE",
                "AWS_SECRET_ACCESS_KEY": "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
            },
        ):
            signed_headers, _ = config.sign_request(
                headers=headers,
                optional_params={"aws_region_name": "us-east-1"},
                request_data={"jsonrpc": "2.0"},
                api_base=GATEWAY_URL,
            )
        assert signed_headers["Authorization"].startswith("AWS4-HMAC-SHA256")
        assert signed_headers["MCP-Protocol-Version"] == AGENTCORE_DEFAULT_MCP_PROTOCOL_VERSION

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

    def test_transform_search_response_raises_on_tool_error(self):
        """A failed tools/call comes back as HTTP 200 with result.isError; it must not be
        reported to the caller as a successful search with zero results."""
        config = AgentCoreSearchConfig()
        mock_response = _make_mock_response(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "result": {
                    "isError": True,
                    "content": [{"type": "text", "text": "AccessDeniedException: not authorized"}],
                },
            }
        )
        with pytest.raises(Exception, match="AccessDeniedException"):
            config.transform_search_response(raw_response=mock_response, logging_obj=MagicMock())

    def test_transform_search_response_reads_structured_content(self):
        """Connector 1.1.0+ puts the machine-readable results in structuredContent and may
        leave the text block as prose, which must not come back as an empty result list."""
        config = AgentCoreSearchConfig()
        mock_response = _make_mock_response(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "result": {
                    "content": [{"type": "text", "text": "Here is a prose summary of what I found."}],
                    "structuredContent": {"id": "824f89d0", "results": MCP_RESULTS},
                },
            }
        )

        response = config.transform_search_response(raw_response=mock_response, logging_obj=MagicMock())
        assert [result.title for result in response.results] == ["Test Result 1", "Test Result 2"]
        assert response.results[0].url == "https://example.com/1"
        assert response.results[0].snippet == "Snippet for result 1"
        assert response.results[0].date == "2026-06-16"

    def test_transform_search_response_does_not_duplicate_structured_content(self):
        """1.1.0+ repeats the same results in both places, so parsing both would double them."""
        config = AgentCoreSearchConfig()
        body = _mcp_response_body()
        body["result"]["structuredContent"] = {"results": MCP_RESULTS}
        mock_response = _make_mock_response(body)

        response = config.transform_search_response(raw_response=mock_response, logging_obj=MagicMock())
        assert len(response.results) == 2

    def test_transform_search_response_parses_crlf_framed_sse(self):
        """SSE streams may be CRLF framed; events must still split into separate events."""
        config = AgentCoreSearchConfig()
        progress = {"jsonrpc": "2.0", "method": "notifications/progress", "params": {"progress": 1}}
        sse_text = (
            f"event: message\r\ndata: {json.dumps(progress)}\r\n\r\n"
            f"event: message\r\ndata: {json.dumps(_mcp_response_body())}\r\n\r\n"
        )
        mock_response = _make_mock_response(text=sse_text)

        response = config.transform_search_response(raw_response=mock_response, logging_obj=MagicMock())
        assert len(response.results) == 2
        assert response.results[0].title == "Test Result 1"

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

    def test_sign_request_uses_bearer_token_from_env(self, monkeypatch):
        """Server token is attached when the request targets the configured gateway host."""
        config = AgentCoreSearchConfig()
        monkeypatch.setenv("AGENTCORE_GATEWAY_TOKEN", "env-jwt-token")
        monkeypatch.setenv("AGENTCORE_GATEWAY_URL", GATEWAY_URL)
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

    def test_sign_request_refuses_server_token_to_untrusted_host(self, monkeypatch):
        """Server-managed token must not be sent to a caller-chosen api_base."""
        config = AgentCoreSearchConfig()
        monkeypatch.setenv("AGENTCORE_GATEWAY_TOKEN", "env-jwt-token")
        monkeypatch.setenv("AGENTCORE_GATEWAY_URL", GATEWAY_URL)
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

    def test_sign_request_uses_env_token_for_gateway_api_base_without_gateway_url(self, monkeypatch):
        """api_base pointing at a real gateway is a trusted destination for the env token,
        so operators configuring api_base in yaml don't also need AGENTCORE_GATEWAY_URL."""
        config = AgentCoreSearchConfig()
        monkeypatch.setenv("AGENTCORE_GATEWAY_TOKEN", "env-jwt-token")
        os.environ.pop("AGENTCORE_GATEWAY_URL", None)
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

    @pytest.mark.parametrize(
        "untrusted_api_base",
        [
            "https://attacker.example.com/mcp",
            # gateway hostname in the path/query must not pass for the host
            "https://attacker.example.com/gw.gateway.bedrock-agentcore.us-east-1.amazonaws.com/mcp",
        ],
    )
    def test_sign_request_refuses_sigv4_to_untrusted_host(self, untrusted_api_base, monkeypatch):
        """A SigV4 signature carries the proxy's credential scope and session token, so it
        must never be sent to a host that is not the operator's gateway."""
        config = AgentCoreSearchConfig()
        os.environ.pop("AGENTCORE_GATEWAY_TOKEN", None)
        monkeypatch.setenv("AGENTCORE_GATEWAY_URL", GATEWAY_URL)
        try:
            with patch.object(
                AgentCoreSearchConfig.__mro__[2],  # BaseAWSLLM
                "_sign_request",
                return_value=({}, b"{}"),
            ) as mock_base_sign:
                with pytest.raises(ValueError, match="Refusing to send"):
                    config.sign_request(
                        headers={},
                        optional_params={"aws_region_name": "us-east-1"},
                        request_data={"jsonrpc": "2.0"},
                        api_base=untrusted_api_base,
                    )
                mock_base_sign.assert_not_called()
        finally:
            os.environ.pop("AGENTCORE_GATEWAY_URL", None)

    @pytest.mark.parametrize(
        "plaintext_api_base",
        [
            "http://gw.gateway.bedrock-agentcore.us-east-1.amazonaws.com/mcp",
            "http://internal-gateway.corp/mcp",
        ],
    )
    def test_sign_request_refuses_server_token_over_plaintext_http(self, plaintext_api_base, monkeypatch):
        """A trusted hostname over plain http would expose the bearer token to
        network observers, so credentials only ride https (or localhost)."""
        config = AgentCoreSearchConfig()
        monkeypatch.setenv("AGENTCORE_GATEWAY_TOKEN", "env-jwt-token")
        monkeypatch.setenv("AGENTCORE_GATEWAY_URL", plaintext_api_base)
        try:
            with pytest.raises(ValueError, match="plaintext"):
                config.sign_request(
                    headers={},
                    optional_params={},
                    request_data={"jsonrpc": "2.0"},
                    api_base=plaintext_api_base,
                )
        finally:
            os.environ.pop("AGENTCORE_GATEWAY_TOKEN", None)
            os.environ.pop("AGENTCORE_GATEWAY_URL", None)

    def test_sign_request_refuses_sigv4_over_plaintext_http(self):
        """Same for SigV4: a signature over plain http is replayable by observers."""
        config = AgentCoreSearchConfig()
        os.environ.pop("AGENTCORE_GATEWAY_TOKEN", None)
        with patch.object(
            AgentCoreSearchConfig.__mro__[2],  # BaseAWSLLM
            "_sign_request",
            return_value=({}, b"{}"),
        ) as mock_base_sign:
            with pytest.raises(ValueError, match="plaintext"):
                config.sign_request(
                    headers={},
                    optional_params={"aws_region_name": "us-east-1"},
                    request_data={"jsonrpc": "2.0"},
                    api_base="http://gw.gateway.bedrock-agentcore.us-east-1.amazonaws.com/mcp",
                )
            mock_base_sign.assert_not_called()

    def test_sign_request_allows_plain_http_for_localhost(self, monkeypatch):
        """Local development against an MCP stub on 127.0.0.1 keeps working."""
        config = AgentCoreSearchConfig()
        monkeypatch.setenv("AGENTCORE_GATEWAY_TOKEN", "env-jwt-token")
        monkeypatch.setenv("AGENTCORE_GATEWAY_URL", "http://127.0.0.1:8931/mcp")
        try:
            headers, _ = config.sign_request(
                headers={},
                optional_params={},
                request_data={"jsonrpc": "2.0"},
                api_base="http://127.0.0.1:8931/mcp",
            )
            assert headers["Authorization"] == "Bearer env-jwt-token"
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

    def test_sign_request_custom_hostname_requires_region(self, monkeypatch):
        """Custom hostname + empty AWS config chain → clear error, no guessed region."""
        config = AgentCoreSearchConfig()
        custom_url = "https://gateway.internal.example.com/mcp"
        monkeypatch.setenv("AGENTCORE_GATEWAY_URL", custom_url)

        mock_session = MagicMock()
        mock_session.region_name = None  # nothing configured anywhere
        try:
            with patch("boto3.Session", return_value=mock_session):
                with pytest.raises(ValueError, match="signing region"):
                    config.sign_request(
                        headers={},
                        optional_params={},
                        request_data={"jsonrpc": "2.0"},
                        api_base=custom_url,
                    )
        finally:
            os.environ.pop("AGENTCORE_GATEWAY_URL", None)

    def test_sign_request_custom_hostname_uses_shared_config_region(self, monkeypatch):
        """Custom hostname + region from AWS shared config (profile) must be honored."""
        config = AgentCoreSearchConfig()
        custom_url = "https://gateway.internal.example.com/mcp"
        monkeypatch.setenv("AGENTCORE_GATEWAY_URL", custom_url)

        mock_session = MagicMock()
        mock_session.region_name = "eu-west-1"  # e.g. from ~/.aws/config profile
        try:
            with (
                patch("boto3.Session", return_value=mock_session),
                patch.object(
                    AgentCoreSearchConfig.__mro__[2],  # BaseAWSLLM
                    "_sign_request",
                    return_value=({}, b"{}"),
                ) as mock_base_sign,
            ):
                config.sign_request(
                    headers={},
                    optional_params={},
                    request_data={"jsonrpc": "2.0"},
                    api_base=custom_url,
                )
                assert mock_base_sign.call_args.kwargs["optional_params"]["aws_region_name"] == "eu-west-1"
        finally:
            os.environ.pop("AGENTCORE_GATEWAY_URL", None)

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


class TestAgentCoreSearchEdgeCases:
    """Branch coverage for response parsing and error mapping."""

    def test_transform_search_response_skips_non_text_and_bad_json_blocks(self):
        """Non-text blocks and unparseable text blocks are skipped, not fatal."""
        config = AgentCoreSearchConfig()
        body = {
            "jsonrpc": "2.0",
            "id": 1,
            "result": {
                "content": [
                    {"type": "image", "data": "..."},
                    {"type": "text", "text": "not-json"},
                    {"type": "text", "text": json.dumps(["scalar", {"title": "T", "url": "u", "text": "s"}])},
                ]
            },
        }
        mock_response = _make_mock_response(body)

        response = config.transform_search_response(raw_response=mock_response, logging_obj=MagicMock())
        # only the one dict item survives; non-dict list entries are skipped
        assert len(response.results) == 1
        assert response.results[0].title == "T"

    def test_parse_mcp_body_sse_without_json_frame_raises(self):
        """An SSE stream carrying no parseable JSON object is a 502."""
        config = AgentCoreSearchConfig()
        mock_response = _make_mock_response(text="event: ping\ndata: not-json\n\n")
        with pytest.raises(Exception, match="SSE without a JSON data frame"):
            config._parse_mcp_body(mock_response)

    def test_parse_mcp_body_returns_last_event_when_no_result_frame(self):
        """A stream of only notifications returns the last parsed event."""
        config = AgentCoreSearchConfig()
        note = {"jsonrpc": "2.0", "method": "notifications/progress"}
        mock_response = _make_mock_response(text=f"data: {json.dumps(note)}\n\n")
        assert config._parse_mcp_body(mock_response) == note

    def test_sign_request_rejects_list_request_body(self):
        config = AgentCoreSearchConfig()
        with pytest.raises(TypeError, match="single dict"):
            config.sign_request(
                headers={},
                optional_params={},
                request_data=[{"jsonrpc": "2.0"}],
                api_base=GATEWAY_URL,
            )

    def test_get_error_class_maps_status_and_message(self):
        config = AgentCoreSearchConfig()
        err = config.get_error_class(error_message="boom", status_code=503, headers={})
        assert getattr(err, "status_code", None) == 503
        assert "boom" in str(err)

    def test_search_cost_lookup_is_mapped(self, monkeypatch):
        """Assert against the map in this checkout: the remote cost map litellm loads by
        default only carries providers already released."""
        from litellm.litellm_core_utils.get_model_cost_map import GetModelCostMap
        from litellm.search.cost_calculator import search_provider_cost_per_query

        monkeypatch.setattr(litellm, "model_cost", GetModelCostMap.load_local_model_cost_map())
        assert search_provider_cost_per_query(model="agentcore/search", custom_llm_provider="agentcore") == (0.0, 0.0)
