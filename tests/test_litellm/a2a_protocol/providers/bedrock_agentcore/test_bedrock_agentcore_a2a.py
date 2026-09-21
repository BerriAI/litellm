"""
Tests for Bedrock AgentCore A2A provider.

Verifies that:
- JSON-RPC envelopes are preserved (not stripped by the completion bridge)
- URLs are derived from the model ARN
- Auth uses JWT Bearer or SigV4
- Config manager routes "bedrock" correctly
- Handler passes litellm_params and allows api_base=None
"""

import json

import httpx
import pytest
import respx
from unittest.mock import AsyncMock, MagicMock, patch


SAMPLE_ARN = "arn:aws:bedrock-agentcore:us-west-2:123456789:runtime/my_agent"
SAMPLE_MODEL = f"bedrock/agentcore/{SAMPLE_ARN}"
SAMPLE_PARAMS = {
    "message": {
        "role": "user",
        "parts": [{"kind": "text", "text": "what is 1+1?"}],
        "messageId": "msg-001",
    }
}
SAMPLE_LITELLM_PARAMS = {
    "model": SAMPLE_MODEL,
    "custom_llm_provider": "bedrock",
    "api_key": "test-jwt-token",
}


class TestTransformation:
    """Test URL construction and JSON-RPC envelope building."""

    def test_json_rpc_envelope_structure(self):
        """Verify JSON-RPC body has jsonrpc, method, id, and params."""
        from litellm.a2a_protocol.providers.bedrock_agentcore.transformation import (
            BedrockAgentCoreA2ATransformation,
        )

        url, headers, body = (
            BedrockAgentCoreA2ATransformation.get_url_and_signed_request(
                request_id="req-001",
                params=SAMPLE_PARAMS,
                litellm_params=SAMPLE_LITELLM_PARAMS,
                method="message/send",
            )
        )
        body_dict = json.loads(body)
        assert body_dict["jsonrpc"] == "2.0"
        assert body_dict["method"] == "message/send"
        assert body_dict["id"] == "req-001"
        assert body_dict["params"] == SAMPLE_PARAMS

    def test_url_derived_from_arn(self):
        """Verify URL is constructed from the ARN, not from api_base."""
        from litellm.a2a_protocol.providers.bedrock_agentcore.transformation import (
            BedrockAgentCoreA2ATransformation,
        )

        url, _, _ = BedrockAgentCoreA2ATransformation.get_url_and_signed_request(
            request_id="req-001",
            params=SAMPLE_PARAMS,
            litellm_params=SAMPLE_LITELLM_PARAMS,
        )
        assert "bedrock-agentcore.us-west-2.amazonaws.com" in url
        assert "/runtimes/" in url
        assert "/invocations" in url

    def test_jwt_auth_uses_bearer_header(self):
        """When api_key is set, Authorization header uses Bearer token."""
        from litellm.a2a_protocol.providers.bedrock_agentcore.transformation import (
            BedrockAgentCoreA2ATransformation,
        )

        _, headers, _ = BedrockAgentCoreA2ATransformation.get_url_and_signed_request(
            request_id="req-001",
            params=SAMPLE_PARAMS,
            litellm_params=SAMPLE_LITELLM_PARAMS,
        )
        assert headers["Authorization"] == "Bearer test-jwt-token"

    def test_session_id_header_set(self):
        """Verify X-Amzn-Bedrock-AgentCore-Runtime-Session-Id is set."""
        from litellm.a2a_protocol.providers.bedrock_agentcore.transformation import (
            BedrockAgentCoreA2ATransformation,
        )

        _, headers, _ = BedrockAgentCoreA2ATransformation.get_url_and_signed_request(
            request_id="req-001",
            params=SAMPLE_PARAMS,
            litellm_params=SAMPLE_LITELLM_PARAMS,
        )
        session_id = headers.get("X-Amzn-Bedrock-AgentCore-Runtime-Session-Id", "")
        assert len(session_id) >= 33

    def test_custom_session_id_header(self):
        """Verify custom runtimeSessionId is used when provided."""
        from litellm.a2a_protocol.providers.bedrock_agentcore.transformation import (
            BedrockAgentCoreA2ATransformation,
        )

        params_with_session = {**SAMPLE_LITELLM_PARAMS, "runtimeSessionId": "a" * 40}
        _, headers, _ = BedrockAgentCoreA2ATransformation.get_url_and_signed_request(
            request_id="req-001",
            params=SAMPLE_PARAMS,
            litellm_params=params_with_session,
        )
        assert headers["X-Amzn-Bedrock-AgentCore-Runtime-Session-Id"] == "a" * 40

    def test_agent_extra_headers_merged_into_signed_headers_jwt(self):
        """agent_extra_headers should appear on the outbound request (JWT path)."""
        from litellm.a2a_protocol.providers.bedrock_agentcore.transformation import (
            BedrockAgentCoreA2ATransformation,
        )

        _, headers, _ = BedrockAgentCoreA2ATransformation.get_url_and_signed_request(
            request_id="req-001",
            params=SAMPLE_PARAMS,
            litellm_params=SAMPLE_LITELLM_PARAMS,
            agent_extra_headers={"x-mcp-token": "mcp-abc", "x-tenant": "t1"},
        )
        assert headers["x-mcp-token"] == "mcp-abc"
        assert headers["x-tenant"] == "t1"

    def test_agent_extra_headers_signed_for_sigv4(self):
        """agent_extra_headers must be present in the dict passed to _sign_request."""
        from litellm.a2a_protocol.providers.bedrock_agentcore.transformation import (
            BedrockAgentCoreA2ATransformation,
        )

        litellm_params_no_key = {
            "model": SAMPLE_MODEL,
            "custom_llm_provider": "bedrock",
            "aws_access_key_id": "AKIAIOSFODNN7EXAMPLE",
            "aws_secret_access_key": "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
            "aws_region_name": "us-west-2",
        }

        captured: dict = {}

        def fake_sign(self, headers, **kwargs):
            captured.update(headers)
            return headers, b'{"jsonrpc":"2.0"}'

        with patch(
            "litellm.llms.bedrock.chat.agentcore.transformation.AmazonAgentCoreConfig._sign_request",
            new=fake_sign,
        ):
            BedrockAgentCoreA2ATransformation.get_url_and_signed_request(
                request_id="req-001",
                params=SAMPLE_PARAMS,
                litellm_params=litellm_params_no_key,
                agent_extra_headers={"x-mcp-token": "mcp-abc"},
            )
        assert captured.get("x-mcp-token") == "mcp-abc"

    def test_reserved_headers_filtered_from_agent_extra_headers(self):
        """
        Reserved AWS / AgentCore headers in agent_extra_headers must NOT overwrite
        the values the proxy sets from trusted server-side config, otherwise a
        caller could spoof the runtime user identity via the x-a2a-{agent}-*
        header rewrite.
        """
        from litellm.a2a_protocol.providers.bedrock_agentcore.transformation import (
            BedrockAgentCoreA2ATransformation,
        )

        litellm_params_with_user = {
            **SAMPLE_LITELLM_PARAMS,
            "runtimeUserId": "legit-user",
        }

        _, headers, _ = BedrockAgentCoreA2ATransformation.get_url_and_signed_request(
            request_id="req-001",
            params=SAMPLE_PARAMS,
            litellm_params=litellm_params_with_user,
            agent_extra_headers={
                # Spoofing attempt — must be dropped.
                "x-amzn-bedrock-agentcore-runtime-user-id": "victim-user",
                "X-Amzn-Bedrock-AgentCore-Runtime-Session-Id": "spoofed-session",
                "Authorization": "Bearer attacker-token",
                "Host": "attacker.example.com",
                "x-amz-content-sha256": "deadbeef",
                # Legitimate per-request header — must pass through.
                "x-mcp-token": "mcp-abc",
            },
        )

        # Legitimate header is preserved.
        assert headers["x-mcp-token"] == "mcp-abc"

        # Reserved headers from agent_extra_headers must not appear at all
        # (case-insensitive) — only the proxy/signer-controlled values may.
        normalized = {k.lower(): v for k, v in headers.items()}

        # Runtime user id is the value set from litellm_params, NOT the spoof.
        assert normalized["x-amzn-bedrock-agentcore-runtime-user-id"] == "legit-user"
        # Session id is the auto-generated one, not the spoofed value.
        assert (
            normalized["x-amzn-bedrock-agentcore-runtime-session-id"]
            != "spoofed-session"
        )
        # Authorization is the JWT bearer set by the signer, not the spoof.
        assert normalized["authorization"] == "Bearer test-jwt-token"
        # Host / x-amz-* must not have been carried over from the client.
        assert normalized.get("host") != "attacker.example.com"
        assert normalized.get("x-amz-content-sha256") != "deadbeef"

    def test_reserved_headers_filtered_before_sigv4_signing(self):
        """
        Reserved headers in agent_extra_headers must be stripped BEFORE the
        SigV4 signer sees them, so the signature does not bind a spoofed
        runtime user identity into a valid SigV4 request.
        """
        from litellm.a2a_protocol.providers.bedrock_agentcore.transformation import (
            BedrockAgentCoreA2ATransformation,
        )

        litellm_params_no_key = {
            "model": SAMPLE_MODEL,
            "custom_llm_provider": "bedrock",
            "aws_access_key_id": "AKIAIOSFODNN7EXAMPLE",
            "aws_secret_access_key": "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
            "aws_region_name": "us-west-2",
            "runtimeUserId": "legit-user",
        }

        captured: dict = {}

        def fake_sign(self, headers, **kwargs):
            captured.update(headers)
            return headers, b'{"jsonrpc":"2.0"}'

        with patch(
            "litellm.llms.bedrock.chat.agentcore.transformation.AmazonAgentCoreConfig._sign_request",
            new=fake_sign,
        ):
            BedrockAgentCoreA2ATransformation.get_url_and_signed_request(
                request_id="req-001",
                params=SAMPLE_PARAMS,
                litellm_params=litellm_params_no_key,
                agent_extra_headers={
                    "x-amzn-bedrock-agentcore-runtime-user-id": "victim-user",
                    "x-amz-date": "20990101T000000Z",
                    "authorization": "Bearer attacker",
                    "x-mcp-token": "mcp-abc",
                },
            )

        normalized = {k.lower(): v for k, v in captured.items()}
        assert normalized["x-amzn-bedrock-agentcore-runtime-user-id"] == "legit-user"
        assert normalized.get("x-amz-date") != "20990101T000000Z"
        assert normalized.get("authorization") != "Bearer attacker"
        # Non-reserved header still makes it into the signed dict.
        assert captured.get("x-mcp-token") == "mcp-abc"

    def test_sigv4_auth_when_no_api_key(self):
        """When no api_key, falls through to SigV4 signing."""
        from litellm.a2a_protocol.providers.bedrock_agentcore.transformation import (
            BedrockAgentCoreA2ATransformation,
        )

        litellm_params_no_key = {
            "model": SAMPLE_MODEL,
            "custom_llm_provider": "bedrock",
            "aws_access_key_id": "AKIAIOSFODNN7EXAMPLE",
            "aws_secret_access_key": "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
            "aws_region_name": "us-west-2",
        }

        # Mock _sign_request to avoid hitting real botocore credential resolution
        fake_sigv4_headers = {
            "Authorization": "AWS4-HMAC-SHA256 Credential=AKIA.../bedrock-agentcore/aws4_request",
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }
        fake_body = b'{"jsonrpc":"2.0"}'

        with patch(
            "litellm.llms.bedrock.chat.agentcore.transformation.AmazonAgentCoreConfig._sign_request",
            return_value=(fake_sigv4_headers, fake_body),
        ):
            _, headers, _ = (
                BedrockAgentCoreA2ATransformation.get_url_and_signed_request(
                    request_id="req-001",
                    params=SAMPLE_PARAMS,
                    litellm_params=litellm_params_no_key,
                )
            )
        # SigV4 produces an Authorization header starting with "AWS4-HMAC-SHA256"
        assert "Authorization" in headers
        assert headers["Authorization"].startswith("AWS4-HMAC-SHA256")


SESSION_HEADER = "X-Amzn-Bedrock-AgentCore-Runtime-Session-Id"
CONTEXT_ID = "conversation-alpha-0001-0000000000000000"
KEY_HASH = "hashed-key-of-caller-one"


def _params_with_context(context_id: object) -> dict:
    return {"message": {**SAMPLE_PARAMS["message"], "contextId": context_id}}


def _scoped(context_id: str, key_hash: str) -> str:
    import hashlib

    return f"{hashlib.sha256(key_hash.encode()).hexdigest()[:16]}-{context_id}"


def _session_header(params: dict, litellm_params: dict) -> str:
    from litellm.a2a_protocol.providers.bedrock_agentcore.transformation import (
        BedrockAgentCoreA2ATransformation,
    )

    _, headers, _ = BedrockAgentCoreA2ATransformation.get_url_and_signed_request(
        request_id="req-001",
        params=params,
        litellm_params=litellm_params,
    )
    return headers[SESSION_HEADER]


@pytest.fixture
def httpx_transport(monkeypatch):
    import litellm

    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setenv("DISABLE_AIOHTTP_TRANSPORT", "True")
    litellm.in_memory_llm_clients_cache.flush_cache()
    yield
    litellm.in_memory_llm_clients_cache.flush_cache()


class TestRequestScopedRuntimeSession:
    """message.contextId selects the AgentCore runtime session, scoped to the calling key."""

    def test_context_id_scoped_to_calling_key(self):
        from litellm.a2a_protocol.litellm_completion_bridge.handler import (
            A2A_USER_API_KEY_HASH_PARAM,
        )

        litellm_params = {**SAMPLE_LITELLM_PARAMS, A2A_USER_API_KEY_HASH_PARAM: KEY_HASH}
        assert _session_header(_params_with_context(CONTEXT_ID), litellm_params) == _scoped(CONTEXT_ID, KEY_HASH)

    def test_context_id_used_verbatim_without_principal(self):
        assert _session_header(_params_with_context(CONTEXT_ID), SAMPLE_LITELLM_PARAMS) == CONTEXT_ID

    def test_same_context_id_reuses_session_and_other_context_isolated(self):
        first = _session_header(_params_with_context(CONTEXT_ID), SAMPLE_LITELLM_PARAMS)
        second = _session_header(_params_with_context(CONTEXT_ID), SAMPLE_LITELLM_PARAMS)
        other = _session_header(
            _params_with_context("conversation-beta-00002-0000000000000000"),
            SAMPLE_LITELLM_PARAMS,
        )
        assert first == second
        assert other != first

    def test_same_context_id_from_different_keys_is_isolated(self):
        from litellm.a2a_protocol.litellm_completion_bridge.handler import (
            A2A_USER_API_KEY_HASH_PARAM,
        )

        params = _params_with_context(CONTEXT_ID)
        caller_one = _session_header(params, {**SAMPLE_LITELLM_PARAMS, A2A_USER_API_KEY_HASH_PARAM: KEY_HASH})
        caller_two = _session_header(
            params, {**SAMPLE_LITELLM_PARAMS, A2A_USER_API_KEY_HASH_PARAM: "hashed-key-of-caller-two"}
        )
        assert caller_one != caller_two
        assert caller_one.endswith(f"-{CONTEXT_ID}")
        assert caller_two.endswith(f"-{CONTEXT_ID}")

    def test_context_id_takes_precedence_over_configured_session(self):
        litellm_params = {**SAMPLE_LITELLM_PARAMS, "runtimeSessionId": "a" * 40}
        assert _session_header(_params_with_context(CONTEXT_ID), litellm_params) == CONTEXT_ID

    def test_configured_session_is_fallback_without_context_id(self):
        litellm_params = {**SAMPLE_LITELLM_PARAMS, "runtimeSessionId": "a" * 40}
        assert _session_header(SAMPLE_PARAMS, litellm_params) == "a" * 40
        assert _session_header(_params_with_context(""), litellm_params) == "a" * 40

    def test_no_context_id_and_no_config_generates_new_session_per_request(self):
        first = _session_header(SAMPLE_PARAMS, SAMPLE_LITELLM_PARAMS)
        second = _session_header(SAMPLE_PARAMS, SAMPLE_LITELLM_PARAMS)
        assert first != second
        assert 33 <= len(first) <= 256

    @pytest.mark.parametrize(
        "context_id",
        [
            "short-context-id",
            "x" * 257,
        ],
    )
    def test_invalid_context_id_rejected_with_clear_error(self, context_id):
        import litellm

        with pytest.raises(litellm.BadRequestError, match="Invalid AgentCore runtime session id") as exc_info:
            _session_header(_params_with_context(context_id), SAMPLE_LITELLM_PARAMS)
        assert exc_info.value.status_code == 400
        assert "33-256" in str(exc_info.value)

    def test_scoped_context_id_shorter_than_33_rejected(self):
        import litellm
        from litellm.a2a_protocol.litellm_completion_bridge.handler import (
            A2A_USER_API_KEY_HASH_PARAM,
        )

        litellm_params = {**SAMPLE_LITELLM_PARAMS, A2A_USER_API_KEY_HASH_PARAM: KEY_HASH}
        with pytest.raises(litellm.BadRequestError, match=_scoped("c" * 15, KEY_HASH)):
            _session_header(_params_with_context("c" * 15), litellm_params)
        assert _session_header(_params_with_context("c" * 16), litellm_params) == _scoped("c" * 16, KEY_HASH)

    def test_invalid_configured_session_rejected(self):
        import litellm

        litellm_params = {**SAMPLE_LITELLM_PARAMS, "runtimeSessionId": "too-short"}
        with pytest.raises(litellm.BadRequestError, match="Invalid AgentCore runtime session id"):
            _session_header(SAMPLE_PARAMS, litellm_params)

    def test_non_string_context_id_falls_back(self):
        litellm_params = {**SAMPLE_LITELLM_PARAMS, "runtimeSessionId": "a" * 40}
        assert _session_header(_params_with_context(12345), litellm_params) == "a" * 40

    def test_spoofed_session_header_does_not_override_context_id(self):
        from litellm.a2a_protocol.providers.bedrock_agentcore.transformation import (
            BedrockAgentCoreA2ATransformation,
        )

        _, headers, _ = BedrockAgentCoreA2ATransformation.get_url_and_signed_request(
            request_id="req-001",
            params=_params_with_context(CONTEXT_ID),
            litellm_params=SAMPLE_LITELLM_PARAMS,
            agent_extra_headers={SESSION_HEADER: "s" * 40},
        )
        assert headers[SESSION_HEADER] == CONTEXT_ID

    @pytest.mark.asyncio
    async def test_context_id_session_header_on_outbound_non_streaming_post(self, httpx_transport):
        from litellm.a2a_protocol.litellm_completion_bridge.handler import (
            A2A_USER_API_KEY_HASH_PARAM,
        )
        from litellm.a2a_protocol.providers.bedrock_agentcore.config import (
            BedrockAgentCoreA2AConfig,
        )

        with respx.mock(assert_all_called=True) as router:
            route = router.post(url__regex=r".*/invocations.*").mock(
                return_value=httpx.Response(200, json={"jsonrpc": "2.0", "id": "req-001", "result": {}})
            )
            await BedrockAgentCoreA2AConfig().handle_non_streaming(
                request_id="req-001",
                params=_params_with_context(CONTEXT_ID),
                litellm_params={**SAMPLE_LITELLM_PARAMS, A2A_USER_API_KEY_HASH_PARAM: KEY_HASH},
            )

        assert route.calls.last.request.headers[SESSION_HEADER] == _scoped(CONTEXT_ID, KEY_HASH)

    @pytest.mark.asyncio
    async def test_context_id_session_header_on_outbound_streaming_post(self, httpx_transport):
        from litellm.a2a_protocol.litellm_completion_bridge.handler import (
            A2A_USER_API_KEY_HASH_PARAM,
        )
        from litellm.a2a_protocol.providers.bedrock_agentcore.config import (
            BedrockAgentCoreA2AConfig,
        )

        with respx.mock(assert_all_called=True) as router:
            route = router.post(url__regex=r".*/invocations.*").mock(
                return_value=httpx.Response(200, json={"jsonrpc": "2.0", "id": "req-001", "result": {}})
            )
            events = [
                event
                async for event in BedrockAgentCoreA2AConfig().handle_streaming(
                    request_id="req-001",
                    params=_params_with_context(CONTEXT_ID),
                    litellm_params={**SAMPLE_LITELLM_PARAMS, A2A_USER_API_KEY_HASH_PARAM: KEY_HASH},
                )
            ]

        assert events == [{"jsonrpc": "2.0", "id": "req-001", "result": {}}]
        assert route.calls.last.request.headers[SESSION_HEADER] == _scoped(CONTEXT_ID, KEY_HASH)


class TestNonStreaming:
    """Test end-to-end non-streaming flow."""

    @pytest.mark.asyncio
    async def test_json_rpc_body_sent_to_agentcore(self):
        """Verify the full JSON-RPC envelope is POSTed, not {"prompt": "..."}."""
        from litellm.a2a_protocol.providers.bedrock_agentcore.config import (
            BedrockAgentCoreA2AConfig,
        )

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "jsonrpc": "2.0",
            "id": "req-001",
            "result": {
                "message": {
                    "role": "agent",
                    "parts": [{"kind": "text", "text": "2"}],
                    "messageId": "resp-001",
                }
            },
        }
        mock_response.raise_for_status = MagicMock()

        with patch(
            "litellm.a2a_protocol.providers.bedrock_agentcore.handler.get_async_httpx_client"
        ) as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            config = BedrockAgentCoreA2AConfig()
            result = await config.handle_non_streaming(
                request_id="req-001",
                params=SAMPLE_PARAMS,
                litellm_params=SAMPLE_LITELLM_PARAMS,
            )

            # Verify the POST was called
            mock_client.post.assert_called_once()
            call_kwargs = mock_client.post.call_args

            # Verify sent body is JSON-RPC, not {"prompt": "..."}
            sent_body = json.loads(call_kwargs.kwargs["data"])
            assert "jsonrpc" in sent_body
            assert "method" in sent_body
            assert sent_body["method"] == "message/send"
            assert sent_body["params"]["message"]["parts"][0]["text"] == "what is 1+1?"

            # Verify response is passed through
            assert result["result"]["message"]["parts"][0]["text"] == "2"

    @pytest.mark.asyncio
    async def test_agent_extra_headers_forwarded_on_outbound_post(self):
        """End-to-end: agent_extra_headers from the bridge land on the HTTP POST."""
        from litellm.a2a_protocol.providers.bedrock_agentcore.config import (
            BedrockAgentCoreA2AConfig,
        )

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "jsonrpc": "2.0",
            "id": "req-001",
            "result": {},
        }
        mock_response.raise_for_status = MagicMock()

        with patch(
            "litellm.a2a_protocol.providers.bedrock_agentcore.handler.get_async_httpx_client"
        ) as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            config = BedrockAgentCoreA2AConfig()
            await config.handle_non_streaming(
                request_id="req-001",
                params=SAMPLE_PARAMS,
                litellm_params=SAMPLE_LITELLM_PARAMS,
                agent_extra_headers={"x-mcp-token": "mcp-abc"},
            )

            sent_headers = mock_client.post.call_args.kwargs["headers"]
            assert sent_headers.get("x-mcp-token") == "mcp-abc"

    @pytest.mark.asyncio
    async def test_a2a_error_response_passthrough(self):
        """JSON-RPC error responses from the agent are returned as-is."""
        from litellm.a2a_protocol.providers.bedrock_agentcore.config import (
            BedrockAgentCoreA2AConfig,
        )

        error_response = {
            "jsonrpc": "2.0",
            "id": "req-001",
            "error": {"code": -32600, "message": "Bad request"},
        }
        mock_response = MagicMock()
        mock_response.json.return_value = error_response
        mock_response.raise_for_status = MagicMock()

        with patch(
            "litellm.a2a_protocol.providers.bedrock_agentcore.handler.get_async_httpx_client"
        ) as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            config = BedrockAgentCoreA2AConfig()
            result = await config.handle_non_streaming(
                request_id="req-001",
                params=SAMPLE_PARAMS,
                litellm_params=SAMPLE_LITELLM_PARAMS,
            )

            assert result["error"]["code"] == -32600
            assert result["error"]["message"] == "Bad request"


class TestConfigManager:
    """Test that config manager routes 'bedrock' correctly."""

    def test_bedrock_returns_config(self):
        from litellm.a2a_protocol.providers.bedrock_agentcore.config import (
            BedrockAgentCoreA2AConfig,
        )
        from litellm.a2a_protocol.providers.config_manager import (
            A2AProviderConfigManager,
        )

        config = A2AProviderConfigManager.get_provider_config(
            "bedrock", model=SAMPLE_MODEL
        )
        assert config is not None
        assert isinstance(config, BedrockAgentCoreA2AConfig)

    def test_bedrock_non_agentcore_returns_none(self):
        """Non-agentcore bedrock models should fall through to completion bridge."""
        from litellm.a2a_protocol.providers.config_manager import (
            A2AProviderConfigManager,
        )

        config = A2AProviderConfigManager.get_provider_config(
            "bedrock", model="bedrock/anthropic.claude-3-sonnet"
        )
        assert config is None

    def test_unknown_provider_returns_none(self):
        from litellm.a2a_protocol.providers.config_manager import (
            A2AProviderConfigManager,
        )

        assert A2AProviderConfigManager.get_provider_config("unknown") is None


class TestHandlerIntegration:
    """Test handler.py changes — litellm_params passed through, api_base not required."""

    @pytest.mark.asyncio
    async def test_provider_config_receives_litellm_params(self):
        """Verify handler passes litellm_params to provider config via kwargs."""
        from litellm.a2a_protocol.litellm_completion_bridge.handler import (
            A2ACompletionBridgeHandler,
        )

        mock_config = AsyncMock()
        mock_config.handle_non_streaming = AsyncMock(
            return_value={"jsonrpc": "2.0", "id": "req-001", "result": {}}
        )

        with patch(
            "litellm.a2a_protocol.litellm_completion_bridge.handler.A2AProviderConfigManager.get_provider_config",
            return_value=mock_config,
        ):
            await A2ACompletionBridgeHandler.handle_non_streaming(
                request_id="req-001",
                params=SAMPLE_PARAMS,
                litellm_params=SAMPLE_LITELLM_PARAMS,
                api_base=None,
            )

            mock_config.handle_non_streaming.assert_called_once_with(
                request_id="req-001",
                params=SAMPLE_PARAMS,
                api_base=None,
                litellm_params=SAMPLE_LITELLM_PARAMS,
                agent_extra_headers=None,
            )

    @pytest.mark.asyncio
    async def test_api_base_none_allowed_with_provider_config(self):
        """api_base=None no longer raises when a provider config is registered."""
        from litellm.a2a_protocol.litellm_completion_bridge.handler import (
            A2ACompletionBridgeHandler,
        )

        mock_config = AsyncMock()
        mock_config.handle_non_streaming = AsyncMock(
            return_value={"jsonrpc": "2.0", "id": "req-001", "result": {}}
        )

        with patch(
            "litellm.a2a_protocol.litellm_completion_bridge.handler.A2AProviderConfigManager.get_provider_config",
            return_value=mock_config,
        ):
            # Should NOT raise ValueError
            result = await A2ACompletionBridgeHandler.handle_non_streaming(
                request_id="req-001",
                params=SAMPLE_PARAMS,
                litellm_params=SAMPLE_LITELLM_PARAMS,
                api_base=None,
            )
            assert result is not None
