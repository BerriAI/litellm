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

import pytest
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
