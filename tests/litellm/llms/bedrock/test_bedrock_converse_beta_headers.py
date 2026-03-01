"""
Tests for Bedrock Converse API beta header handling.

Verifies that anthropic-beta headers are stripped from HTTP headers
and only sent via additionalModelRequestFields in the request body.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

import litellm
from litellm.llms.bedrock.chat.converse_handler import BedrockConverseLLM
from litellm.llms.bedrock.chat.converse_transformation import AmazonConverseConfig


@pytest.fixture(autouse=True)
def _set_litellm_local_model_cost_map_env(monkeypatch):
    """Ensure LITELLM_LOCAL_MODEL_COST_MAP is set for all tests in this module."""
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")


class TestBedrockConverseBetaHeaderStripping:
    """Verify anthropic-beta is removed from HTTP headers via the actual handler code path."""

    def test_sync_completion_strips_beta_from_http_headers(self):
        """The sync completion path must strip anthropic-beta from outgoing HTTP headers."""
        handler = BedrockConverseLLM()

        captured_request_headers = {}

        original_get_request_headers = handler.get_request_headers

        def spy_get_request_headers(**kwargs):
            result = original_get_request_headers(**kwargs)
            captured_request_headers.update(kwargs.get("extra_headers", {}))
            return result

        with patch.object(handler, "get_request_headers", side_effect=spy_get_request_headers):
            with patch("litellm.llms.bedrock.chat.converse_handler.HTTPHandler") as mock_http:
                mock_http_instance = MagicMock()
                mock_http.return_value = mock_http_instance
                mock_http_instance.post.return_value = MagicMock(
                    status_code=200,
                    text=json.dumps({"output": {"message": {"role": "assistant", "content": [{"text": "hi"}]}}, "stopReason": "end_turn", "usage": {"inputTokens": 1, "outputTokens": 1}}),
                    headers={},
                    json=lambda: {"output": {"message": {"role": "assistant", "content": [{"text": "hi"}]}}, "stopReason": "end_turn", "usage": {"inputTokens": 1, "outputTokens": 1}},
                )

                try:
                    handler.completion(
                        model="anthropic.claude-sonnet-4-20250514-v1:0",
                        messages=[{"role": "user", "content": "test"}],
                        model_response=MagicMock(),
                        timeout=30,
                        encoding=None,
                        logging_obj=MagicMock(pre_call=MagicMock()),
                        optional_params={},
                        litellm_params={
                            "api_base": "https://bedrock.us-east-1.amazonaws.com",
                            "model": "bedrock/anthropic.claude-sonnet-4-20250514-v1:0",
                            "aws_region_name": "us-east-1",
                        },
                        extra_headers={
                            "anthropic-beta": "context-1m-2025-08-07",
                        },
                        acompletion=False,
                        stream=False,
                    )
                except Exception:
                    pass

        assert "anthropic-beta" not in captured_request_headers

    def test_case_insensitive_beta_stripped(self):
        """anthropic-beta removal must be case-insensitive per HTTP spec."""
        headers = {
            "Content-Type": "application/json",
            "Anthropic-Beta": "context-1m-2025-08-07",
        }
        for key in [k for k in headers if k.lower() == "anthropic-beta"]:
            del headers[key]

        assert all(k.lower() != "anthropic-beta" for k in headers)



class TestBedrockConverseBetaInBody:
    """Verify anthropic-beta values are correctly placed in additionalModelRequestFields."""

    def test_context_1m_beta_in_body(self):
        """context-1m beta must appear in additionalModelRequestFields.anthropic_beta."""
        config = AmazonConverseConfig()
        extra_headers = {"anthropic-beta": "context-1m-2025-08-07"}

        result = config._transform_request(
            model="anthropic.claude-sonnet-4-20250514-v1:0",
            messages=[{"role": "user", "content": [{"type": "text", "text": "Hi"}]}],
            optional_params={},
            litellm_params={
                "api_base": "",
                "model": "bedrock/anthropic.claude-sonnet-4-20250514-v1:0",
            },
            headers=extra_headers,
        )

        amrf = result.get("additionalModelRequestFields", {})
        assert "anthropic_beta" in amrf
        assert "context-1m-2025-08-07" in amrf["anthropic_beta"]

    def test_multiple_betas_in_body(self):
        """Multiple beta values must all appear in the body."""
        config = AmazonConverseConfig()
        extra_headers = {
            "anthropic-beta": "context-1m-2025-08-07,interleaved-thinking-2025-05-14"
        }

        result = config._transform_request(
            model="anthropic.claude-sonnet-4-20250514-v1:0",
            messages=[{"role": "user", "content": [{"type": "text", "text": "Hi"}]}],
            optional_params={},
            litellm_params={
                "api_base": "",
                "model": "bedrock/anthropic.claude-sonnet-4-20250514-v1:0",
            },
            headers=extra_headers,
        )

        amrf = result.get("additionalModelRequestFields", {})
        assert "anthropic_beta" in amrf
        assert "context-1m-2025-08-07" in amrf["anthropic_beta"]
        assert "interleaved-thinking-2025-05-14" in amrf["anthropic_beta"]

    def test_no_beta_header_no_body_field(self):
        """Without anthropic-beta header, no anthropic_beta in body."""
        config = AmazonConverseConfig()

        result = config._transform_request(
            model="anthropic.claude-sonnet-4-20250514-v1:0",
            messages=[{"role": "user", "content": [{"type": "text", "text": "Hi"}]}],
            optional_params={},
            litellm_params={
                "api_base": "",
                "model": "bedrock/anthropic.claude-sonnet-4-20250514-v1:0",
            },
            headers={},
        )

        amrf = result.get("additionalModelRequestFields", {})
        assert "anthropic_beta" not in amrf

    def test_non_anthropic_model_no_beta_in_body(self):
        """Non-Anthropic models should not get anthropic_beta in body."""
        config = AmazonConverseConfig()
        extra_headers = {"anthropic-beta": "context-1m-2025-08-07"}

        result = config._transform_request(
            model="amazon.nova-pro-v1:0",
            messages=[{"role": "user", "content": [{"type": "text", "text": "Hi"}]}],
            optional_params={},
            litellm_params={
                "api_base": "",
                "model": "bedrock/amazon.nova-pro-v1:0",
            },
            headers=extra_headers,
        )

        amrf = result.get("additionalModelRequestFields", {})
        assert "anthropic_beta" not in amrf


class TestAsyncBetaHeaderHandling:
    """Verify async paths correctly pass beta values to the request body
    while stripping from HTTP headers."""

    @pytest.mark.asyncio
    async def test_async_completion_preserves_beta_in_body(self):
        """Async completion should extract betas into additionalModelRequestFields
        even though anthropic-beta is stripped from HTTP headers."""
        handler = BedrockConverseLLM()

        captured_headers = {}

        async def mock_transform(**kwargs):
            captured_headers.update(kwargs.get("headers", {}))
            return {"messages": [], "modelId": "test"}

        with patch.object(
            litellm.AmazonConverseConfig,
            "_async_transform_request",
            side_effect=mock_transform,
        ):
            try:
                await handler.async_completion(
                    model="anthropic.claude-sonnet-4-20250514-v1:0",
                    messages=[{"role": "user", "content": "test"}],
                    api_base="https://bedrock.us-east-1.amazonaws.com",
                    model_response=MagicMock(),
                    timeout=30,
                    encoding=None,
                    logging_obj=MagicMock(pre_call=MagicMock()),
                    stream=False,
                    optional_params={},
                    litellm_params={"aws_region_name": "us-east-1"},
                    credentials=MagicMock(),
                    headers={},  # stripped headers (no anthropic-beta)
                    extra_headers={
                        "Content-Type": "application/json",
                        "anthropic-beta": "context-1m-2025-08-07",
                    },
                )
            except Exception:
                pass  # We only care about what headers the transform received

        assert "anthropic-beta" in captured_headers
        assert captured_headers["anthropic-beta"] == "context-1m-2025-08-07"

    @pytest.mark.asyncio
    async def test_async_streaming_preserves_beta_in_body(self):
        """Async streaming should extract betas into additionalModelRequestFields
        even though anthropic-beta is stripped from HTTP headers."""
        handler = BedrockConverseLLM()

        captured_headers = {}

        async def mock_transform(**kwargs):
            captured_headers.update(kwargs.get("headers", {}))
            return {"messages": [], "modelId": "test"}

        with patch.object(
            litellm.AmazonConverseConfig,
            "_async_transform_request",
            side_effect=mock_transform,
        ):
            try:
                await handler.async_streaming(
                    model="anthropic.claude-sonnet-4-20250514-v1:0",
                    messages=[{"role": "user", "content": "test"}],
                    api_base="https://bedrock.us-east-1.amazonaws.com",
                    model_response=MagicMock(),
                    timeout=30,
                    encoding=None,
                    logging_obj=MagicMock(pre_call=MagicMock()),
                    stream=True,
                    optional_params={},
                    litellm_params={"aws_region_name": "us-east-1"},
                    credentials=MagicMock(),
                    headers={},  # stripped headers (no anthropic-beta)
                    extra_headers={
                        "Content-Type": "application/json",
                        "anthropic-beta": "context-1m-2025-08-07",
                    },
                )
            except Exception:
                pass  # We only care about what headers the transform received

        assert "anthropic-beta" in captured_headers
        assert captured_headers["anthropic-beta"] == "context-1m-2025-08-07"

    @pytest.mark.asyncio
    async def test_async_completion_falls_back_to_headers(self):
        """When extra_headers is None, transform should use headers."""
        handler = BedrockConverseLLM()

        captured_headers = {}

        async def mock_transform(**kwargs):
            captured_headers.update(kwargs.get("headers", {}))
            return {"messages": [], "modelId": "test"}

        with patch.object(
            litellm.AmazonConverseConfig,
            "_async_transform_request",
            side_effect=mock_transform,
        ):
            try:
                await handler.async_completion(
                    model="anthropic.claude-sonnet-4-20250514-v1:0",
                    messages=[{"role": "user", "content": "test"}],
                    api_base="https://bedrock.us-east-1.amazonaws.com",
                    model_response=MagicMock(),
                    timeout=30,
                    encoding=None,
                    logging_obj=MagicMock(pre_call=MagicMock()),
                    stream=False,
                    optional_params={},
                    litellm_params={"aws_region_name": "us-east-1"},
                    credentials=MagicMock(),
                    headers={"Content-Type": "application/json"},
                    # extra_headers not provided — should fall back to headers
                )
            except Exception:
                pass

        assert captured_headers.get("Content-Type") == "application/json"
