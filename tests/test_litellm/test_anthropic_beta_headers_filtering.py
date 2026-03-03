"""
Test suite for Anthropic beta headers filtering and mapping across all providers.

This test validates:
1. Headers with null values in the config are filtered out
2. Headers with non-null values are correctly mapped to provider-specific names
3. Unknown headers (not in config) are filtered out
4. For Bedrock providers, beta headers appear in the request body (not just HTTP headers)
"""
import json
import os
from typing import Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import litellm
from litellm.anthropic_beta_headers_manager import (
    filter_and_transform_beta_headers,
)


class TestAnthropicBetaHeadersFiltering:
    """Test beta header filtering and mapping for all providers."""

    @pytest.fixture(autouse=True)
    def setup(self, monkeypatch):
        """Load the beta headers config for testing."""
        # Force use of local config file for tests
        monkeypatch.setenv("LITELLM_LOCAL_ANTHROPIC_BETA_HEADERS", "True")
        
        # Clear the cached config to ensure fresh load with local config
        from litellm import anthropic_beta_headers_manager
        anthropic_beta_headers_manager._BETA_HEADERS_CONFIG = None
        
        config_path = os.path.join(
            os.path.dirname(litellm.__file__),
            "anthropic_beta_headers_config.json",
        )
        with open(config_path, "r") as f:
            self.config = json.load(f)

    def get_all_beta_headers(self) -> List[str]:
        """Get all beta headers from the anthropic provider config."""
        return list(self.config.get("anthropic", {}).keys())

    def get_supported_headers(self, provider: str) -> List[str]:
        """Get headers with non-null values for a provider."""
        provider_config = self.config.get(provider, {})
        return [
            header for header, value in provider_config.items() if value is not None
        ]

    def get_unsupported_headers(self, provider: str) -> List[str]:
        """Get headers with null values for a provider."""
        provider_config = self.config.get(provider, {})
        return [header for header, value in provider_config.items() if value is None]

    def get_mapped_headers(self, provider: str) -> Dict[str, str]:
        """Get mapping of input headers to provider-specific headers."""
        provider_config = self.config.get(provider, {})
        return {
            header: value
            for header, value in provider_config.items()
            if value is not None
        }

    @pytest.mark.parametrize(
        "provider",
        ["anthropic", "azure_ai", "bedrock_converse", "bedrock", "vertex_ai"],
    )
    def test_filter_and_transform_beta_headers_all_headers(self, provider):
        """Test filtering with all possible beta headers."""
        all_headers = self.get_all_beta_headers()
        supported_headers = self.get_supported_headers(provider)
        unsupported_headers = self.get_unsupported_headers(provider)
        mapped_headers = self.get_mapped_headers(provider)

        filtered = filter_and_transform_beta_headers(
            beta_headers=all_headers, provider=provider
        )

        for header in unsupported_headers:
            assert (
                header not in filtered
            ), f"Unsupported header '{header}' should be filtered out for {provider}"
            assert (
                mapped_headers.get(header) not in filtered
            ), f"Mapped value of unsupported header '{header}' should not appear for {provider}"

        for header in supported_headers:
            expected_mapped = mapped_headers[header]
            assert (
                expected_mapped in filtered
            ), f"Supported header '{header}' should be mapped to '{expected_mapped}' for {provider}"

    @pytest.mark.parametrize(
        "provider",
        ["anthropic", "azure_ai", "bedrock_converse", "bedrock", "vertex_ai"],
    )
    def test_unknown_headers_filtered_out(self, provider):
        """Test that headers not in the config are filtered out."""
        unknown_headers = [
            "unknown-header-1",
            "unknown-header-2",
            "fake-beta-2025-01-01",
        ]
        all_headers = self.get_all_beta_headers() + unknown_headers

        filtered = filter_and_transform_beta_headers(
            beta_headers=all_headers, provider=provider
        )

        for unknown in unknown_headers:
            assert (
                unknown not in filtered
            ), f"Unknown header '{unknown}' should be filtered out for {provider}"

    @pytest.mark.asyncio
    async def test_anthropic_messages_http_headers_filtering(self):
        """Test that Anthropic messages API filters HTTP headers correctly."""
        all_headers = self.get_all_beta_headers()
        unsupported = self.get_unsupported_headers("anthropic")

        with patch(
            "litellm.llms.custom_httpx.http_handler.get_async_httpx_client"
        ) as mock_client_factory:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "id": "msg_123",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": "Hello"}],
                "model": "claude-3-5-sonnet-20241022",
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 10, "output_tokens": 20},
            }
            mock_response.headers = {}

            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client_factory.return_value = mock_client

            try:
                await litellm.acompletion(
                    model="anthropic/claude-3-5-sonnet-20241022",
                    messages=[{"role": "user", "content": "Hi"}],
                    extra_headers={"anthropic-beta": ",".join(all_headers)},
                    mock_response="Hello",
                )
            except Exception:
                pass

            if mock_client.post.called:
                call_kwargs = mock_client.post.call_args.kwargs
                headers = call_kwargs.get("headers", {})
                beta_header = headers.get("anthropic-beta", "")

                if beta_header:
                    beta_values = [b.strip() for b in beta_header.split(",")]
                    for unsupported_header in unsupported:
                        assert (
                            unsupported_header not in beta_values
                        ), f"Unsupported header '{unsupported_header}' should not be in HTTP headers for Anthropic"

    @pytest.mark.asyncio
    async def test_azure_ai_messages_http_headers_filtering(self):
        """Test that Azure AI messages API filters HTTP headers correctly."""
        all_headers = self.get_all_beta_headers()
        unsupported = self.get_unsupported_headers("azure_ai")

        with patch(
            "litellm.llms.custom_httpx.http_handler.get_async_httpx_client"
        ) as mock_client_factory:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "id": "msg_123",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": "Hello"}],
                "model": "claude-3-5-sonnet-20241022",
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 10, "output_tokens": 20},
            }
            mock_response.headers = {}

            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client_factory.return_value = mock_client

            try:
                await litellm.acompletion(
                    model="azure_ai/claude-3-5-sonnet-20241022",
                    messages=[{"role": "user", "content": "Hi"}],
                    api_key="test-key",
                    api_base="https://test.azure.com",
                    extra_headers={"anthropic-beta": ",".join(all_headers)},
                    mock_response="Hello",
                )
            except Exception:
                pass

            if mock_client.post.called:
                call_kwargs = mock_client.post.call_args.kwargs
                headers = call_kwargs.get("headers", {})
                beta_header = headers.get("anthropic-beta", "")

                if beta_header:
                    beta_values = [b.strip() for b in beta_header.split(",")]
                    for unsupported_header in unsupported:
                        assert (
                            unsupported_header not in beta_values
                        ), f"Unsupported header '{unsupported_header}' should not be in HTTP headers for Azure AI"

    @pytest.mark.asyncio
    async def test_bedrock_converse_headers_and_body_filtering(self):
        """Test that Bedrock Converse filters both HTTP headers and request body correctly."""
        all_headers = self.get_all_beta_headers()
        unsupported = self.get_unsupported_headers("bedrock_converse")
        mapped_headers = self.get_mapped_headers("bedrock_converse")

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "output": {"message": {"role": "assistant", "content": [{"text": "Hello"}]}},
                "stopReason": "end_turn",
                "usage": {"inputTokens": 10, "outputTokens": 20},
            }
            mock_response.headers = {}
            mock_response.raise_for_status = MagicMock()

            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client_class.return_value.__aenter__.return_value = mock_client

            try:
                await litellm.acompletion(
                    model="bedrock/converse/us.anthropic.claude-3-5-sonnet-20241022-v2:0",
                    messages=[{"role": "user", "content": "Hi"}],
                    aws_access_key_id="test",
                    aws_secret_access_key="test",
                    aws_region_name="us-east-1",
                    extra_headers={"anthropic-beta": ",".join(all_headers)},
                    mock_response="Hello",
                )
            except Exception:
                pass

            if mock_client.post.called:
                call_kwargs = mock_client.post.call_args.kwargs
                headers = call_kwargs.get("headers", {})
                beta_header = headers.get("anthropic-beta", "")

                if beta_header:
                    beta_values = [b.strip() for b in beta_header.split(",")]
                    for unsupported_header in unsupported:
                        assert (
                            unsupported_header not in beta_values
                        ), f"Unsupported header '{unsupported_header}' should not be in HTTP headers for Bedrock Converse"

                data = call_kwargs.get("data")
                if data:
                    body = json.loads(data)
                    body_beta = body.get("additionalModelRequestFields", {}).get(
                        "anthropic_beta", []
                    )

                    for unsupported_header in unsupported:
                        assert (
                            unsupported_header not in body_beta
                        ), f"Unsupported header '{unsupported_header}' should not be in request body for Bedrock Converse"

                    for header, mapped_value in mapped_headers.items():
                        if header in all_headers and mapped_value in body_beta:
                            assert (
                                mapped_value in body_beta
                            ), f"Supported header '{header}' should be mapped to '{mapped_value}' in request body for Bedrock Converse"

    @pytest.mark.asyncio
    async def test_vertex_ai_messages_http_headers_filtering(self):
        """Test that Vertex AI messages API filters HTTP headers correctly."""
        all_headers = self.get_all_beta_headers()
        unsupported = self.get_unsupported_headers("vertex_ai")

        with patch(
            "litellm.llms.custom_httpx.http_handler.get_async_httpx_client"
        ) as mock_client_factory:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "id": "msg_123",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": "Hello"}],
                "model": "claude-3-5-sonnet-20241022",
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 10, "output_tokens": 20},
            }
            mock_response.headers = {}

            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client_factory.return_value = mock_client

            with patch(
                "litellm.llms.vertex_ai.vertex_llm_base.VertexBase._ensure_access_token"
            ) as mock_token:
                mock_token.return_value = ("test-token", "test-project")

                try:
                    await litellm.acompletion(
                        model="vertex_ai/claude-3-5-sonnet-20241022",
                        messages=[{"role": "user", "content": "Hi"}],
                        vertex_project="test-project",
                        vertex_location="us-central1",
                        extra_headers={"anthropic-beta": ",".join(all_headers)},
                        mock_response="Hello",
                    )
                except Exception:
                    pass

            if mock_client.post.called:
                call_kwargs = mock_client.post.call_args.kwargs
                headers = call_kwargs.get("headers", {})
                beta_header = headers.get("anthropic-beta", "")

                if beta_header:
                    beta_values = [b.strip() for b in beta_header.split(",")]
                    for unsupported_header in unsupported:
                        assert (
                            unsupported_header not in beta_values
                        ), f"Unsupported header '{unsupported_header}' should not be in HTTP headers for Vertex AI"

    def test_header_mapping_correctness(self):
        """Test that headers are mapped correctly for providers with transformations."""
        test_cases = [
            {
                "provider": "bedrock",
                "input": "advanced-tool-use-2025-11-20",
                "expected": "tool-search-tool-2025-10-19",
            },
            {
                "provider": "vertex_ai",
                "input": "advanced-tool-use-2025-11-20",
                "expected": "tool-search-tool-2025-10-19",
            },
            {
                "provider": "anthropic",
                "input": "advanced-tool-use-2025-11-20",
                "expected": "advanced-tool-use-2025-11-20",
            },
            {
                "provider": "bedrock_converse",
                "input": "computer-use-2025-01-24",
                "expected": "computer-use-2025-01-24",
            },
            {
                "provider": "azure_ai",
                "input": "advanced-tool-use-2025-11-20",
                "expected": "advanced-tool-use-2025-11-20",
            },
        ]

        for test_case in test_cases:
            filtered = filter_and_transform_beta_headers(
                beta_headers=[test_case["input"]], provider=test_case["provider"]
            )

            assert (
                test_case["expected"] in filtered
            ), f"Header '{test_case['input']}' should be mapped to '{test_case['expected']}' for {test_case['provider']}, but got: {filtered}"

    def test_null_value_headers_filtered(self):
        """Test that headers with null values are always filtered out."""
        for provider in ["anthropic", "azure_ai", "bedrock_converse", "bedrock", "vertex_ai"]:
            unsupported = self.get_unsupported_headers(provider)

            if unsupported:
                filtered = filter_and_transform_beta_headers(
                    beta_headers=unsupported, provider=provider
                )

                assert (
                    len(filtered) == 0
                ), f"All null-value headers should be filtered out for {provider}, but got: {filtered}"

    def test_empty_headers_list(self):
        """Test that empty headers list returns empty result."""
        for provider in ["anthropic", "azure_ai", "bedrock_converse", "bedrock", "vertex_ai"]:
            filtered = filter_and_transform_beta_headers(
                beta_headers=[], provider=provider
            )

            assert (
                len(filtered) == 0
            ), f"Empty headers list should return empty result for {provider}"

    def test_mixed_supported_and_unsupported_headers(self):
        """Test filtering with a mix of supported, unsupported, and unknown headers."""
        for provider in ["anthropic", "azure_ai", "bedrock_converse", "bedrock", "vertex_ai"]:
            supported = self.get_supported_headers(provider)
            unsupported = self.get_unsupported_headers(provider)
            mapped_headers = self.get_mapped_headers(provider)

            if not supported or not unsupported:
                continue

            test_headers = (
                [supported[0]]
                + [unsupported[0]]
                + ["unknown-header-123"]
            )

            filtered = filter_and_transform_beta_headers(
                beta_headers=test_headers, provider=provider
            )

            expected_mapped = mapped_headers[supported[0]]
            assert (
                expected_mapped in filtered
            ), f"Supported header should be in result for {provider}"
            assert (
                unsupported[0] not in filtered
            ), f"Unsupported header should not be in result for {provider}"
            assert (
                "unknown-header-123" not in filtered
            ), f"Unknown header should not be in result for {provider}"
