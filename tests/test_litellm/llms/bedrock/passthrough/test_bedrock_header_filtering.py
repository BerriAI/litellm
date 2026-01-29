"""
Test header filtering for Bedrock passthrough endpoints.

Tests that unsupported anthropic-beta headers are filtered out before forwarding
requests to AWS Bedrock to prevent "invalid beta flag" errors.

Uses whitelist-based filtering consistent with the Invoke API approach in PR #19877.
"""

import pytest
from unittest.mock import Mock, patch
from litellm.llms.bedrock.passthrough.transformation import BedrockPassthroughConfig


class TestBedrockHeaderFiltering:
    """Test suite for anthropic-beta header filtering in Bedrock passthrough."""

    def test_unsupported_beta_header_filtered(self):
        """Test that unsupported anthropic-beta headers are filtered out."""
        config = BedrockPassthroughConfig()

        headers = {
            "content-type": "application/json",
            "anthropic-beta": "oauth-2025-04-20",  # Unsupported - should be filtered
            "x-api-key": "test-key",
            "user-agent": "claude-code/1.0",
        }

        litellm_params = {
            "aws_region_name": "us-east-1",
            "aws_access_key_id": "test-access-key",
            "aws_secret_access_key": "test-secret-key",
        }

        with patch.object(config, "_sign_request") as mock_sign:
            mock_sign.return_value = ({}, None)

            config.sign_request(
                headers=headers,
                litellm_params=litellm_params,
                request_data={"messages": []},
                api_base="https://bedrock-runtime.us-east-1.amazonaws.com",
                model="anthropic.claude-v2",
            )

            # Verify _sign_request was called
            assert mock_sign.called

            # Get the headers that were passed to _sign_request
            call_args = mock_sign.call_args
            signed_headers = call_args[1]["headers"]

            # anthropic-beta should be filtered out (unsupported value)
            assert "anthropic-beta" not in signed_headers

            # Other headers should remain
            assert "content-type" in signed_headers
            assert "x-api-key" in signed_headers
            assert "user-agent" in signed_headers

    def test_supported_beta_header_preserved(self):
        """Test that supported anthropic-beta headers are preserved."""
        config = BedrockPassthroughConfig()

        # Test each supported beta flag
        supported_betas = [
            "computer-use-2024-10-22",
            "computer-use-2025-01-24",
            "token-efficient-tools-2025-02-19",
            "interleaved-thinking-2025-05-14",
            "output-128k-2025-02-19",
            "dev-full-thinking-2025-05-14",
            "context-1m-2025-08-07",
            "context-management-2025-06-27",
            "effort-2025-11-24",
            "tool-search-tool-2025-10-19",
            "tool-examples-2025-10-29",
        ]

        for beta_value in supported_betas:
            headers = {
                "anthropic-beta": beta_value,
                "content-type": "application/json",
            }

            litellm_params = {
                "aws_region_name": "us-east-1",
                "aws_access_key_id": "test-access-key",
                "aws_secret_access_key": "test-secret-key",
            }

            with patch.object(config, "_sign_request") as mock_sign:
                mock_sign.return_value = ({}, None)

                config.sign_request(
                    headers=headers,
                    litellm_params=litellm_params,
                    request_data={},
                    api_base="https://bedrock-runtime.us-east-1.amazonaws.com",
                    model="anthropic.claude-v2",
                )

                call_args = mock_sign.call_args
                signed_headers = call_args[1]["headers"]

                # Supported beta should be preserved
                assert "anthropic-beta" in signed_headers
                assert signed_headers["anthropic-beta"] == beta_value

    def test_mixed_beta_values_filtered(self):
        """Test filtering with mix of supported and unsupported beta values."""
        config = BedrockPassthroughConfig()

        # Mix of supported and unsupported betas (comma-separated)
        headers = {
            "anthropic-beta": "computer-use-2024-10-22,mcp-servers-2025-12-04,oauth-2025-04-20,token-efficient-tools-2025-02-19",
            "content-type": "application/json",
        }

        litellm_params = {
            "aws_region_name": "us-east-1",
            "aws_access_key_id": "test-access-key",
            "aws_secret_access_key": "test-secret-key",
        }

        with patch.object(config, "_sign_request") as mock_sign:
            mock_sign.return_value = ({}, None)

            config.sign_request(
                headers=headers,
                litellm_params=litellm_params,
                request_data={},
                api_base="https://bedrock-runtime.us-east-1.amazonaws.com",
                model="anthropic.claude-v2",
            )

            call_args = mock_sign.call_args
            signed_headers = call_args[1]["headers"]

            # anthropic-beta should be present with only supported values
            assert "anthropic-beta" in signed_headers
            beta_values = signed_headers["anthropic-beta"].split(",")

            # Supported betas should be kept
            assert "computer-use-2024-10-22" in beta_values
            assert "token-efficient-tools-2025-02-19" in beta_values

            # Unsupported betas should be filtered
            assert "mcp-servers-2025-12-04" not in beta_values
            assert "oauth-2025-04-20" not in beta_values

    def test_mcp_servers_beta_filtered(self):
        """
        Test that mcp-servers beta header is filtered out.

        This is the main issue from Claude Code - when MCP servers are configured,
        Claude Code sends mcp-servers-2025-12-04 which Bedrock doesn't support.

        Fixes: https://github.com/BerriAI/litellm/issues/16726
        """
        config = BedrockPassthroughConfig()

        headers = {
            "anthropic-beta": "mcp-servers-2025-12-04",
            "content-type": "application/json",
        }

        litellm_params = {
            "aws_region_name": "us-east-1",
            "aws_access_key_id": "test-access-key",
            "aws_secret_access_key": "test-secret-key",
        }

        with patch.object(config, "_sign_request") as mock_sign:
            mock_sign.return_value = ({}, None)

            config.sign_request(
                headers=headers,
                litellm_params=litellm_params,
                request_data={},
                api_base="https://bedrock-runtime.us-east-1.amazonaws.com",
                model="anthropic.claude-v2",
            )

            call_args = mock_sign.call_args
            signed_headers = call_args[1]["headers"]

            # mcp-servers beta should be filtered out entirely
            assert "anthropic-beta" not in signed_headers

    def test_case_insensitive_header_matching(self):
        """Test that header matching is case-insensitive."""
        config = BedrockPassthroughConfig()

        test_cases = [
            "anthropic-beta",
            "Anthropic-Beta",
            "ANTHROPIC-BETA",
            "Anthropic-beta",
            "AnThRoPiC-bEtA",
        ]

        for header_name in test_cases:
            headers = {
                header_name: "oauth-2025-04-20",  # Unsupported
                "content-type": "application/json",
            }

            litellm_params = {
                "aws_region_name": "us-east-1",
                "aws_access_key_id": "test-access-key",
                "aws_secret_access_key": "test-secret-key",
            }

            with patch.object(config, "_sign_request") as mock_sign:
                mock_sign.return_value = ({}, None)

                config.sign_request(
                    headers=headers,
                    litellm_params=litellm_params,
                    request_data={},
                    api_base="https://bedrock-runtime.us-east-1.amazonaws.com",
                    model="anthropic.claude-v2",
                )

                call_args = mock_sign.call_args
                signed_headers = call_args[1]["headers"]

                # Verify header was processed (unsupported value filtered)
                assert not any(
                    k.lower() == "anthropic-beta" for k in signed_headers.keys()
                )

    def test_other_headers_preserved(self):
        """Test that non-beta headers are preserved correctly."""
        config = BedrockPassthroughConfig()

        headers = {
            "content-type": "application/json",
            "anthropic-beta": "oauth-2025-04-20",  # Should be filtered (unsupported)
            "anthropic-version": "2023-06-01",  # Should be preserved
            "x-api-key": "test-key",  # Should be preserved
            "user-agent": "claude-code/2.0",  # Should be preserved
            "authorization": "Bearer token",  # Should be preserved
        }

        litellm_params = {
            "aws_region_name": "us-east-1",
            "aws_access_key_id": "test-access-key",
            "aws_secret_access_key": "test-secret-key",
        }

        with patch.object(config, "_sign_request") as mock_sign:
            mock_sign.return_value = ({}, None)

            config.sign_request(
                headers=headers,
                litellm_params=litellm_params,
                request_data={},
                api_base="https://bedrock-runtime.us-east-1.amazonaws.com",
                model="anthropic.claude-v2",
            )

            call_args = mock_sign.call_args
            signed_headers = call_args[1]["headers"]

            # Verify anthropic-beta is filtered (unsupported value)
            assert "anthropic-beta" not in signed_headers

            # Verify all other headers are preserved
            assert signed_headers["content-type"] == "application/json"
            assert signed_headers["anthropic-version"] == "2023-06-01"
            assert signed_headers["x-api-key"] == "test-key"
            assert signed_headers["user-agent"] == "claude-code/2.0"
            assert signed_headers["authorization"] == "Bearer token"

    def test_empty_headers(self):
        """Test that empty headers dict doesn't cause errors."""
        config = BedrockPassthroughConfig()

        headers = {}

        litellm_params = {
            "aws_region_name": "us-east-1",
            "aws_access_key_id": "test-access-key",
            "aws_secret_access_key": "test-secret-key",
        }

        with patch.object(config, "_sign_request") as mock_sign:
            mock_sign.return_value = ({}, None)

            # Should not raise any errors
            config.sign_request(
                headers=headers,
                litellm_params=litellm_params,
                request_data={},
                api_base="https://bedrock-runtime.us-east-1.amazonaws.com",
                model="anthropic.claude-v2",
            )

            call_args = mock_sign.call_args
            signed_headers = call_args[1]["headers"]

            # Should be an empty dict
            assert signed_headers == {}

    def test_only_unsupported_beta_results_in_no_header(self):
        """Test that only unsupported beta values results in header being omitted."""
        config = BedrockPassthroughConfig()

        headers = {"anthropic-beta": "oauth-2025-04-20,mcp-servers-2025-12-04"}

        litellm_params = {
            "aws_region_name": "us-east-1",
            "aws_access_key_id": "test-access-key",
            "aws_secret_access_key": "test-secret-key",
        }

        with patch.object(config, "_sign_request") as mock_sign:
            mock_sign.return_value = ({}, None)

            config.sign_request(
                headers=headers,
                litellm_params=litellm_params,
                request_data={},
                api_base="https://bedrock-runtime.us-east-1.amazonaws.com",
                model="anthropic.claude-v2",
            )

            call_args = mock_sign.call_args
            signed_headers = call_args[1]["headers"]

            # Should result in no anthropic-beta header
            assert "anthropic-beta" not in signed_headers


class TestBedrockSupportedBetasConstant:
    """Test the BEDROCK_SUPPORTED_BETAS constant is properly defined."""

    def test_supported_betas_is_set(self):
        """Verify BEDROCK_SUPPORTED_BETAS is a set."""
        assert isinstance(BedrockPassthroughConfig.BEDROCK_SUPPORTED_BETAS, set)

    def test_supported_betas_contains_expected_values(self):
        """Verify all expected supported betas are in the set."""
        expected_betas = {
            "computer-use-2024-10-22",
            "computer-use-2025-01-24",
            "token-efficient-tools-2025-02-19",
            "interleaved-thinking-2025-05-14",
            "output-128k-2025-02-19",
            "dev-full-thinking-2025-05-14",
            "context-1m-2025-08-07",
            "context-management-2025-06-27",
            "effort-2025-11-24",
            "tool-search-tool-2025-10-19",
            "tool-examples-2025-10-29",
        }

        assert BedrockPassthroughConfig.BEDROCK_SUPPORTED_BETAS == expected_betas
