from unittest.mock import patch

import pytest
from fastapi import HTTPException

from litellm.proxy._experimental.mcp_server.utils import (
    build_synthetic_mcp_request,
    logging_safe_mcp_headers,
    validate_and_normalize_mcp_server_payload,
    validate_tool_display_names,
)
from litellm.proxy._types import NewMCPServerRequest


class TestValidateToolDisplayNames:
    def test_allows_none_and_empty(self):
        validate_tool_display_names(None)
        validate_tool_display_names({})

    @pytest.mark.parametrize(
        "display_name",
        ["browse_repo_docs", "browse-repo-docs", "BrowseRepoDocs123"],
    )
    def test_allows_bedrock_safe_names(self, display_name):
        validate_tool_display_names({"read_wiki_structure": display_name})

    @pytest.mark.parametrize(
        "display_name",
        ["Browse Repo Docs", "browse.repo.docs", "browse/repo", "browse@docs"],
    )
    def test_rejects_names_bedrock_would_reject(self, display_name):
        with pytest.raises(HTTPException) as exc_info:
            validate_tool_display_names({"read_wiki_structure": display_name})
        assert exc_info.value.status_code == 400
        assert display_name in str(exc_info.value.detail)


class TestValidateAndNormalizeMcpServerPayload:
    def test_rejects_invalid_tool_display_name_on_create(self):
        payload = NewMCPServerRequest(
            server_name="deepwiki_mcp",
            tool_name_to_display_name={"read_wiki_structure": "Browse Repo Docs"},
        )
        with pytest.raises(HTTPException) as exc_info:
            validate_and_normalize_mcp_server_payload(payload)
        assert exc_info.value.status_code == 400

    def test_accepts_valid_tool_display_name_on_create(self):
        payload = NewMCPServerRequest(
            server_name="deepwiki_mcp",
            tool_name_to_display_name={"read_wiki_structure": "browse_repo_docs"},
        )
        validate_and_normalize_mcp_server_payload(payload)


class TestLoggingSafeMcpHeaders:
    def test_returns_empty_for_missing_headers(self):
        assert logging_safe_mcp_headers(None) == {}
        assert logging_safe_mcp_headers({}) == {}

    def test_exposes_custom_headers_and_masks_credentials(self):
        safe = logging_safe_mcp_headers(
            {
                "x-nuid": "nuid-1",
                "x-app-id": "app-1",
                "x-litellm-api-key": "sk-proxy",
                "cookie": "session=secret",
            }
        )
        assert safe == {
            "x-nuid": "nuid-1",
            "x-app-id": "app-1",
            "cookie": "***REDACTED***",
        }

    def test_strips_custom_litellm_key_header(self):
        """general_settings.litellm_key_header_name carries the proxy virtual key, so it must
        never reach a callback or a guardrail even though clean_headers cannot know its name."""
        with patch.dict(
            "litellm.proxy.proxy_server.general_settings",
            {"litellm_key_header_name": "x-company-key"},
            clear=False,
        ):
            safe = logging_safe_mcp_headers({"x-company-key": "sk-proxy", "x-nuid": "nuid-1"})

        assert safe == {"x-nuid": "nuid-1"}

    def test_strips_client_controlled_redaction_opt_out(self):
        """litellm-disable-message-redaction is read back out of the logged metadata to turn off
        redaction, so leaving it in place lets any MCP client undo what an admin configured."""
        safe = logging_safe_mcp_headers({"litellm-disable-message-redaction": "true", "x-nuid": "nuid-1"})

        assert safe == {"x-nuid": "nuid-1"}

    def test_strips_upstream_mcp_credentials(self):
        safe = logging_safe_mcp_headers(
            {
                "x-mcp-auth": "Bearer upstream",
                "x-mcp-github-authorization": "Bearer gh_token",
                "x-mcp-zapier-x-api-key": "zapier-key",
                "x-nuid": "nuid-1",
            }
        )

        assert safe == {"x-nuid": "nuid-1"}

    def test_strips_custom_mcp_client_side_auth_header(self):
        with patch.dict(
            "litellm.proxy.proxy_server.general_settings",
            {"mcp_client_side_auth_header_name": "x-upstream-token"},
            clear=False,
        ):
            safe = logging_safe_mcp_headers({"x-upstream-token": "Bearer upstream", "x-nuid": "nuid-1"})

        assert safe == {"x-nuid": "nuid-1"}


class TestBuildSyntheticMcpRequest:
    def test_forwards_client_headers_without_upstream_credentials(self):
        """The synthetic request feeds add_litellm_data_to_request, which derives
        metadata.headers, so upstream MCP credentials must not ride along."""
        request = build_synthetic_mcp_request(
            path="/mcp/tools/call",
            raw_headers={
                "x-nuid": "nuid-1",
                "x-mcp-auth": "Bearer upstream",
                "x-mcp-github-authorization": "Bearer gh_token",
            },
        )

        assert request.headers.get("x-nuid") == "nuid-1"
        assert "x-mcp-auth" not in request.headers
        assert "x-mcp-github-authorization" not in request.headers

    def test_drops_custom_litellm_key_header(self):
        """Callers such as the sampling flow build metadata off this request, so the
        deployment's custom proxy key header must never be forwarded on it."""
        with patch.dict(
            "litellm.proxy.proxy_server.general_settings",
            {"litellm_key_header_name": "x-company-key"},
            clear=False,
        ):
            request = build_synthetic_mcp_request(
                path="/mcp/sampling/createMessage",
                raw_headers={"x-company-key": "sk-proxy-secret", "x-nuid": "nuid-1"},
            )

        assert request.headers.get("x-nuid") == "nuid-1"
        assert "x-company-key" not in request.headers
