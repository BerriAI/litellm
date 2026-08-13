import pytest
from fastapi import HTTPException

from litellm.proxy._experimental.mcp_server.utils import (
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
