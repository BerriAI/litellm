import pytest
from fastapi import HTTPException

from litellm.proxy._experimental.mcp_server.utils import (
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
