from unittest.mock import patch

import pytest
from fastapi import HTTPException

from litellm.proxy._experimental.mcp_server.utils import (
    _upstream_credential_headers,
    build_synthetic_mcp_request,
    logging_safe_mcp_headers,
    validate_and_normalize_mcp_server_payload,
    validate_tool_display_names,
)
from litellm.proxy._types import NewMCPServerRequest
from litellm.types.mcp_server.mcp_server_manager import MCPServer


def _server_forwarding(*header_names: str) -> MCPServer:
    return MCPServer(
        server_id="srv-1",
        name="deepwiki",
        transport="http",
        url="https://mcp.example.com/mcp",
        extra_headers=list(header_names),
    )


def _configured_servers(*servers: MCPServer):
    return patch.dict(
        "litellm.proxy._experimental.mcp_server.mcp_server_manager.global_mcp_server_manager.config_mcp_servers",
        {server.server_id: server for server in servers},
        clear=False,
    )


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

    def test_strips_headers_a_server_forwards_upstream(self):
        """mcp_servers.<name>.extra_headers names the headers the proxy relays upstream, so a
        caller supplied value under one of them is an upstream credential no prefix rule can spot.
        Config is written in canonical casing while the wire header arrives lowercased."""
        with _configured_servers(_server_forwarding("X-GitHub-Token", "X-Tenant")):
            safe = logging_safe_mcp_headers({"x-github-token": "ghp_secret", "x-tenant": "acct-1", "x-nuid": "nuid-1"})

        assert safe == {"x-nuid": "nuid-1"}

    def test_strips_caller_asserted_host(self):
        """This mapping reaches the guardrail payload and the list_tools spend row, so a caller
        must not be able to name the deployment there either."""
        safe = logging_safe_mcp_headers({"host": "evil.attacker.example", "x-nuid": "nuid-1"})

        assert safe == {"x-nuid": "nuid-1"}

    def test_keeps_identity_header_a_server_also_forwards(self):
        """get_user_from_headers resolves end user attribution off this same request, so a header
        the deployment reads identity from stays even when a server forwards it upstream."""
        with patch.dict(
            "litellm.proxy.proxy_server.general_settings",
            {"user_header_name": "x-user-email"},
            clear=False,
        ):
            with _configured_servers(_server_forwarding("x-user-email", "x-github-token")):
                safe = logging_safe_mcp_headers({"x-user-email": "alice@corp.example", "x-github-token": "ghp_secret"})

        assert safe == {"x-user-email": "alice@corp.example"}

    @pytest.mark.parametrize(
        "configured",
        [
            [{"header_name": "X-User", "litellm_user_role": "customer"}],
            {"header_name": "X-User", "litellm_user_role": "customer"},
        ],
        ids=["list-of-mappings", "bare-mapping"],
    )
    def test_keeps_identity_header_from_user_header_mappings(self, configured):
        """get_internal_user_header_from_mapping and get_customer_user_header_from_mapping both
        accept a bare mapping as well as a list, and config_settings.md documents the key as a
        dict, so the exemption has to read both shapes."""
        with patch.dict(
            "litellm.proxy.proxy_server.general_settings",
            {"user_header_mappings": configured},
            clear=False,
        ):
            with _configured_servers(_server_forwarding("X-User", "X-GitHub-Token")):
                safe = logging_safe_mcp_headers({"x-user": "alice", "x-github-token": "ghp_secret"})

        assert safe == {"x-user": "alice"}

    def test_keeps_authorization_classification_for_oauth_passthrough(self):
        """clean_headers already strips authorization, and claiming it here would change which
        header authenticated_with_header resolves to on a config that lists it by design."""
        with _configured_servers(_server_forwarding("Authorization", "X-GitHub-Token")):
            assert "authorization" not in _upstream_credential_headers(["authorization", "x-github-token"])
            assert "x-github-token" in _upstream_credential_headers(["authorization", "x-github-token"])

    def test_keeps_headers_when_no_server_forwards_them(self):
        with _configured_servers(_server_forwarding("x-github-token")):
            safe = logging_safe_mcp_headers({"x-other-token": "not-forwarded", "x-nuid": "nuid-1"})

        assert safe == {"x-other-token": "not-forwarded", "x-nuid": "nuid-1"}


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

    def test_drops_caller_host_so_the_logged_url_is_not_client_steerable(self):
        """add_litellm_data_to_request records str(request.url) as proxy_server_request.url, and
        Request.url is built from the host header, so forwarding it hands the caller that value."""
        request = build_synthetic_mcp_request(
            path="/mcp/tools/call",
            raw_headers={"host": "evil.attacker.example", "x-nuid": "nuid-1"},
        )

        assert "evil.attacker.example" not in str(request.url)
        assert "host" not in request.headers
        assert request.headers.get("x-nuid") == "nuid-1"

    def test_drops_headers_a_server_forwards_upstream(self):
        with _configured_servers(_server_forwarding("x-github-token")):
            request = build_synthetic_mcp_request(
                path="/mcp/tools/call",
                raw_headers={"x-github-token": "ghp_secret", "x-nuid": "nuid-1"},
            )

        assert "x-github-token" not in request.headers
        assert request.headers.get("x-nuid") == "nuid-1"

    def test_keeps_identity_header_so_end_user_attribution_survives(self):
        """add_litellm_data_to_request reads user_header_name off this request to fill
        end_user_id, so forwarding that header upstream must not remove it here."""
        with patch.dict(
            "litellm.proxy.proxy_server.general_settings",
            {"user_header_name": "x-user-email"},
            clear=False,
        ):
            with _configured_servers(_server_forwarding("x-user-email")):
                request = build_synthetic_mcp_request(
                    path="/mcp/tools/call",
                    raw_headers={"x-user-email": "alice@corp.example"},
                )

        assert request.headers.get("x-user-email") == "alice@corp.example"
