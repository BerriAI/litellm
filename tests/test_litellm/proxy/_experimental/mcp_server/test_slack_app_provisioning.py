from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def test_build_slack_mcp_manifest_uses_callback_and_user_scopes():
    from litellm.proxy._experimental.mcp_server.slack_app_provisioning import (
        SLACK_MCP_USER_SCOPES,
        build_slack_mcp_manifest,
    )

    manifest = build_slack_mcp_manifest(
        callback_url="https://gw.acme.com/callback", app_name="Acme Connector"
    )

    assert manifest["display_information"]["name"] == "Acme Connector"
    assert manifest["oauth_config"]["redirect_urls"] == ["https://gw.acme.com/callback"]
    # MCP is a user-token flow: scopes must be declared under "user", not "bot"
    assert manifest["oauth_config"]["scopes"]["user"] == SLACK_MCP_USER_SCOPES
    assert "bot" not in manifest["oauth_config"]["scopes"]
    assert "search:read.public" in manifest["oauth_config"]["scopes"]["user"]


def test_build_slack_mcp_manifest_allows_scope_override():
    from litellm.proxy._experimental.mcp_server.slack_app_provisioning import (
        build_slack_mcp_manifest,
    )

    manifest = build_slack_mcp_manifest(
        callback_url="https://gw.acme.com/callback",
        app_name="Acme",
        user_scopes=["chat:write", "channels:read"],
    )
    assert manifest["oauth_config"]["scopes"]["user"] == ["chat:write", "channels:read"]


def test_build_slack_mcp_manifest_enables_token_rotation():
    from litellm.proxy._experimental.mcp_server.slack_app_provisioning import (
        build_slack_mcp_manifest,
    )

    manifest = build_slack_mcp_manifest(
        callback_url="https://gw.acme.com/callback", app_name="Acme"
    )
    assert manifest["settings"]["token_rotation_enabled"] is True


def test_slack_token_refresh_url_remaps_only_slack_hosts():
    from litellm.proxy._experimental.mcp_server.slack_app_provisioning import (
        SLACK_MCP_TOKEN_REFRESH_URL,
        slack_token_refresh_url,
    )

    # Slack refreshes at oauth.v2.access, not the oauth.v2.user.access exchange URL
    assert (
        slack_token_refresh_url("https://slack.com/api/oauth.v2.user.access")
        == SLACK_MCP_TOKEN_REFRESH_URL
    )
    # every other provider keeps its single token endpoint for refresh
    assert (
        slack_token_refresh_url("https://auth.example.com/oauth/token")
        == "https://auth.example.com/oauth/token"
    )
    assert slack_token_refresh_url(None) is None


@pytest.mark.asyncio
async def test_provision_slack_app_returns_credentials():
    from litellm.proxy._experimental.mcp_server.slack_app_provisioning import (
        provision_slack_app,
    )

    mock_response = MagicMock()
    mock_response.json.return_value = {
        "ok": True,
        "app_id": "A0123",
        "credentials": {
            "client_id": "1601185624273.8899143856786",
            "client_secret": "real-secret",
            "signing_secret": "sign",
        },
    }
    mock_client = MagicMock()
    mock_client.post = AsyncMock(return_value=mock_response)

    with patch(
        "litellm.proxy._experimental.mcp_server.slack_app_provisioning.get_async_httpx_client",
        return_value=mock_client,
    ):
        result = await provision_slack_app(
            app_config_token="xoxe.xoxp-token",
            manifest={"display_information": {"name": "x"}},
        )

    assert result.app_id == "A0123"
    assert result.client_id == "1601185624273.8899143856786"
    assert result.client_secret == "real-secret"

    call = mock_client.post.call_args
    assert call.args[0] == "https://slack.com/api/apps.manifest.create"
    assert call.kwargs["headers"]["Authorization"] == "Bearer xoxe.xoxp-token"
    assert "manifest" in call.kwargs["json"]


@pytest.mark.asyncio
async def test_provision_slack_app_raises_on_slack_error():
    from fastapi import HTTPException

    from litellm.proxy._experimental.mcp_server.slack_app_provisioning import (
        provision_slack_app,
    )

    mock_response = MagicMock()
    mock_response.json.return_value = {"ok": False, "error": "invalid_auth"}
    mock_client = MagicMock()
    mock_client.post = AsyncMock(return_value=mock_response)

    with patch(
        "litellm.proxy._experimental.mcp_server.slack_app_provisioning.get_async_httpx_client",
        return_value=mock_client,
    ):
        with pytest.raises(HTTPException) as exc_info:
            await provision_slack_app(
                app_config_token="bad-token",
                manifest={"display_information": {"name": "x"}},
            )

    assert exc_info.value.status_code == 502
    assert "invalid_auth" in str(exc_info.value.detail)
