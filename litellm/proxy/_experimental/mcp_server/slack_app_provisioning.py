"""Manifest-assisted provisioning for the hosted Slack MCP server.

Slack does not support Dynamic Client Registration, so a gateway needs a
registered Slack app's ``client_id``/``client_secret`` to run the user-token
OAuth flow against ``mcp.slack.com``. This module creates that app for the
operator from a manifest via Slack's ``apps.manifest.create`` API, so they never
hand-build a manifest or copy credentials. The operator still completes the
governance steps Slack requires by hand (enable the MCP toggle, then publish or
make the app internal with admin approval); those cannot be automated.
"""

import json
from dataclasses import dataclass
from typing import Optional, cast
from urllib.parse import urlparse

from fastapi import HTTPException

from litellm.llms.custom_httpx.http_handler import (
    get_async_httpx_client,
    httpxSpecialProvider,
)

SLACK_DEFAULT_API_BASE = "https://slack.com/api"

# Slack rotates user tokens at oauth.v2.access, a different endpoint than the
# oauth.v2.user.access used for the initial code exchange, so a refresh cannot
# reuse the server's token_url the way an RFC 6749 provider does.
SLACK_MCP_TOKEN_REFRESH_URL = "https://slack.com/api/oauth.v2.access"


def slack_token_refresh_url(token_url: Optional[str]) -> Optional[str]:
    """Return the endpoint that refreshes a Slack user token, else ``token_url``.

    Only Slack hosts are remapped; every other provider keeps the single token
    endpoint used for both the authorization_code and refresh_token grants.
    """
    if token_url is None:
        return None
    host = urlparse(token_url).hostname
    if host == "slack.com" or (host is not None and host.endswith(".slack.com")):
        return SLACK_MCP_TOKEN_REFRESH_URL
    return token_url


# User scopes the hosted Slack MCP server requests. Kept in sync with the
# scopes Slack's MCP authorize flow asks for.
SLACK_MCP_USER_SCOPES: list[str] = [
    "search:read.public",
    "search:read.private",
    "search:read.mpim",
    "search:read.im",
    "search:read.files",
    "search:read.users",
    "chat:write",
    "channels:history",
    "groups:history",
    "im:history",
    "mpim:history",
    "channels:read",
    "groups:read",
    "mpim:read",
    "channels:write",
    "groups:write",
    "im:write",
    "mpim:write",
    "canvases:read",
    "canvases:write",
    "users:read",
    "users:read.email",
    "reactions:read",
    "reactions:write",
    "emoji:read",
    "files:read",
]


@dataclass(frozen=True)
class SlackProvisionedApp:
    app_id: str
    client_id: str
    client_secret: str


def _string_field(mapping: dict[str, object], key: str) -> str:
    value = mapping.get(key)
    if isinstance(value, str) and value.strip():
        return value
    raise HTTPException(
        status_code=502,
        detail=f"Slack apps.manifest.create response omitted {key}",
    )


def build_slack_mcp_manifest(
    callback_url: str,
    app_name: str,
    user_scopes: Optional[list[str]] = None,
) -> dict[str, object]:
    """Build a Slack app manifest for the hosted MCP server.

    ``callback_url`` is this gateway's OAuth callback so the registered redirect
    URL always matches the deployment's own domain. ``user_scopes`` overrides the
    default MCP scope set so operators can track Slack's evolving scope surface
    without a code change.
    """
    return {
        "display_information": {
            "name": app_name,
            "description": "Connects this gateway's users to the Slack MCP server",
        },
        "oauth_config": {
            "redirect_urls": [callback_url],
            "scopes": {"user": user_scopes or SLACK_MCP_USER_SCOPES},
        },
        "settings": {
            "org_deploy_enabled": False,
            "socket_mode_enabled": False,
            "token_rotation_enabled": True,
        },
    }


async def provision_slack_app(
    app_config_token: str,
    manifest: dict[str, object],
    slack_api_base: str = SLACK_DEFAULT_API_BASE,
) -> SlackProvisionedApp:
    """Create a Slack app from ``manifest`` via ``apps.manifest.create``.

    ``app_config_token`` is a Slack app-configuration token the operator
    generates once; it is used for this single call and never stored.
    """
    async_client = get_async_httpx_client(llm_provider=httpxSpecialProvider.Oauth2Register)
    response = await async_client.post(
        f"{slack_api_base.rstrip('/')}/apps.manifest.create",
        headers={
            "Authorization": f"Bearer {app_config_token}",
            "Content-Type": "application/json",
        },
        json={"manifest": json.dumps(manifest)},
    )
    if response is None:
        raise HTTPException(
            status_code=502,
            detail="Slack apps.manifest.create returned no response",
        )

    raw: object = response.json()
    if not isinstance(raw, dict):
        raise HTTPException(
            status_code=502,
            detail="Slack apps.manifest.create returned a malformed response",
        )
    payload = cast(dict[str, object], raw)
    if payload.get("ok") is not True:
        error = payload.get("error")
        detail = error if isinstance(error, str) else "unknown_error"
        raise HTTPException(
            status_code=502,
            detail=f"Slack apps.manifest.create failed: {detail}",
        )

    credentials = payload.get("credentials")
    if not isinstance(credentials, dict):
        raise HTTPException(
            status_code=502,
            detail="Slack apps.manifest.create response omitted app credentials",
        )

    return SlackProvisionedApp(
        app_id=_string_field(payload, "app_id"),
        client_id=_string_field(cast(dict[str, object], credentials), "client_id"),
        client_secret=_string_field(cast(dict[str, object], credentials), "client_secret"),
    )
