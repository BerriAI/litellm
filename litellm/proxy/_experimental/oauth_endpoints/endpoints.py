"""
OAuth endpoints for the proxy to support oauth2 for MCP servers
"""

import time
import uuid
from typing import Dict

import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import RedirectResponse

from litellm.types.mcp import MCPAuth

router = APIRouter()

sessions: Dict[str, Dict] = {}  # state -> { created_at, redirect_to }


@router.get("/oauth/{mcp_server_id}/callback")
async def mcp_server_oauth_callback(
    code: str, state: str, mcp_server_id: str, request: Request
):
    """
    Callback for the MCP server
    """
    from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
        global_mcp_server_manager,
    )
    from litellm.proxy.management_endpoints.ui_sso import SSOAuthenticationHandler

    mcp_server = global_mcp_server_manager.get_mcp_server_by_id(mcp_server_id)

    if mcp_server is None:
        raise HTTPException(status_code=404, detail="MCP server not found")

    if mcp_server.auth_type != MCPAuth.oauth2:
        raise HTTPException(
            status_code=400, detail="MCP server is not an OAuth2 server"
        )

    if mcp_server.oauth_info is None:
        raise HTTPException(status_code=400, detail="MCP server has no OAuth2 info")

    if mcp_server.oauth_info.token_url is None:
        raise HTTPException(
            status_code=400, detail="MCP server has no OAuth2 token URL"
        )

    if state not in sessions:
        raise HTTPException(400, "Invalid state")

    json_data: Dict[str, str] = {
        "code": code,
        "redirect_uri": SSOAuthenticationHandler.get_redirect_url_for_sso(
            request=request, sso_callback_route=f"oauth/{mcp_server_id}/callback"
        ),
        "state": state,
    }

    if mcp_server.oauth_info.client_id is not None:
        json_data["client_id"] = mcp_server.oauth_info.client_id
    if mcp_server.oauth_info.client_secret is not None:
        json_data["client_secret"] = mcp_server.oauth_info.client_secret

    async with httpx.AsyncClient() as client:
        resp = await client.post(
            mcp_server.oauth_info.token_url,
            headers={"Accept": "application/json"},
            json=json_data,
        )
        data = resp.json()

    if "error" in data:
        raise HTTPException(400, data)

    return data


@router.get("/oauth/{mcp_server_id}/login")
async def mcp_server_oauth_login(
    request: Request, mcp_server_id: str, redirect_to: str = "/"
):
    from litellm.proxy.management_endpoints.ui_sso import SSOAuthenticationHandler

    """
    Oauth flow for the MCP server

    1. Redirect to the MCP server's oauth login page
    2. MCP server redirects to the callback URL with the code
    3. Callback URL validates the code and returns the access token
    4. Access token is returned to the client
    """
    from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
        global_mcp_server_manager,
    )

    ## get the mcp server from the name
    mcp_server = global_mcp_server_manager.get_mcp_server_by_id(mcp_server_id)

    if mcp_server is None:
        raise HTTPException(status_code=404, detail="MCP server not found")

    if mcp_server.auth_type != MCPAuth.oauth2:
        raise HTTPException(
            status_code=400, detail="MCP server is not an OAuth2 server"
        )

    if mcp_server.oauth_info is None:
        raise HTTPException(status_code=400, detail="MCP server has no OAuth2 info")

    if mcp_server.oauth_info.authorization_url is None:
        raise HTTPException(
            status_code=400, detail="MCP server has no OAuth2 authorization URL"
        )

    state = str(uuid.uuid4())
    sessions[state] = {"created_at": time.time(), "redirect_to": redirect_to}

    params = {
        "redirect_uri": SSOAuthenticationHandler.get_redirect_url_for_sso(
            request=request, sso_callback_route=f"oauth/{mcp_server_id}/callback"
        ),
        "state": state,
    }

    if mcp_server.oauth_info.scopes is not None:
        params["scope"] = " ".join(mcp_server.oauth_info.scopes)
    if mcp_server.oauth_info.client_id is not None:
        params["client_id"] = mcp_server.oauth_info.client_id
    if mcp_server.oauth_info.client_secret is not None:
        params["client_secret"] = mcp_server.oauth_info.client_secret

    url = mcp_server.oauth_info.authorization_url
    return RedirectResponse(f"{url}?{httpx.QueryParams(params)}")
