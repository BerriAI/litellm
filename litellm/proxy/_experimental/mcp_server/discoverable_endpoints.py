import json
from typing import Optional, Tuple
from urllib.parse import urlencode, urlparse, urlunparse

from fastapi import APIRouter, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

from litellm.llms.custom_httpx.http_handler import (
    get_async_httpx_client,
    httpxSpecialProvider,
)
from litellm.proxy.common_utils.encrypt_decrypt_utils import (
    decrypt_value_helper,
    encrypt_value_helper,
)

router = APIRouter(
    tags=["mcp"],
)


def encode_state_with_base_url(base_url: str, original_state: str) -> str:
    """
    Encode the base_url and original state using encryption.

    Args:
        base_url: The base URL to encode
        original_state: The original state parameter

    Returns:
        An encrypted string that encodes both values
    """
    state_data = {"base_url": base_url, "original_state": original_state}
    state_json = json.dumps(state_data, sort_keys=True)
    encrypted_state = encrypt_value_helper(state_json)
    return encrypted_state


def decode_state_hash(encrypted_state: str) -> Tuple[str, str]:
    """
    Decode an encrypted state to retrieve the base_url and original state.

    Args:
        encrypted_state: The encrypted string to decode

    Returns:
        A tuple of (base_url, original_state)

    Raises:
        Exception: If decryption fails or data is malformed
    """
    decrypted_json = decrypt_value_helper(encrypted_state, "oauth_state")
    if decrypted_json is None:
        raise ValueError("Failed to decrypt state parameter")

    state_data = json.loads(decrypted_json)
    return state_data["base_url"], state_data["original_state"]


@router.get("/{mcp_server_name}/authorize")
@router.get("/authorize")
async def authorize(
    request: Request,
    client_id: str,
    redirect_uri: str,
    state: str = "",
    mcp_server_name: Optional[str] = None,
):
    # Redirect to real GitHub OAuth
    from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
        global_mcp_server_manager,
    )

    mcp_server = global_mcp_server_manager.get_mcp_server_by_name(client_id)
    if mcp_server is None:
        raise HTTPException(status_code=404, detail="MCP server not found")
    if mcp_server.auth_type != "oauth2":
        raise HTTPException(status_code=400, detail="MCP server is not OAuth2")
    if mcp_server.client_id is None:
        raise HTTPException(status_code=400, detail="MCP server client id is not set")
    if mcp_server.authorization_url is None:
        raise HTTPException(
            status_code=400, detail="MCP server authorization url is not set"
        )
    if mcp_server.scopes is None:
        raise HTTPException(status_code=400, detail="MCP server scopes is not set")

    # Parse it to remove any existing query
    parsed = urlparse(redirect_uri)
    base_url = urlunparse(parsed._replace(query=""))
    request_base_url = str(request.base_url).rstrip("/")

    # Encode the base_url and original state in a unique hash
    encoded_state = encode_state_with_base_url(base_url, state)

    params = {
        "client_id": mcp_server.client_id,
        "redirect_uri": f"{request_base_url}/callback",
        "scope": " ".join(mcp_server.scopes),
        "state": encoded_state,
    }
    return RedirectResponse(f"{mcp_server.authorization_url}?{urlencode(params)}")


@router.post("/token")
async def token_endpoint(
    request: Request,
    grant_type: str = Form(...),
    code: str = Form(None),
    redirect_uri: str = Form(None),
    client_id: str = Form(...),
    client_secret: str = Form(...),
):
    """
    Accept the authorization code from Claude and exchange it for GitHub token.
    Forward the GitHub token back to Claude in standard OAuth format.

    1. Call the token endpoint
    2. Store the user's PAT in the db - and generate a LiteLLM virtual key
    2. Return the token
    3. Return a virtual key in this response
    """
    from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
        global_mcp_server_manager,
    )

    mcp_server = global_mcp_server_manager.get_mcp_server_by_name(client_id)
    if mcp_server is None:
        raise HTTPException(status_code=404, detail="MCP server not found")

    if grant_type != "authorization_code":
        raise HTTPException(status_code=400, detail="Unsupported grant_type")

    if mcp_server.token_url is None:
        raise HTTPException(status_code=400, detail="MCP server token url is not set")

    proxy_base_url = str(request.base_url).rstrip("/")

    # Exchange code for real GitHub token
    async_client = get_async_httpx_client(llm_provider=httpxSpecialProvider.Oauth2Check)
    response = await async_client.post(
        mcp_server.token_url,
        headers={"Accept": "application/json"},
        data={
            "client_id": mcp_server.client_id,
            "client_secret": mcp_server.client_secret,
            "code": code,
            "redirect_uri": f"{proxy_base_url}/callback",
        },
    )

    response.raise_for_status()
    github_token = response.json()["access_token"]

    # Return to Claude in expected OAuth 2 format

    ### return a virtual key in this response

    return JSONResponse(
        {"access_token": github_token, "token_type": "Bearer", "expires_in": 3600}
    )


@router.get("/callback")
async def callback(code: str, state: str):
    try:
        # Decode the state hash to get base_url and original state
        base_url, original_state = decode_state_hash(state)

        # Exchange code for token with GitHub
        params = {"code": code, "state": original_state}

        # Forward token to Claude ephemeral endpoint
        complete_returned_url = f"{base_url}?{urlencode(params)}"
        return RedirectResponse(url=complete_returned_url, status_code=302)

    except Exception:
        # fallback if state hash not found
        return HTMLResponse(
            "<html><body>Authentication incomplete. You can close this window.</body></html>"
        )


# ------------------------------
# Optional .well-known endpoints for MCP + OAuth discovery
# ------------------------------
@router.get("/.well-known/oauth-protected-resource/{mcp_server_name}/mcp")
@router.get("/.well-known/oauth-protected-resource")
async def oauth_protected_resource_mcp(
    request: Request, mcp_server_name: Optional[str] = None
):
    request_base_url = str(request.base_url).rstrip("/")
    return {
        "authorization_servers": [
            (
                f"{request_base_url}/{mcp_server_name}"
                if mcp_server_name
                else f"{request_base_url}"
            )
        ],
        "resource": (
            f"{request_base_url}/{mcp_server_name}/mcp"
            if mcp_server_name
            else f"{request_base_url}/mcp"
        ),  # this is what Claude will call
    }


@router.get("/.well-known/oauth-authorization-server/{mcp_server_name}")
@router.get("/.well-known/oauth-authorization-server")
async def oauth_authorization_server_mcp(
    request: Request, mcp_server_name: Optional[str] = None
):
    request_base_url = str(request.base_url).rstrip("/")
    return {
        "issuer": request_base_url,  # point to your proxy
        "authorization_endpoint": f"{request_base_url}/authorize",
        "token_endpoint": f"{request_base_url}/token",
        "response_types_supported": ["code"],
        "grant_types_supported": ["authorization_code"],
        "code_challenge_methods_supported": ["S256"],
        "token_endpoint_auth_methods_supported": ["client_secret_post"],
        # Claude expects a registration endpoint, even if we just fake it
        "registration_endpoint": f"{request_base_url}/{mcp_server_name}/register",
    }


# Alias for standard OpenID discovery
@router.get("/.well-known/openid-configuration")
async def openid_configuration(request: Request):
    return await oauth_authorization_server_mcp(request)


@router.get("/.well-known/oauth-authorization-server/{mcp_server_name}/mcp")
@router.get("/.well-known/oauth-authorization-server")
async def oauth_authorization_server_root(
    request: Request, mcp_server_name: Optional[str] = None
):
    return await oauth_authorization_server_mcp(request, mcp_server_name)


@router.post("/{mcp_server_name}/register")
@router.post("/register")
async def register_client(request: Request, mcp_server_name: Optional[str] = None):
    request_base_url = str(request.base_url).rstrip("/")

    # return fixed GitHub client credentials
    return {
        "client_id": mcp_server_name or "dummy_client",
        "client_secret": "dummy",
        "redirect_uris": [f"{request_base_url}/mcp/callback"],
    }
