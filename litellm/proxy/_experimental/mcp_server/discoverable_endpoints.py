import json
from typing import Optional
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
from litellm.proxy.common_utils.http_parsing_utils import _read_request_body
from litellm.types.mcp_server.mcp_server_manager import MCPServer

router = APIRouter(
    tags=["mcp"],
)


def get_request_base_url(request: Request) -> str:
    """
    Get the base URL for the request, considering X-Forwarded-* headers.

    When behind a proxy (like nginx), the proxy may set:
    - X-Forwarded-Proto: The original protocol (http/https)
    - X-Forwarded-Host: The original host (may include port)
    - X-Forwarded-Port: The original port (if not in Host header)

    Args:
        request: FastAPI Request object

    Returns:
        The reconstructed base URL (e.g., "https://proxy.example.com")
    """
    base_url = str(request.base_url).rstrip("/")
    parsed = urlparse(base_url)

    # Get forwarded headers
    x_forwarded_proto = request.headers.get("X-Forwarded-Proto")
    x_forwarded_host = request.headers.get("X-Forwarded-Host")
    x_forwarded_port = request.headers.get("X-Forwarded-Port")

    # Start with the original scheme
    scheme = x_forwarded_proto if x_forwarded_proto else parsed.scheme

    # Handle host and port
    if x_forwarded_host:
        # X-Forwarded-Host may already include port (e.g., "example.com:8080")
        if ":" in x_forwarded_host and not x_forwarded_host.startswith("["):
            # Host includes port
            netloc = x_forwarded_host
        elif x_forwarded_port:
            # Port is separate
            netloc = f"{x_forwarded_host}:{x_forwarded_port}"
        else:
            # Just host, no explicit port
            netloc = x_forwarded_host
    else:
        # No X-Forwarded-Host, use original netloc
        netloc = parsed.netloc
        if x_forwarded_port and ":" not in netloc:
            # Add forwarded port if not already in netloc
            netloc = f"{netloc}:{x_forwarded_port}"

    # Reconstruct the URL
    return urlunparse((scheme, netloc, parsed.path, "", "", ""))


def encode_state_with_base_url(
    base_url: str,
    original_state: str,
    code_challenge: Optional[str] = None,
    code_challenge_method: Optional[str] = None,
    client_redirect_uri: Optional[str] = None,
) -> str:
    """
    Encode the base_url, original state, and PKCE parameters using encryption.

    Args:
        base_url: The base URL to encode
        original_state: The original state parameter
        code_challenge: PKCE code challenge from client
        code_challenge_method: PKCE code challenge method from client
        client_redirect_uri: Original redirect_uri from client

    Returns:
        An encrypted string that encodes all values
    """
    state_data = {
        "base_url": base_url,
        "original_state": original_state,
        "code_challenge": code_challenge,
        "code_challenge_method": code_challenge_method,
        "client_redirect_uri": client_redirect_uri,
    }
    state_json = json.dumps(state_data, sort_keys=True)
    encrypted_state = encrypt_value_helper(state_json)
    return encrypted_state


def decode_state_hash(encrypted_state: str) -> dict:
    """
    Decode an encrypted state to retrieve all OAuth session data.

    Args:
        encrypted_state: The encrypted string to decode

    Returns:
        A dict containing base_url, original_state, and optional PKCE parameters

    Raises:
        Exception: If decryption fails or data is malformed
    """
    decrypted_json = decrypt_value_helper(encrypted_state, "oauth_state")
    if decrypted_json is None:
        raise ValueError("Failed to decrypt state parameter")

    state_data = json.loads(decrypted_json)
    return state_data


async def authorize_with_server(
    request: Request,
    mcp_server: MCPServer,
    client_id: str,
    redirect_uri: str,
    state: str = "",
    code_challenge: Optional[str] = None,
    code_challenge_method: Optional[str] = None,
    response_type: Optional[str] = None,
    scope: Optional[str] = None,
):
    if mcp_server.auth_type != "oauth2":
        raise HTTPException(status_code=400, detail="MCP server is not OAuth2")
    if mcp_server.authorization_url is None:
        raise HTTPException(
            status_code=400, detail="MCP server authorization url is not set"
        )

    parsed = urlparse(redirect_uri)
    base_url = urlunparse(parsed._replace(query=""))
    request_base_url = get_request_base_url(request)
    encoded_state = encode_state_with_base_url(
        base_url=base_url,
        original_state=state,
        code_challenge=code_challenge,
        code_challenge_method=code_challenge_method,
        client_redirect_uri=redirect_uri,
    )

    params = {
        "client_id": mcp_server.client_id if mcp_server.client_id else client_id,
        "redirect_uri": f"{request_base_url}/callback",
        "state": encoded_state,
        "response_type": response_type or "code",
    }
    if scope:
        params["scope"] = scope
    elif mcp_server.scopes:
        params["scope"] = " ".join(mcp_server.scopes)

    if code_challenge:
        params["code_challenge"] = code_challenge
    if code_challenge_method:
        params["code_challenge_method"] = code_challenge_method

    return RedirectResponse(f"{mcp_server.authorization_url}?{urlencode(params)}")


async def exchange_token_with_server(
    request: Request,
    mcp_server: MCPServer,
    grant_type: str,
    code: Optional[str],
    redirect_uri: Optional[str],
    client_id: str,
    client_secret: Optional[str],
    code_verifier: Optional[str],
):
    if grant_type != "authorization_code":
        raise HTTPException(status_code=400, detail="Unsupported grant_type")

    if mcp_server.token_url is None:
        raise HTTPException(status_code=400, detail="MCP server token url is not set")

    proxy_base_url = get_request_base_url(request)
    token_data = {
        "grant_type": "authorization_code",
        "client_id": mcp_server.client_id if mcp_server.client_id else client_id,
        "client_secret": mcp_server.client_secret
        if mcp_server.client_secret
        else client_secret,
        "code": code,
        "redirect_uri": f"{proxy_base_url}/callback",
    }

    if code_verifier:
        token_data["code_verifier"] = code_verifier

    async_client = get_async_httpx_client(llm_provider=httpxSpecialProvider.Oauth2Check)
    response = await async_client.post(
        mcp_server.token_url,
        headers={"Accept": "application/json"},
        data=token_data,
    )

    response.raise_for_status()
    token_response = response.json()
    access_token = token_response["access_token"]

    result = {
        "access_token": access_token,
        "token_type": token_response.get("token_type", "Bearer"),
        "expires_in": token_response.get("expires_in", 3600),
    }

    if "refresh_token" in token_response and token_response["refresh_token"]:
        result["refresh_token"] = token_response["refresh_token"]
    if "scope" in token_response and token_response["scope"]:
        result["scope"] = token_response["scope"]

    return JSONResponse(result)


async def register_client_with_server(
    request: Request,
    mcp_server: MCPServer,
    client_name: str,
    grant_types: Optional[list],
    response_types: Optional[list],
    token_endpoint_auth_method: Optional[str],
    fallback_client_id: Optional[str] = None,
):
    request_base_url = get_request_base_url(request)
    dummy_return = {
        "client_id": fallback_client_id or mcp_server.server_name,
        "client_secret": "dummy",
        "redirect_uris": [f"{request_base_url}/callback"],
    }

    if mcp_server.client_id and mcp_server.client_secret:
        return dummy_return

    if mcp_server.authorization_url is None:
        raise HTTPException(
            status_code=400, detail="MCP server authorization url is not set"
        )

    if mcp_server.registration_url is None:
        return dummy_return

    register_data = {
        "client_name": client_name,
        "redirect_uris": [f"{request_base_url}/callback"],
        "grant_types": grant_types or [],
        "response_types": response_types or [],
        "token_endpoint_auth_method": token_endpoint_auth_method or "",
    }
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    async_client = get_async_httpx_client(
        llm_provider=httpxSpecialProvider.Oauth2Register
    )
    response = await async_client.post(
        mcp_server.registration_url,
        headers=headers,
        json=register_data,
    )
    response.raise_for_status()

    token_response = response.json()

    return JSONResponse(token_response)


@router.get("/{mcp_server_name}/authorize")
@router.get("/authorize")
async def authorize(
    request: Request,
    client_id: str,
    redirect_uri: str,
    state: str = "",
    mcp_server_name: Optional[str] = None,
    code_challenge: Optional[str] = None,
    code_challenge_method: Optional[str] = None,
    response_type: Optional[str] = None,
    scope: Optional[str] = None,
):
    # Redirect to real OAuth provider with PKCE support
    from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
        global_mcp_server_manager,
    )

    lookup_name = mcp_server_name or client_id
    mcp_server = global_mcp_server_manager.get_mcp_server_by_name(lookup_name)
    if mcp_server is None:
        raise HTTPException(status_code=404, detail="MCP server not found")
    return await authorize_with_server(
        request=request,
        mcp_server=mcp_server,
        client_id=client_id,
        redirect_uri=redirect_uri,
        state=state,
        code_challenge=code_challenge,
        code_challenge_method=code_challenge_method,
        response_type=response_type,
        scope=scope,
    )


@router.post("/{mcp_server_name}/token")
@router.post("/token")
async def token_endpoint(
    request: Request,
    grant_type: str = Form(...),
    code: str = Form(None),
    redirect_uri: str = Form(None),
    client_id: str = Form(...),
    client_secret: Optional[str] = Form(None),
    code_verifier: str = Form(None),
    mcp_server_name: Optional[str] = None,
):
    """
    Accept the authorization code from client and exchange it for OAuth token.
    Supports PKCE flow by forwarding code_verifier to upstream provider.

    1. Call the token endpoint with PKCE parameters
    2. Store the user's token in the db - and generate a LiteLLM virtual key
    3. Return the token
    4. Return a virtual key in this response
    """
    from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
        global_mcp_server_manager,
    )

    lookup_name = mcp_server_name or client_id
    mcp_server = global_mcp_server_manager.get_mcp_server_by_name(lookup_name)
    if mcp_server is None:
        raise HTTPException(status_code=404, detail="MCP server not found")
    return await exchange_token_with_server(
        request=request,
        mcp_server=mcp_server,
        grant_type=grant_type,
        code=code,
        redirect_uri=redirect_uri,
        client_id=client_id,
        client_secret=client_secret,
        code_verifier=code_verifier,
    )


@router.get("/callback")
async def callback(code: str, state: str):
    try:
        # Decode the state hash to get base_url, original state, and PKCE params
        state_data = decode_state_hash(state)
        base_url = state_data["base_url"]
        original_state = state_data["original_state"]

        # Forward code and original state back to client
        params = {"code": code, "state": original_state}

        # Forward to client's callback endpoint
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
    # Get the correct base URL considering X-Forwarded-* headers
    request_base_url = get_request_base_url(request)
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
    # Get the correct base URL considering X-Forwarded-* headers
    request_base_url = get_request_base_url(request)

    authorization_endpoint = (
        f"{request_base_url}/{mcp_server_name}/authorize"
        if mcp_server_name
        else f"{request_base_url}/authorize"
    )
    token_endpoint = (
        f"{request_base_url}/{mcp_server_name}/token"
        if mcp_server_name
        else f"{request_base_url}/token"
    )

    return {
        "issuer": request_base_url,  # point to your proxy
        "authorization_endpoint": authorization_endpoint,
        "token_endpoint": token_endpoint,
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
    from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
        global_mcp_server_manager,
    )

    # Get the correct base URL considering X-Forwarded-* headers
    request_base_url = get_request_base_url(request)

    request_data = await _read_request_body(request=request)
    data: dict = {**request_data}

    dummy_return = {
        "client_id": mcp_server_name or "dummy_client",
        "client_secret": "dummy",
        "redirect_uris": [f"{request_base_url}/callback"],
    }
    if not mcp_server_name:
        return dummy_return

    mcp_server = global_mcp_server_manager.get_mcp_server_by_name(mcp_server_name)
    if mcp_server is None:
        return dummy_return
    return await register_client_with_server(
        request=request,
        mcp_server=mcp_server,
        client_name=data.get("client_name", ""),
        grant_types=data.get("grant_types", []),
        response_types=data.get("response_types", []),
        token_endpoint_auth_method=data.get("token_endpoint_auth_method", ""),
        fallback_client_id=mcp_server_name,
    )
