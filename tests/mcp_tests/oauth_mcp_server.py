import argparse
import base64
import hashlib
import os
import secrets
import time
from typing import Any, Dict, Optional
from urllib.parse import urlencode

import uvicorn
from mcp.server.fastmcp import FastMCP
from starlette.applications import Starlette
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, RedirectResponse, Response
from starlette.routing import Mount, Route

VALID_CLIENT_ID = "litellm-test-client"
VALID_CLIENT_SECRET = "litellm-test-secret"
DEFAULT_TOKEN_TTL = 3600
AUTH_CODE_TTL = 300


class TestM2MServer:
    def __init__(self) -> None:
        self.client_id = VALID_CLIENT_ID
        self.client_secret = VALID_CLIENT_SECRET
        self.token_url = "http://127.0.0.1:0/token"
        self.scopes = ["read", "call"]
        # permission management
        self.allow_all_keys = True
        self.extra_headers = ["x-litellm-session-id: dummy"]
        self.internal_network = False
        self.available_on_public_internet = True


class TestInteractiveServer:
    def __init__(self) -> None:
        self.client_id = VALID_CLIENT_ID
        self.client_secret = VALID_CLIENT_SECRET
        self.authorization_url = "http://127.0.0.1:0/authorize"
        self.token_url = "http://127.0.0.1:0/token"
        self.registration_url = "http://127.0.0.1:0/register"
        self.scopes = ["read", "call"]
        self.allow_all_keys = True


def _pkce_s256(code_verifier: str) -> str:
    digest = hashlib.sha256(code_verifier.encode("ascii")).digest()
    return base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")


class TokenStore:
    def __init__(self) -> None:
        self.tokens: Dict[str, Dict[str, Any]] = {}
        self.refresh_tokens: Dict[str, Dict[str, Any]] = {}
        self.auth_codes: Dict[str, Dict[str, Any]] = {}

    def issue(self, client_id: str, scope: str, resource: str, audience: str) -> str:
        token = "tk_" + secrets.token_urlsafe(24)
        self.tokens[token] = {
            "client_id": client_id,
            "scope": scope,
            "resource": resource,
            "audience": audience,
            "issued_at": time.time(),
            "expires_at": time.time() + DEFAULT_TOKEN_TTL,
        }
        return token

    def issue_with_refresh(self, client_id: str, scope: str, resource: str, audience: str) -> Dict[str, Any]:
        access_token = self.issue(client_id, scope, resource, audience)
        refresh_token = "rt_" + secrets.token_urlsafe(24)
        self.refresh_tokens[refresh_token] = {
            "client_id": client_id,
            "scope": scope,
            "resource": resource,
            "audience": audience,
        }
        return {"access_token": access_token, "refresh_token": refresh_token}

    def issue_auth_code(
        self,
        client_id: str,
        scope: str,
        resource: str,
        code_challenge: str,
        code_challenge_method: str,
        redirect_uri: str,
    ) -> str:
        code = "ac_" + secrets.token_urlsafe(24)
        self.auth_codes[code] = {
            "client_id": client_id,
            "scope": scope,
            "resource": resource,
            "code_challenge": code_challenge,
            "code_challenge_method": code_challenge_method,
            "redirect_uri": redirect_uri,
            "expires_at": time.time() + AUTH_CODE_TTL,
        }
        return code

    def consume_auth_code(self, code: str) -> Optional[Dict[str, Any]]:
        rec = self.auth_codes.pop(code, None)
        if rec is None or rec["expires_at"] <= time.time():
            return None
        return rec

    def consume_refresh_token(self, refresh_token: str) -> Optional[Dict[str, Any]]:
        return self.refresh_tokens.get(refresh_token)

    def is_valid(self, token: str) -> bool:
        rec = self.tokens.get(token)
        if rec is None:
            return False
        return rec["expires_at"] > time.time()

    def invalidate_all(self) -> int:
        n = len(self.tokens)
        self.tokens.clear()
        return n


store = TokenStore()
fastmcp = FastMCP("OAuthProtected")
fastmcp.settings.streamable_http_path = "/"


@fastmcp.tool()
def whoami_tool(label: str) -> str:
    return f"authorized:{label}"


@fastmcp.tool()
def echo_oauth(value: str) -> str:
    return value


async def authorize_endpoint(request: Request) -> Response:
    """Auto-approving PKCE authorization endpoint.

    Issues a code bound to the supplied code_challenge and redirects back to
    redirect_uri with code+state. No login UI; approval is implicit so the e2e
    flow can run headless.
    """
    params = dict(request.query_params)
    redirect_uri = params.get("redirect_uri")
    if not redirect_uri:
        return JSONResponse({"error": "invalid_request"}, status_code=400)
    if params.get("client_id") != VALID_CLIENT_ID:
        return JSONResponse({"error": "unauthorized_client"}, status_code=400)

    code = store.issue_auth_code(
        client_id=str(params.get("client_id", "")),
        scope=str(params.get("scope", "")),
        resource=str(params.get("resource", "")),
        code_challenge=str(params.get("code_challenge", "")),
        code_challenge_method=str(params.get("code_challenge_method", "")),
        redirect_uri=redirect_uri,
    )
    query = {"code": code}
    state = params.get("state")
    if state is not None:
        query["state"] = state
    sep = "&" if "?" in redirect_uri else "?"
    return RedirectResponse(url=f"{redirect_uri}{sep}{urlencode(query)}", status_code=302)


def _authorization_code_grant(body: Dict[str, Any]) -> Response:
    rec = store.consume_auth_code(str(body.get("code", "")))
    if rec is None:
        return JSONResponse({"error": "invalid_grant"}, status_code=400)

    code_challenge = rec.get("code_challenge")
    if code_challenge:
        code_verifier = str(body.get("code_verifier", ""))
        method = rec.get("code_challenge_method", "plain")
        derived = _pkce_s256(code_verifier) if method == "S256" else code_verifier
        if derived != code_challenge:
            return JSONResponse({"error": "invalid_grant"}, status_code=400)

    issued = store.issue_with_refresh(
        client_id=str(rec.get("client_id", "")),
        scope=str(rec.get("scope", "")),
        resource=str(rec.get("resource", "")),
        audience=str(rec.get("resource", "")),
    )
    return JSONResponse(
        {
            "access_token": issued["access_token"],
            "refresh_token": issued["refresh_token"],
            "token_type": "Bearer",
            "expires_in": DEFAULT_TOKEN_TTL,
        }
    )


def _refresh_token_grant(body: Dict[str, Any]) -> Response:
    rec = store.consume_refresh_token(str(body.get("refresh_token", "")))
    if rec is None:
        return JSONResponse({"error": "invalid_grant"}, status_code=400)
    token = store.issue(
        client_id=str(rec.get("client_id", "")),
        scope=str(rec.get("scope", "")),
        resource=str(rec.get("resource", "")),
        audience=str(rec.get("audience", "")),
    )
    return JSONResponse({"access_token": token, "token_type": "Bearer", "expires_in": DEFAULT_TOKEN_TTL})


def _client_credentials_grant(body: Dict[str, Any]) -> Response:
    if body.get("client_id") != VALID_CLIENT_ID or body.get("client_secret") != VALID_CLIENT_SECRET:
        return JSONResponse({"error": "invalid_client"}, status_code=401)
    token = store.issue(
        client_id=str(body.get("client_id", "")),
        scope=str(body.get("scope", "")),
        resource=str(body.get("resource", "")),
        audience=str(body.get("audience", "")),
    )
    return JSONResponse({"access_token": token, "token_type": "Bearer", "expires_in": DEFAULT_TOKEN_TTL})


async def token_endpoint(request: Request) -> Response:
    form = await request.form()
    body = {k: form[k] for k in form.keys()}
    grant_type = body.get("grant_type")
    if grant_type == "client_credentials":
        return _client_credentials_grant(body)
    if grant_type == "authorization_code":
        return _authorization_code_grant(body)
    if grant_type == "refresh_token":
        return _refresh_token_grant(body)
    return JSONResponse({"error": "unsupported_grant_type"}, status_code=400)


async def well_known_authorization_server(request: Request) -> Response:
    base = str(request.base_url).rstrip("/")
    return JSONResponse(
        {
            "issuer": base,
            "authorization_endpoint": f"{base}/authorize",
            "token_endpoint": f"{base}/token",
            "grant_types_supported": [
                "client_credentials",
                "authorization_code",
                "refresh_token",
            ],
            "response_types_supported": ["code"],
            "code_challenge_methods_supported": ["S256"],
            "token_endpoint_auth_methods_supported": ["client_secret_post"],
        }
    )


async def register_endpoint(request: Request) -> Response:
    """RFC 7591 dynamic client registration; echoes back the static test client."""
    try:
        body = await request.json()
    except Exception:
        body = {}
    return JSONResponse(
        {
            "client_id": VALID_CLIENT_ID,
            "client_secret": VALID_CLIENT_SECRET,
            "redirect_uris": body.get("redirect_uris", []),
            "grant_types": body.get("grant_types", []),
            "response_types": body.get("response_types", []),
            "token_endpoint_auth_method": body.get("token_endpoint_auth_method", "client_secret_post"),
        },
        status_code=201,
    )


async def well_known_protected_resource(request: Request) -> Response:
    base = str(request.base_url).rstrip("/")
    return JSONResponse(
        {
            "resource": f"{base}/mcp",
            "authorization_servers": [base],
            "bearer_methods_supported": ["header"],
        }
    )


async def test_issued_tokens(request: Request) -> Response:
    now = time.time()
    valid = [t for t, rec in store.tokens.items() if rec["expires_at"] > now]
    return JSONResponse(valid)


async def test_invalidate(request: Request) -> Response:
    n = store.invalidate_all()
    return JSONResponse({"invalidated": n})


class BearerAuthMiddleware(BaseHTTPMiddleware):
    """Reject any /mcp request without a valid Bearer token from `store`."""

    async def dispatch(self, request: Request, call_next):
        if not request.url.path.startswith("/mcp"):
            return await call_next(request)
        auth = request.headers.get("authorization", "")
        if not auth.lower().startswith("bearer "):
            base = str(request.base_url).rstrip("/")
            return JSONResponse(
                {"error": "missing_bearer"},
                status_code=401,
                headers={
                    "WWW-Authenticate": (f'Bearer resource_metadata="{base}/.well-known/oauth-protected-resource"')
                },
            )
        token = auth.split(" ", 1)[1].strip()
        if not store.is_valid(token):
            return JSONResponse({"error": "invalid_token"}, status_code=401)
        return await call_next(request)


def build_app() -> Starlette:
    mcp_app = fastmcp.streamable_http_app()
    routes = [
        Route("/authorize", authorize_endpoint, methods=["GET"]),
        Route("/token", token_endpoint, methods=["POST"]),
        Route("/register", register_endpoint, methods=["POST"]),
        Route(
            "/.well-known/oauth-authorization-server",
            well_known_authorization_server,
            methods=["GET"],
        ),
        Route(
            "/.well-known/oauth-protected-resource",
            well_known_protected_resource,
            methods=["GET"],
        ),
        Route("/_test/issued_tokens", test_issued_tokens, methods=["GET"]),
        Route("/_test/invalidate", test_invalidate, methods=["POST"]),
        Mount("/mcp", app=mcp_app),
    ]
    app = Starlette(
        routes=routes,
        middleware=[],
        lifespan=mcp_app.router.lifespan_context,
    )
    app.add_middleware(BearerAuthMiddleware)
    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="OAuth-protected MCP test server")
    parser.add_argument("--host", default=os.getenv("MCP_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("MCP_PORT", "0")))
    args = parser.parse_args()
    if args.port <= 0:
        raise ValueError("OAuth MCP server requires --port > 0")
    uvicorn.run(build_app(), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
