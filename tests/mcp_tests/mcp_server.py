# math_server.py
import argparse
import os
from typing import Optional

from mcp.server.fastmcp import FastMCP
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

DEFAULT_API_KEY = "test-api-key"
DEFAULT_BEARER_TOKEN = "test-bearer-token"
DEFAULT_AUTHORIZATION_VALUE = "Custom raw-auth-value"
DEFAULT_CUSTOM_HEADER = "x-custom-token"
DEFAULT_CUSTOM_HEADER_VALUE = "custom-header-value"
DEFAULT_CLIENT_ID = "test-client"
DEFAULT_CLIENT_SECRET = "test-secret"
DEFAULT_OAUTH_ACCESS_TOKEN = "test-oauth-access-token"
DEFAULT_OBO_ACCESS_TOKEN = "test-obo-exchanged-token"

TOKEN_EXCHANGE_GRANT_TYPE = "urn:ietf:params:oauth:grant-type:token-exchange"
VALID_OAUTH_BEARER_TOKENS = {DEFAULT_OAUTH_ACCESS_TOKEN, DEFAULT_OBO_ACCESS_TOKEN}

mcp = FastMCP("Math")

_auth_mode = "none"
_auth_secret: Optional[str] = None
_client_id = DEFAULT_CLIENT_ID
_client_secret = DEFAULT_CLIENT_SECRET
_expected_subject_token: Optional[str] = None
_expected_audience: Optional[str] = None
_expected_scope: Optional[str] = None


def _request_is_authorized(headers) -> bool:
    if _auth_mode == "none":
        return True
    if _auth_mode == "api_key":
        return headers.get("x-api-key") == _auth_secret
    if _auth_mode == "bearer_token":
        return headers.get("authorization") == f"Bearer {_auth_secret}"
    if _auth_mode == "authorization":
        return headers.get("authorization") == _auth_secret
    if _auth_mode == "custom_header":
        return headers.get(DEFAULT_CUSTOM_HEADER) == _auth_secret
    if _auth_mode == "oauth2":
        auth = headers.get("authorization") or ""
        return (
            auth.startswith("Bearer ")
            and auth[len("Bearer ") :] in VALID_OAUTH_BEARER_TOKENS
        )
    return False


class _AuthMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        request = Request(scope)
        if request.url.path.startswith("/oauth/"):
            await self.app(scope, receive, send)
            return
        if not _request_is_authorized(request.headers):
            response = JSONResponse(
                {"error": "unauthorized", "auth_mode": _auth_mode}, status_code=401
            )
            await response(scope, receive, send)
            return
        await self.app(scope, receive, send)


@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b


@mcp.tool()
def multiply(a: int, b: int) -> int:
    """Multiply two numbers"""
    return a * b


@mcp.custom_route("/oauth/token", methods=["POST"])
async def oauth_token(request: Request) -> JSONResponse:
    form = await request.form()
    grant_type = form.get("grant_type")

    if (
        form.get("client_id") != _client_id
        or form.get("client_secret") != _client_secret
    ):
        return JSONResponse({"error": "invalid_client"}, status_code=401)

    if grant_type == "client_credentials":
        return JSONResponse(
            {
                "access_token": DEFAULT_OAUTH_ACCESS_TOKEN,
                "token_type": "Bearer",
                "expires_in": 3600,
            }
        )

    if grant_type == TOKEN_EXCHANGE_GRANT_TYPE:
        if not form.get("subject_token"):
            return JSONResponse({"error": "invalid_request"}, status_code=400)
        if (
            _expected_subject_token
            and form.get("subject_token") != _expected_subject_token
        ):
            return JSONResponse({"error": "invalid_subject_token"}, status_code=400)
        if _expected_audience and form.get("audience") != _expected_audience:
            return JSONResponse({"error": "invalid_audience"}, status_code=400)
        if _expected_scope and form.get("scope") != _expected_scope:
            return JSONResponse({"error": "invalid_scope"}, status_code=400)
        return JSONResponse(
            {
                "access_token": DEFAULT_OBO_ACCESS_TOKEN,
                "token_type": "Bearer",
                "expires_in": 3600,
            }
        )

    return JSONResponse({"error": "unsupported_grant_type"}, status_code=400)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MCP math test server")
    parser.add_argument(
        "--transport",
        default=os.getenv("MCP_TRANSPORT", "stdio"),
        help="Transport to use (stdio or http)",
    )
    parser.add_argument(
        "--host",
        default=os.getenv("MCP_HOST", "127.0.0.1"),
        help="Host to bind when serving over HTTP",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("MCP_PORT", "0")),
        help="Port to bind when serving over HTTP",
    )
    parser.add_argument("--auth-mode", default="none")
    parser.add_argument("--auth-secret", default=None)
    parser.add_argument("--client-id", default=DEFAULT_CLIENT_ID)
    parser.add_argument("--client-secret", default=DEFAULT_CLIENT_SECRET)
    parser.add_argument("--expected-subject-token", default=None)
    parser.add_argument("--expected-audience", default=None)
    parser.add_argument("--expected-scope", default=None)
    return parser.parse_args()


def main() -> None:
    global _auth_mode, _auth_secret, _client_id, _client_secret
    global _expected_audience, _expected_scope, _expected_subject_token
    args = _parse_args()
    transport = (args.transport or "stdio").lower()

    _auth_mode = args.auth_mode
    _auth_secret = args.auth_secret
    _client_id = args.client_id
    _client_secret = args.client_secret
    _expected_subject_token = args.expected_subject_token
    _expected_audience = args.expected_audience
    _expected_scope = args.expected_scope

    if transport == "stdio":
        mcp.run(transport="stdio")
        return

    if transport in {"http", "streamable_http", "streamable-http"}:
        if args.port <= 0:
            raise ValueError("HTTP transport requires a valid --port value")
        mcp.settings.host = args.host
        mcp.settings.port = args.port

        original_app = mcp.streamable_http_app

        def _app_with_auth():
            return _AuthMiddleware(original_app())

        mcp.streamable_http_app = _app_with_auth  # type: ignore[method-assign]
        mcp.run(transport="streamable-http")
        return

    raise ValueError(f"Unsupported transport: {transport}")


if __name__ == "__main__":
    main()
