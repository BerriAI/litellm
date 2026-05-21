#!/usr/bin/env python3
"""
Minimal OAuth2/OIDC mock IdP for local LiteLLM Generic SSO + CLI SSO testing.

Usage:
  python scripts/mock_oidc_server_for_cli_sso.py

Point LiteLLM at this server (see scripts/test_cli_sso_claims_e2e.py header).
"""

from __future__ import annotations

import os
import secrets
from typing import Any, Dict
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, RedirectResponse

MOCK_PORT = int(os.getenv("MOCK_OIDC_PORT", "8765"))
MOCK_CLIENT_ID = os.getenv("MOCK_OIDC_CLIENT_ID", "litellm-cli-test")
MOCK_CLIENT_SECRET = os.getenv("MOCK_OIDC_CLIENT_SECRET", "litellm-cli-test-secret")
MOCK_AUTH_CODE = os.getenv("MOCK_OIDC_AUTH_CODE", "mock-auth-code")

# Claims returned from /userinfo (tune for CLI_SSO_CLAIM_MAP tests)
MOCK_USERINFO: Dict[str, Any] = {
    "sub": os.getenv("MOCK_OIDC_SUB", "cli-test-user"),
    "preferred_username": os.getenv("MOCK_OIDC_SUB", "cli-test-user"),
    "email": os.getenv("MOCK_OIDC_EMAIL", "cli-test@example.com"),
    "employment_type": os.getenv("MOCK_OIDC_EMPLOYMENT_TYPE", "contractor"),
    "org_info": {"department": os.getenv("MOCK_OIDC_DEPARTMENT", "Engineering")},
}

app = FastAPI(title="LiteLLM CLI SSO mock OIDC", docs_url=None, redoc_url=None)
_issued_tokens: Dict[str, str] = {}


def _append_query(url: str, params: Dict[str, str]) -> str:
    parsed = urlparse(url)
    query = parse_qs(parsed.query)
    for key, value in params.items():
        query[key] = [value]
    flat = [(k, v[0]) for k, values in query.items() for v in [values]]
    return urlunparse(parsed._replace(query=urlencode(flat)))


@app.get("/authorize")
async def authorize(request: Request):
    """Auto-approve: redirect back to LiteLLM /sso/callback with a fixed code."""
    query = dict(request.query_params)
    redirect_uri = query.get("redirect_uri")
    state = query.get("state", "")
    if not redirect_uri:
        return JSONResponse({"error": "missing redirect_uri"}, status_code=400)

    if query.get("client_id") not in (None, MOCK_CLIENT_ID):
        return JSONResponse({"error": "invalid client_id"}, status_code=400)

    return RedirectResponse(
        _append_query(
            redirect_uri,
            {"code": MOCK_AUTH_CODE, "state": state},
        ),
        status_code=302,
    )


@app.post("/token")
async def token(request: Request):
    """Return a bearer token for any authorization_code grant."""
    body = await request.body()
    form = {k: v[0] for k, v in parse_qs(body.decode()).items()}

    if form.get("grant_type") != "authorization_code":
        return JSONResponse({"error": "unsupported_grant_type"}, status_code=400)

    if form.get("code") != MOCK_AUTH_CODE:
        return JSONResponse({"error": "invalid_grant"}, status_code=400)

    if form.get("client_secret") not in (None, MOCK_CLIENT_SECRET):
        return JSONResponse({"error": "invalid_client"}, status_code=401)

    if form.get("client_id") not in (None, MOCK_CLIENT_ID):
        return JSONResponse({"error": "invalid_client"}, status_code=401)

    access_token = secrets.token_urlsafe(24)
    _issued_tokens[access_token] = MOCK_USERINFO["sub"]
    return {
        "access_token": access_token,
        "token_type": "Bearer",
        "expires_in": 3600,
    }


@app.get("/userinfo")
async def userinfo(request: Request):
    auth = request.headers.get("authorization", "")
    if not auth.lower().startswith("bearer "):
        return JSONResponse({"error": "invalid_token"}, status_code=401)
    token = auth.split(" ", 1)[1].strip()
    if token not in _issued_tokens:
        return JSONResponse({"error": "invalid_token"}, status_code=401)
    return MOCK_USERINFO


@app.get("/health")
async def health():
    return {"status": "ok", "userinfo": MOCK_USERINFO}


if __name__ == "__main__":
    import uvicorn

    print(f"Mock OIDC listening on http://127.0.0.1:{MOCK_PORT}")
    print(f"  client_id={MOCK_CLIENT_ID}  client_secret={MOCK_CLIENT_SECRET}")
    print(f"  userinfo claims: {MOCK_USERINFO}")
    uvicorn.run(app, host="127.0.0.1", port=MOCK_PORT, log_level="info")
