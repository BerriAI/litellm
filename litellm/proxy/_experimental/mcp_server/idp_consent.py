"""Pure logic for the delegated-OBO consent-capture flow (Path B, item 2.2).

A server-terminal ``authorization_code + offline_access`` flow against the user's IdP: unlike the
per-user MCP-server OAuth flow (which relays the code back to a browser client that finishes the
exchange), here the gateway is the OAuth client end to end. It generates its own PKCE, seals the user
and the target IdP into the state, and on callback exchanges the code for the user's tokens itself,
then stores the refresh_token as the user's IdP grant for the mint arm to use.

This module is the pure core (PKCE, state sealing, authorize-URL building, code exchange); the HTTP
POST, the crypto, and the persistence are injected so it is testable without a live IdP or DB. The
endpoints that wire it onto the request path live with the other ``/v1/mcp`` management routes.
"""

from __future__ import annotations

import base64
import hashlib
import secrets
from collections.abc import Awaitable, Callable
from urllib.parse import urlencode, urlsplit, urlunsplit

from pydantic import BaseModel, ConfigDict

from litellm.proxy._experimental.mcp_server.auth.token_endpoint_auth import (
    TokenEndpointAuthConfigError,
    build_token_endpoint_client_auth,
)
from litellm.proxy._experimental.mcp_server.outbound_credentials.idp_oauth_config import (
    IdpOAuthProvider,
)

# POSTs an OAuth form to a token endpoint and returns the parsed JSON body, or None on any failure.
TokenEndpointPost = Callable[[str, "dict[str, str]", "dict[str, str]"], Awaitable["dict[str, object] | None"]]

_PKCE_VERIFIER_BYTES = 64
_CODE_CHALLENGE_METHOD = "S256"


_STATE_MAX_AGE_SECONDS = 600.0


class CaptureState(BaseModel):
    """The sealed state carried through the IdP round-trip: who is consenting, for which IdP, and the
    PKCE verifier to replay at the token exchange. Sealed (encrypted) so the client cannot tamper with
    the bound user or forge a verifier, and stamped with ``issued_at`` so a captured state cannot be
    replayed indefinitely."""

    model_config = ConfigDict(frozen=True)

    user_id: str
    token_url: str
    code_verifier: str
    issued_at: float


def state_is_fresh(state: CaptureState, *, now: float, max_age_seconds: float = _STATE_MAX_AGE_SECONDS) -> bool:
    """Whether the sealed state is within its lifetime, bounding replay of a captured state."""
    return 0 <= (now - state.issued_at) <= max_age_seconds


class CapturedGrant(BaseModel):
    """The user's IdP grant captured from the authorization_code exchange."""

    model_config = ConfigDict(frozen=True)

    access_token: str
    refresh_token: str | None
    expires_in: int | None
    scopes: tuple[str, ...]


def generate_pkce() -> tuple[str, str]:
    """Return a fresh ``(code_verifier, code_challenge)`` pair (RFC 7636 S256)."""
    verifier = secrets.token_urlsafe(_PKCE_VERIFIER_BYTES)
    digest = hashlib.sha256(verifier.encode()).digest()
    challenge = base64.urlsafe_b64encode(digest).decode().rstrip("=")
    return verifier, challenge


def seal_capture_state(state: CaptureState, encrypt: Callable[[str], str]) -> str:
    return encrypt(state.model_dump_json())


def unseal_capture_state(blob: str, decrypt: Callable[[str], str | None]) -> CaptureState | None:
    decrypted = decrypt(blob)
    if decrypted is None:
        return None
    try:
        return CaptureState.model_validate_json(decrypted)
    except ValueError:
        return None


def build_authorize_url(
    provider: IdpOAuthProvider,
    *,
    redirect_uri: str,
    state: str,
    code_challenge: str,
) -> str:
    """Build the IdP authorize URL for the consent redirect, merging the OAuth params onto any query
    the configured ``authorize_url`` already carries."""
    params = {
        "response_type": "code",
        "client_id": provider.client_id,
        "redirect_uri": redirect_uri,
        "state": state,
        "scope": " ".join(provider.scopes),
        "code_challenge": code_challenge,
        "code_challenge_method": _CODE_CHALLENGE_METHOD,
    }
    parts = urlsplit(provider.authorize_url)
    merged_query = "&".join(q for q in (parts.query, urlencode(params)) if q)
    return urlunsplit((parts.scheme, parts.netloc, parts.path, merged_query, parts.fragment))


def _parse_expires_in(raw: object) -> int | None:
    if isinstance(raw, bool):
        return None
    if isinstance(raw, (int, float)):
        return int(raw)
    if isinstance(raw, str):
        try:
            return int(float(raw))
        except ValueError:
            return None
    return None


def _parse_scopes(raw: object, fallback: tuple[str, ...]) -> tuple[str, ...]:
    if isinstance(raw, str) and raw:
        return tuple(raw.split())
    return fallback


async def default_token_post(url: str, form: dict[str, str], headers: dict[str, str]) -> dict[str, object] | None:
    """The real httpx POST to an IdP token endpoint for the consent exchange (Oauth2Check provider).

    Returns the parsed JSON body, or None on any failure so the exchange fails closed. Mirrors the
    per-provider token-endpoint POST helpers; extracting one shared helper across them is a follow-up.
    """
    from litellm._logging import verbose_logger  # noqa: PLC0415  # lazy import; avoids cycle
    from litellm.llms.custom_httpx.http_handler import (  # noqa: PLC0415  # lazy import; avoids cycle
        get_async_httpx_client,  # pyright: ignore[reportUnknownVariableType]  # httpx handler untyped
    )
    from litellm.types.llms.custom_http import httpxSpecialProvider  # noqa: PLC0415  # lazy import

    request_headers = {"Accept": "application/json", **headers}
    try:
        client = get_async_httpx_client(llm_provider=httpxSpecialProvider.Oauth2Check)
        response = await client.post(url, headers=request_headers, data=form)  # pyright: ignore[reportUnknownMemberType]  # AsyncHTTPHandler.post untyped
        if response is None:
            return None
        response.raise_for_status()
        body: dict[str, object] = response.json()  # pyright: ignore[reportAny]  # untyped JSON body, validated below
    except Exception as exc:  # noqa: BLE001  # any IdP/transport error is a capture miss, not a 500
        verbose_logger.warning("MCP IdP consent code exchange failed: %s", exc)
        return None
    else:
        return body


async def exchange_code_for_grant(
    provider: IdpOAuthProvider,
    *,
    code: str,
    redirect_uri: str,
    code_verifier: str,
    post: TokenEndpointPost,
) -> CapturedGrant | None:
    """Exchange the authorization code for the user's IdP grant (server-side), or None on failure.

    Fails closed (None) rather than raising: a missing access_token, a bad client-auth config, or any
    transport error is a capture miss the caller surfaces as an error, never a partial grant.
    """
    try:
        client_auth = build_token_endpoint_client_auth(
            auth_method=None,
            client_id=provider.client_id,
            client_secret=provider.client_secret.get_secret_value(),
        )
    except TokenEndpointAuthConfigError:
        return None
    form = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": redirect_uri,
        "code_verifier": code_verifier,
        **client_auth.body,
    }
    body = await post(provider.token_url, form, client_auth.headers)
    if body is None:
        return None
    access_token = body.get("access_token")
    if not isinstance(access_token, str) or not access_token:
        return None
    refresh_token = body.get("refresh_token")
    return CapturedGrant(
        access_token=access_token,
        refresh_token=refresh_token if isinstance(refresh_token, str) and refresh_token else None,
        expires_in=_parse_expires_in(body.get("expires_in")),
        scopes=_parse_scopes(body.get("scope"), provider.scopes),
    )
