"""OAuth token response validation.

The LiteLLM bearer credential is always the OAuth **access token**. The ID
token is never used as an API credential and is never persisted, and access
token claims are never treated as trusted identity -- the proxy remains
responsible for verifying signature, issuer, audience, expiry and claims.
"""

import base64
import binascii
import json
import time
from collections.abc import Mapping
from dataclasses import dataclass

from litellm.litellm_core_utils.native_oidc_validation import is_valid_nqchar_string, is_valid_scope_token

from .errors import NativeOIDCError

SUPPORTED_TOKEN_TYPE = "bearer"

# Guards against a provider advertising an absurd lifetime.
MAX_EXPIRES_IN_SECONDS = 90 * 24 * 3600

# Used only when the provider gives no usable expiry at all. Deliberately short
# relative to the legacy proxy-minted CLI token: identity-provider access
# tokens are typically an hour or less, and a too-long guess would mean sending
# a dead token to the proxy.
FALLBACK_LIFETIME_SECONDS = 3600

# Bounds an otherwise unbounded provider-controlled string before it is printed.
MAX_OAUTH_ERROR_LENGTH = 128


@dataclass(frozen=True)
class TokenResponse:
    """A validated OAuth token response.

    `expires_at` is a *local cache deadline* only -- never an authorization
    decision.
    """

    access_token: str
    token_type: str
    expires_at: float | None
    refresh_token: str | None
    scopes: tuple[str, ...] | None


def _decode_unverified_jwt_exp(access_token: str) -> float | None:
    """Read `exp` from a structurally-JWT access token as an untrusted hint.

    Used *only* to shorten a local cache deadline. Never for identity,
    authorization, issuer trust, audience trust or display.
    """
    parts = access_token.split(".")
    if len(parts) != 3:
        return None
    payload_segment = parts[1]
    padding = "=" * (-len(payload_segment) % 4)
    try:
        payload = json.loads(base64.urlsafe_b64decode(payload_segment + padding))
    except (ValueError, binascii.Error):
        return None
    if not isinstance(payload, Mapping):
        return None
    exp = payload.get("exp")
    if isinstance(exp, bool) or not isinstance(exp, (int, float)):
        return None
    return float(exp)


def _parse_expires_in(raw: object) -> int | None:
    """Validate `expires_in`, accepting the numeric-string form some IdPs send."""
    if raw is None:
        return None
    if isinstance(raw, bool):
        raise NativeOIDCError("token response expires_in is not a positive integer")
    if isinstance(raw, str):
        if not raw.strip().isdigit():
            raise NativeOIDCError("token response expires_in is not a positive integer")
        value = int(raw.strip())
    elif isinstance(raw, int):
        value = raw
    else:
        raise NativeOIDCError("token response expires_in is not a positive integer")
    if value <= 0 or value > MAX_EXPIRES_IN_SECONDS:
        raise NativeOIDCError("token response expires_in is out of range")
    return value


def _parse_scope(raw: object) -> tuple[str, ...] | None:
    if raw is None:
        return None
    if not isinstance(raw, str):
        raise NativeOIDCError("token response scope is not a string")
    scopes = tuple(raw.split())
    if any(not is_valid_scope_token(scope) for scope in scopes):
        raise NativeOIDCError("token response scope contains an invalid scope-token")
    return scopes or None


def compute_expires_at(access_token: str, expires_in: int | None, *, now: float | None = None) -> float:
    """Derive the local cache deadline.

    Prefers a valid `expires_in`; falls back to an untrusted JWT `exp`; and when
    both are available takes whichever is earlier, so the CLI errs towards
    refreshing sooner rather than sending a dead token.
    """
    current_time = time.time() if now is None else now
    jwt_exp = _decode_unverified_jwt_exp(access_token)
    candidates = tuple(
        value
        for value in (
            current_time + expires_in if expires_in is not None else None,
            jwt_exp if jwt_exp is not None and jwt_exp > current_time else None,
        )
        if value is not None
    )
    if not candidates:
        return current_time + FALLBACK_LIFETIME_SECONDS
    return min(candidates)


def parse_token_response(payload: object, *, now: float | None = None) -> TokenResponse:
    """Validate a browser, device, or refresh token response."""
    if not isinstance(payload, dict):
        raise NativeOIDCError("token endpoint did not return a JSON object")

    access_token = payload.get("access_token")
    if not isinstance(access_token, str) or not access_token:
        raise NativeOIDCError("token response did not contain an access_token")

    token_type = payload.get("token_type")
    if not isinstance(token_type, str) or not token_type:
        raise NativeOIDCError("token response did not contain a token_type")
    if token_type.lower() != SUPPORTED_TOKEN_TYPE:
        raise NativeOIDCError(f"unsupported token_type '{token_type}'; only Bearer tokens are supported")

    refresh_token = payload.get("refresh_token")
    if refresh_token is not None and (not isinstance(refresh_token, str) or not refresh_token):
        raise NativeOIDCError("token response refresh_token is not a non-empty string")

    expires_in = _parse_expires_in(payload.get("expires_in"))

    return TokenResponse(
        access_token=access_token,
        token_type="Bearer",
        expires_at=compute_expires_at(access_token, expires_in, now=now),
        refresh_token=refresh_token,
        scopes=_parse_scope(payload.get("scope")),
    )


def extract_oauth_error(payload: Mapping[str, object] | None) -> str | None:
    """Return the OAuth `error` code from an error response, if present.

    A code that is not a bounded RFC 6749 NQCHAR string is discarded rather than
    reported, so a hostile provider cannot smuggle ANSI or OSC escape sequences
    into the terminal through an error response. Callers then fall back to the
    status-code-only message.
    """
    if not isinstance(payload, Mapping):
        return None
    error = payload.get("error")
    if not isinstance(error, str) or len(error) > MAX_OAUTH_ERROR_LENGTH or not is_valid_nqchar_string(error):
        return None
    return error


def describe_token_error(status_code: int, payload: Mapping[str, object] | None) -> str:
    """Build a safe message for a failed token request.

    Only the stable OAuth error code is surfaced -- never the raw body, which
    can echo the submitted code or verifier.
    """
    error = extract_oauth_error(payload)
    if error:
        return f"token endpoint returned OAuth error '{error}' (HTTP {status_code})"
    return f"token endpoint returned HTTP {status_code}"
