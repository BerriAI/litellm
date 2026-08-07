"""Native OIDC credential persistence, verification and refresh.

The stored credential keeps the usable bearer token under the existing `key`
field so current CLI and SDK integrations keep working unchanged; native-only
metadata is added alongside it under an explicit `auth_type`.

Never persisted: authorization codes, device codes, user codes, OAuth state,
PKCE verifiers/challenges, client secrets, raw provider responses, or ID tokens.
"""

import errno
import os
import time
from collections.abc import Callable, Generator, Mapping
from contextlib import contextmanager
from typing import Final, Protocol

import requests

from litellm.litellm_core_utils.cli_token_utils import (
    CLI_TOKEN_FRESHNESS_BUFFER_HOURS,
    get_cli_token_file_path,
    load_cli_token,
)

from ..commands.private_json import write_private_json
from .errors import NativeOIDCAuthRejected, NativeOIDCError
from .http_client import post_form
from .metadata import (
    NativeOIDCMetadata,
    ProviderMetadata,
    fetch_native_oidc_metadata,
    fetch_provider_metadata,
)
from .tokens import TokenResponse, describe_token_error, extract_oauth_error, parse_token_response

AUTH_TYPE_NATIVE_OIDC: Final = "native_oidc"
TOKEN_SCHEMA_VERSION: Final = 2

# Refresh this far ahead of the recorded expiry, absorbing clock skew and the
# round trip to the provider. Kept equal to the freshness buffer callers judge the
# same credential by: any gap would leave a token stale but refused for refresh,
# forcing a needless browser login.
REFRESH_BUFFER_SECONDS: Final = int(CLI_TOKEN_FRESHNESS_BUFFER_HOURS * 3600)

LOCK_ACQUIRE_TIMEOUT_SECONDS: Final = 30.0
LOCK_POLL_INTERVAL_SECONDS: Final = 0.1
LOCK_STALE_AFTER_SECONDS: Final = 60.0

VERIFY_TIMEOUT_SECONDS: Final = 10


class _HttpGet(Protocol):
    def __call__(self, url: str, *, headers: Mapping[str, str], timeout: float) -> requests.Response: ...


def is_native_credential(token_data: Mapping[str, object] | None) -> bool:
    """True for a native OIDC credential.

    A missing `auth_type` means a legacy proxy-minted credential, which stays
    supported -- it is never treated as an unknown format.
    """
    if not token_data:
        return False
    return token_data.get("auth_type") == AUTH_TYPE_NATIVE_OIDC


def build_native_credential(
    *,
    base_url: str,
    metadata: NativeOIDCMetadata,
    token: TokenResponse,
    previous_refresh_token: str | None = None,
    now: float | None = None,
) -> Mapping[str, object]:
    """Build the token-file payload for a native OIDC credential.

    Refresh-token rotation: a newly issued refresh token replaces the stored
    one; when the response validly omits one, the previous token is retained.
    """
    current_time: Final = time.time() if now is None else now
    credential: Final[dict[str, object]] = {  # mutable-ok: the JSON object written verbatim to the token file
        "schema_version": TOKEN_SCHEMA_VERSION,
        "auth_type": AUTH_TYPE_NATIVE_OIDC,
        "base_url": base_url.rstrip("/"),
        "key": token.access_token,
        "issuer": metadata.issuer,
        "client_id": metadata.client_id,
        "scopes": metadata.scopes,
        "token_type": token.token_type,
        "timestamp": current_time,
        "expires_at": token.expires_at,
    }
    refresh_token: Final = token.refresh_token or previous_refresh_token
    if refresh_token:
        credential["refresh_token"] = refresh_token
    return credential


def save_credential(credential: Mapping[str, object]) -> None:
    """Atomically write the credential with owner-only permissions."""
    write_private_json(get_cli_token_file_path(), credential)


def verify_token_with_litellm(
    base_url: str,
    access_token: str,
    *,
    get: _HttpGet = requests.get,
) -> None:
    """Confirm the proxy accepts the access token before it is persisted.

    Uses the same user-accessible `/v1/models` probe the CLI already relies on:
    no proxy-admin permission required. The token is sent only to the configured
    origin, and neither the token nor the response body is logged.
    """
    url: Final = base_url.rstrip("/") + "/v1/models"
    try:
        response: Final = get(
            url,
            headers={"Authorization": f"Bearer {access_token}"},  # mutable-ok: request headers for requests
            timeout=VERIFY_TIMEOUT_SECONDS,
        )
    except requests.RequestException as error:
        raise NativeOIDCError(
            f"could not reach the LiteLLM proxy at {base_url.rstrip('/')} to verify the token: {type(error).__name__}"
        ) from error

    if response.status_code in (401, 403):
        raise NativeOIDCAuthRejected(
            f"LiteLLM rejected the identity provider's access token (HTTP "
            f"{response.status_code}). The provider issued a token that this proxy "
            "does not accept. Check that the proxy trusts the token's issuer and "
            "signing keys, that the audience matches what the proxy expects, and "
            "that the required user/team/role claim mapping is configured."
        )
    if response.status_code == 404:
        raise NativeOIDCError(
            f"{url} returned HTTP 404, so the access token could not be verified; "
            "check that the configured base URL points at a LiteLLM proxy"
        )
    # Every other status is inconclusive rather than a rejection. A 5xx or a 429
    # says nothing about the token, and failing here would send the user back
    # through the whole browser flow over a transient proxy blip.


@contextmanager
def refresh_lock(
    lock_path: str | None = None,
    *,
    timeout: float = LOCK_ACQUIRE_TIMEOUT_SECONDS,
    sleep: Callable[[float], None] = time.sleep,
) -> Generator[None]:
    """Bounded cross-process lock built on O_CREAT|O_EXCL.

    Standard library only, so it adds no dependency and works on every platform
    the CLI supports. A lock older than LOCK_STALE_AFTER_SECONDS is reclaimed so
    a crashed process cannot wedge future logins. The file holds only a pid.
    """
    path: Final = lock_path or (get_cli_token_file_path() + ".lock")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    handle: Final = _acquire_lock_handle(path, timeout=timeout, sleep=sleep)

    try:
        try:
            os.write(handle, str(os.getpid()).encode("ascii"))
        finally:
            os.close(handle)
        yield
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def _acquire_lock_handle(path: str, *, timeout: float, sleep: Callable[[float], None]) -> int:
    deadline: Final = time.monotonic() + timeout
    while True:
        try:
            return os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except OSError as error:
            if error.errno != errno.EEXIST:
                raise NativeOIDCError(f"could not acquire the token refresh lock: {type(error).__name__}") from error
            _reclaim_if_stale(path)
            if time.monotonic() >= deadline:
                raise NativeOIDCError(
                    f"timed out after {int(timeout)}s waiting for another process to finish refreshing the token"
                )
            sleep(LOCK_POLL_INTERVAL_SECONDS)


def _reclaim_if_stale(path: str) -> None:
    try:
        if time.time() - os.stat(path).st_mtime > LOCK_STALE_AFTER_SECONDS:
            os.unlink(path)
    except OSError:
        pass


def native_credential_expires_at(token_data: Mapping[str, object]) -> float | None:
    expires_at: Final = token_data.get("expires_at")
    if isinstance(expires_at, bool) or not isinstance(expires_at, (int, float)):
        return None
    return float(expires_at)


def needs_refresh(token_data: Mapping[str, object], *, now: float | None = None) -> bool:
    """True when a native credential is expired or close enough to warrant refresh."""
    expires_at: Final = native_credential_expires_at(token_data)
    if expires_at is None:
        # Malformed native expiry metadata fails closed.
        return True
    current_time: Final = time.time() if now is None else now
    return current_time >= (expires_at - REFRESH_BUFFER_SECONDS)


def _request_refreshed_token(token_endpoint: str, *, refresh_token: str, client_id: str) -> TokenResponse:
    response: Final = post_form(
        token_endpoint,
        {  # mutable-ok: form body for the token request
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            "client_id": client_id,
        },
    )
    if response.status_code != 200 or response.payload is None:
        if extract_oauth_error(response.payload) == "invalid_grant":
            raise NativeOIDCError("the identity provider revoked or expired this session")
        raise NativeOIDCError(describe_token_error(response.status_code, response.payload))
    return parse_token_response(response.payload)


def refresh_native_credential(
    token_data: Mapping[str, object],
    *,
    verify: Callable[[str, str], None] = verify_token_with_litellm,
    fetch_metadata: Callable[[str], NativeOIDCMetadata] = fetch_native_oidc_metadata,
    fetch_provider: Callable[[str], ProviderMetadata] = fetch_provider_metadata,
) -> Mapping[str, object]:
    """Refresh a native credential, re-validating the whole trust chain.

    Everything is re-derived from the stored, origin-bound `base_url`: the
    advertised issuer and client id must still match the stored credential, the
    provider document is rediscovered and its issuer checked exactly, and the
    token endpoint used is the freshly validated one -- never a value read back
    out of the credential file.
    """
    stored_refresh_token: Final = token_data.get("refresh_token")
    if not isinstance(stored_refresh_token, str) or not stored_refresh_token:
        raise NativeOIDCError("stored credential has no refresh token; run 'lite login'")

    base_url: Final = token_data.get("base_url")
    if not isinstance(base_url, str) or not base_url:
        raise NativeOIDCError("stored credential has no base_url; run 'lite login'")

    with refresh_lock():
        # Another process may have refreshed while we waited for the lock.
        current: Final = load_cli_token()
        if (
            is_native_credential(current)
            and current is not None
            and current.get("base_url") == base_url
            and not needs_refresh(current)
        ):
            return current

        metadata: Final = fetch_metadata(base_url)
        if metadata.issuer != token_data.get("issuer") or metadata.client_id != token_data.get("client_id"):
            raise NativeOIDCError(
                "the proxy now advertises a different OIDC issuer or client id than "
                "the stored credential; run 'lite login' again"
            )

        provider: Final = fetch_provider(metadata.issuer)
        token: Final = _request_refreshed_token(
            provider.require_token_endpoint(),
            refresh_token=stored_refresh_token,
            client_id=metadata.client_id,
        )

        credential: Final = build_native_credential(
            base_url=base_url,
            metadata=metadata,
            token=token,
            previous_refresh_token=stored_refresh_token,
        )
        try:
            verify(base_url, token.access_token)
        except Exception:
            # The provider may rotate refresh tokens, in which case the stored one is
            # already spent and dropping the replacement would strand the credential
            # until an interactive login. Keep it, but backdate the expiry so the
            # unverified access token is never handed out: the next use refreshes again.
            save_credential({**credential, "expires_at": time.time() - 1})
            raise
        save_credential(credential)
        return credential
