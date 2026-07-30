"""Device Authorization Grant (RFC 8628).

The device code is a bearer secret for the pending authorization: it is
displayed nowhere and logged nowhere. Only the user code and verification URI
are shown.
"""

import time
import webbrowser
from dataclasses import dataclass
from typing import Any, Dict, Optional

import click

from litellm.litellm_core_utils.native_oidc_validation import (
    format_scopes,
    validate_endpoint_url,
)

from .errors import NativeOIDCError
from .http_client import post_form
from .metadata import DEVICE_CODE_GRANT_TYPE, NativeOIDCMetadata, ProviderMetadata
from .tokens import (
    TokenResponse,
    describe_token_error,
    extract_oauth_error,
    parse_token_response,
)

# RFC 8628 section 3.2: `interval` defaults to 5 seconds when omitted.
DEFAULT_POLL_INTERVAL_SECONDS = 5
# RFC 8628 section 3.5: `slow_down` increases the interval by 5 seconds.
SLOW_DOWN_INCREMENT_SECONDS = 5

MAX_POLL_INTERVAL_SECONDS = 60
MAX_DEVICE_CODE_LIFETIME_SECONDS = 30 * 60
CONNECTION_BACKOFF_SECONDS = 5


@dataclass(frozen=True)
class DeviceAuthorization:
    """A validated device authorization response."""

    device_code: str
    user_code: str
    verification_uri: str
    verification_uri_complete: Optional[str]
    expires_in: int
    interval: int


def _require_non_empty_string(raw: Dict[str, Any], key: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value:
        raise NativeOIDCError(f"device authorization response is missing {key}")
    return value


def _optional_safe_uri(raw: Dict[str, Any], key: str) -> Optional[str]:
    value = raw.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise NativeOIDCError(f"device authorization response {key} is not a string")
    try:
        return validate_endpoint_url(value)
    except ValueError as error:
        raise NativeOIDCError(f"device authorization response {key} {error}") from error


def _bounded_positive_int(raw: Dict[str, Any], key: str, maximum: int) -> Optional[int]:
    value = raw.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise NativeOIDCError(f"device authorization response {key} is not an integer")
    if value <= 0 or value > maximum:
        raise NativeOIDCError(f"device authorization response {key} is out of range")
    return value


def parse_device_authorization(payload: Any) -> DeviceAuthorization:
    if not isinstance(payload, dict):
        raise NativeOIDCError("device authorization endpoint did not return a JSON object")

    verification_uri = _optional_safe_uri(payload, "verification_uri")
    if verification_uri is None:
        raise NativeOIDCError("device authorization response is missing verification_uri")

    expires_in = _bounded_positive_int(payload, "expires_in", MAX_DEVICE_CODE_LIFETIME_SECONDS)
    if expires_in is None:
        raise NativeOIDCError("device authorization response is missing expires_in")

    interval = _bounded_positive_int(payload, "interval", MAX_POLL_INTERVAL_SECONDS)

    return DeviceAuthorization(
        device_code=_require_non_empty_string(payload, "device_code"),
        user_code=_require_non_empty_string(payload, "user_code"),
        verification_uri=verification_uri,
        verification_uri_complete=_optional_safe_uri(payload, "verification_uri_complete"),
        expires_in=expires_in,
        interval=interval if interval is not None else DEFAULT_POLL_INTERVAL_SECONDS,
    )


def request_device_authorization(
    device_authorization_endpoint: str, metadata: NativeOIDCMetadata
) -> DeviceAuthorization:
    """Start a device authorization as a public client (no client secret)."""
    response = post_form(
        device_authorization_endpoint,
        {"client_id": metadata.client_id, "scope": format_scopes(metadata.scopes)},
    )
    if response.status_code not in (200, 201) or response.payload is None:
        raise NativeOIDCError(
            describe_token_error(response.status_code, response.payload).replace(
                "token endpoint", "device authorization endpoint"
            )
        )
    return parse_device_authorization(response.payload)


def poll_for_device_token(
    token_endpoint: str,
    authorization: DeviceAuthorization,
    metadata: NativeOIDCMetadata,
    *,
    sleep=time.sleep,
    monotonic=time.monotonic,
) -> TokenResponse:
    """Poll the token endpoint per RFC 8628 section 3.5."""
    interval = authorization.interval
    deadline = monotonic() + authorization.expires_in

    while True:
        if monotonic() >= deadline:
            raise NativeOIDCError("device authorization expired before it was approved")

        sleep(interval)

        try:
            response = post_form(
                token_endpoint,
                {
                    "grant_type": DEVICE_CODE_GRANT_TYPE,
                    "device_code": authorization.device_code,
                    "client_id": metadata.client_id,
                },
            )
        except NativeOIDCError:
            # Transient connection problem: back off rather than busy-looping,
            # and let the device-code deadline end the loop.
            interval = min(interval + CONNECTION_BACKOFF_SECONDS, MAX_POLL_INTERVAL_SECONDS)
            continue

        if response.retry_after is not None:
            interval = min(max(interval, response.retry_after), MAX_POLL_INTERVAL_SECONDS)

        if response.status_code == 200 and response.payload is not None:
            return parse_token_response(response.payload)

        error = extract_oauth_error(response.payload)
        if error == "authorization_pending":
            continue
        if error == "slow_down":
            # Applies to every subsequent poll, not just the next one.
            interval = min(interval + SLOW_DOWN_INCREMENT_SECONDS, MAX_POLL_INTERVAL_SECONDS)
            continue
        if error == "access_denied":
            raise NativeOIDCError("device authorization was denied")
        if error == "expired_token":
            raise NativeOIDCError("device authorization expired before it was approved")
        # Any other OAuth error stops polling.
        raise NativeOIDCError(describe_token_error(response.status_code, response.payload))


def run_device_flow(
    metadata: NativeOIDCMetadata,
    provider: ProviderMetadata,
    *,
    open_browser: bool = True,
    echo=click.echo,
    sleep=time.sleep,
    monotonic=time.monotonic,
) -> TokenResponse:
    """Run the full device authorization flow."""
    provider.assert_device_flow_supported()
    device_endpoint = provider.require_device_authorization_endpoint()
    token_endpoint = provider.require_token_endpoint()

    authorization = request_device_authorization(device_endpoint, metadata)

    echo(f"Open: {authorization.verification_uri}")
    echo(f"Enter code: {authorization.user_code}")
    if authorization.verification_uri_complete:
        echo(f"Or open directly: {authorization.verification_uri_complete}")

    if open_browser:
        target = authorization.verification_uri_complete or authorization.verification_uri
        try:
            webbrowser.open(target)
        except Exception:  # noqa: BLE001 - browser launch is best-effort
            pass

    echo("Waiting for approval...")
    return poll_for_device_token(token_endpoint, authorization, metadata, sleep=sleep, monotonic=monotonic)
