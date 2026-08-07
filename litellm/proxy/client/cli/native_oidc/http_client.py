"""Bounded, redirect-refusing JSON HTTP helpers for the native OIDC flows.

Deliberately small and stdlib+`requests` only so it stays importable from the
thin ``litellm[cli]`` installation.

Security properties enforced here:

- redirects are never followed (a 3xx is an error), so a token or an
  authorization code can never be forwarded cross-origin or downgraded to
  plaintext HTTP
- responses are read with a hard byte ceiling
- explicit connect/read timeouts
- TLS verification is always on
- raw response bodies never appear in raised errors
"""

import json
from collections.abc import Mapping
from dataclasses import dataclass
from functools import reduce
from typing import Final

import requests

from .errors import NativeOIDCError

# Generous enough for real provider documents (some IdPs publish large
# `claims_supported` lists) while still bounding memory.
MAX_RESPONSE_BYTES: Final = 256 * 1024

# (connect, read) seconds.
DEFAULT_TIMEOUT: Final[tuple[float, float]] = (5.0, 15.0)

MAX_RETRY_AFTER_SECONDS: Final = 300


@dataclass(frozen=True)
class JsonResponse:
    """A bounded, parsed HTTP response.

    `payload` is None when the body was not a JSON object. OAuth error
    responses legitimately arrive with a 4xx status and a JSON body, so callers
    inspect `status_code` themselves rather than having it raised for them.
    """

    status_code: int
    payload: Mapping[str, object] | None
    retry_after: int | None


def _is_json_content_type(content_type: str) -> bool:
    media_type: Final = content_type.split(";", 1)[0].strip().lower()
    return media_type == "application/json" or media_type.endswith("+json")


def _parse_retry_after(value: str | None) -> int | None:
    """Parse a bounded delta-seconds Retry-After. HTTP-date form is ignored."""
    if not value:
        return None
    try:
        seconds: Final = int(value.strip())
    except ValueError:
        return None
    if seconds < 0:
        return None
    return min(seconds, MAX_RETRY_AFTER_SECONDS)


def _append_bounded(body: bytes, chunk: bytes) -> bytes:
    combined: Final = body + chunk
    if len(combined) > MAX_RESPONSE_BYTES:
        raise NativeOIDCError(f"response exceeded the {MAX_RESPONSE_BYTES} byte limit")
    return combined


def _read_bounded_body(response: requests.Response) -> bytes:
    return reduce(_append_bounded, response.iter_content(chunk_size=8192), b"")


def _decode_json_object(body: bytes, content_type: str) -> Mapping[str, object] | None:
    """Strictly decode a JSON object, or return None.

    `json.loads` rejects trailing data after the top-level value, so a response
    such as `{"a":1} {"b":2}` is refused rather than silently truncated.
    """
    if not _is_json_content_type(content_type):
        return None
    try:
        decoded: Final = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        return None
    return decoded if isinstance(decoded, dict) else None


def _request(
    method: str,
    url: str,
    *,
    data: Mapping[str, str] | None = None,
    headers: Mapping[str, str] | None = None,
    timeout: tuple[float, float] = DEFAULT_TIMEOUT,
) -> JsonResponse:
    request_headers: Final = {"Accept": "application/json", **(headers or {})}  # mutable-ok: header dict for requests

    try:
        with requests.request(
            method,
            url,
            data=data,
            headers=request_headers,
            timeout=timeout,
            allow_redirects=False,
            stream=True,
        ) as response:
            if 300 <= response.status_code < 400:
                # Refusing rather than following: a redirect here could move
                # credentials to another origin or downgrade to plaintext HTTP.
                raise NativeOIDCError(f"{url} returned HTTP {response.status_code}; redirects are not followed")
            body: Final = _read_bounded_body(response)
            return JsonResponse(
                status_code=response.status_code,
                payload=_decode_json_object(body, response.headers.get("content-type", "")),
                retry_after=_parse_retry_after(response.headers.get("retry-after")),
            )
    except requests.RequestException as error:
        raise NativeOIDCError(f"could not reach {url}: {type(error).__name__}") from error


def get_json_response(url: str, *, timeout: tuple[float, float] = DEFAULT_TIMEOUT) -> JsonResponse:
    """GET and return the bounded parsed response without asserting the status.

    Used where the status code itself is meaningful -- e.g. an older proxy
    answering 404/405 for the discovery route.
    """
    return _request("GET", url, timeout=timeout)


def get_json(url: str, *, timeout: tuple[float, float] = DEFAULT_TIMEOUT) -> Mapping[str, object]:
    """GET a JSON object, raising unless the response is 200 with a JSON object."""
    response: Final = _request("GET", url, timeout=timeout)
    if response.status_code != 200:
        raise NativeOIDCError(f"{url} returned HTTP {response.status_code}")
    if response.payload is None:
        raise NativeOIDCError(f"{url} did not return a JSON object")
    return response.payload


def post_form(
    url: str,
    data: Mapping[str, str],
    *,
    timeout: tuple[float, float] = DEFAULT_TIMEOUT,
) -> JsonResponse:
    """POST form-encoded parameters and return the bounded parsed response."""
    return _request(
        "POST",
        url,
        data=data,
        headers={"Content-Type": "application/x-www-form-urlencoded"},  # mutable-ok: request headers
        timeout=timeout,
    )
