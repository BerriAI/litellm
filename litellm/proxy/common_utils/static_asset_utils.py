"""
Helpers for the unauthenticated logo / favicon endpoints (``/get_image`` and
``/get_favicon``). Both read an admin-set environment variable that may be a
local filesystem path or an HTTP URL, fetch the resource, and return the
bytes verbatim to any unauthenticated caller.

Without these helpers:

* a misconfigured / hostile env var like ``UI_LOGO_PATH=/etc/passwd`` lets
  any unauthenticated caller exfiltrate the file (LFI — GHSA-3pcp-536p-ghjc).
* a legitimate-looking ``UI_LOGO_PATH=http://internal-service/branding.png``
  pointing at a private host lets any unauthenticated caller exfiltrate
  whatever that host returns (SSRF — GHSA-pjc9-2hw6-78rr), regardless of
  whether the body is actually an image.
"""

import os
from typing import List, Optional

from litellm._logging import verbose_proxy_logger
from litellm.litellm_core_utils.url_utils import SSRFError, validate_url
from litellm.llms.custom_httpx.http_handler import get_async_httpx_client
from litellm.types.llms.custom_http import httpxSpecialProvider

# Conservative allowlist of image MIME types. Anything else is refused —
# without this, an admin-configured URL whose upstream returns
# ``application/json`` (e.g. cloud metadata, internal API) would still be
# served back to the caller verbatim.
ALLOWED_IMAGE_CONTENT_TYPES = frozenset(
    {
        "image/jpeg",
        "image/jpg",
        "image/png",
        "image/gif",
        "image/svg+xml",
        "image/webp",
        "image/x-icon",
        "image/vnd.microsoft.icon",
    }
)


def resolve_local_asset_path(candidate: str, allowed_roots: List[str]) -> Optional[str]:
    """
    Resolve ``candidate`` and return its absolute path only if it lives
    within one of ``allowed_roots``. Returns None on any miss (caller
    falls back to the bundled default asset).

    Resolution uses ``realpath`` to follow symlinks, so a symlink inside
    ``allowed_roots`` pointing at ``/etc/passwd`` is rejected the same as
    a direct ``/etc/passwd`` config.
    """
    if not candidate:
        return None
    try:
        resolved = os.path.realpath(os.path.expanduser(candidate))
    except (OSError, ValueError):
        return None
    if not os.path.isfile(resolved):
        return None
    for root in allowed_roots:
        if not root:
            continue
        try:
            root_resolved = os.path.realpath(root)
        except (OSError, ValueError):
            continue
        if resolved == root_resolved:
            return resolved
        if resolved.startswith(root_resolved + os.sep):
            return resolved
    return None


async def fetch_validated_image_bytes(
    url: str, *, timeout_s: float = 5.0
) -> Optional[bytes]:
    """
    Fetch ``url`` with SSRF protection (always-on) and Content-Type
    validation. Returns the raw bytes on success, ``None`` on any
    failure (blocked target, non-200, or non-image response).

    The SSRF guard is enforced unconditionally — these endpoints are
    unauthenticated, so the admin-facing ``litellm.user_url_validation``
    toggle does not apply. An admin who opted out of URL validation for
    LLM provider paths should not also expose ``/get_image`` to SSRF.
    """
    if not url:
        return None
    try:
        rewritten_url, host_header = validate_url(url)
    except SSRFError as exc:
        verbose_proxy_logger.warning(
            "Blocked unauthenticated asset fetch — SSRF guard rejected %r: %s",
            url,
            exc,
        )
        return None

    # ``validate_url`` rewrites HTTP URLs to point at a validated IP and
    # returns the original hostname for the Host header. For HTTPS with
    # ssl_verify enabled, it returns the URL unchanged (TLS hostname
    # validation handles DNS rebinding).
    async_client = get_async_httpx_client(
        llm_provider=httpxSpecialProvider.UI,
        params={"timeout": timeout_s},
    )
    try:
        if rewritten_url != url:
            response = await async_client.get(
                rewritten_url, headers={"host": host_header}
            )
        else:
            response = await async_client.get(rewritten_url)
    except Exception as exc:
        verbose_proxy_logger.debug("Asset fetch failed for %r: %s", url, exc)
        return None

    if response.status_code != 200:
        return None

    raw_content_type = (
        response.headers.get("content-type") if hasattr(response, "headers") else None
    )
    if not isinstance(raw_content_type, str):
        # Defensive: if upstream omits Content-Type entirely, treat as
        # non-image. (Also keeps ``Mock`` responses without a configured
        # ``headers`` from blowing up the content-type check.)
        return None
    content_type = raw_content_type.split(";")[0].strip().lower()
    if content_type not in ALLOWED_IMAGE_CONTENT_TYPES:
        verbose_proxy_logger.warning(
            "Asset fetch from %r returned non-image content-type %r — refusing to serve.",
            url,
            content_type,
        )
        return None

    return response.content
