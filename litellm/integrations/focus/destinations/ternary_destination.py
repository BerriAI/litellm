"""Ternary API destination for FOCUS export: uploads each day's FOCUS CSV via a Ternary-signed GCS URL."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final, Protocol
from urllib.parse import quote, urlparse

from typing_extensions import NotRequired, ReadOnly, TypedDict

from litellm._logging import verbose_logger
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler

from .base import FocusDestination, FocusTimeWindow

if TYPE_CHECKING:
    import httpx

TERNARY_UPLOAD_TIMEOUT_SECONDS: Final = 120.0

_LOOPBACK_HOSTS: Final = frozenset({"localhost", "127.0.0.1", "::1"})


class TernaryUploadUrlBody(TypedDict):
    url: ReadOnly[NotRequired[str]]
    method: ReadOnly[NotRequired[str]]
    headers: ReadOnly[NotRequired[dict[str, str]]]


class _UploadUrlResponse(Protocol):
    def json(self) -> TernaryUploadUrlBody: ...


def _upload_url_body(response: _UploadUrlResponse) -> TernaryUploadUrlBody:
    return response.json()


def _require_secure_base_url(base_url: str) -> None:
    parsed: Final = urlparse(base_url)
    if parsed.scheme == "https":
        return
    if parsed.scheme == "http" and (parsed.hostname or "").lower() in _LOOPBACK_HOSTS:
        return
    raise ValueError(f"base_url must be an HTTPS URL (got {base_url!r}); http is allowed only for loopback")


def _require_gcs_url(url: str) -> None:
    parsed: Final = urlparse(url)
    hostname: Final = (parsed.hostname or "").lower()
    is_gcs: Final = hostname == "storage.googleapis.com" or hostname.endswith(".storage.googleapis.com")
    if parsed.scheme != "https" or not is_gcs:
        raise ValueError(f"Ternary destination: upload URL must be an HTTPS GCS URL (got host {hostname!r})")


class _NoRedirectHTTPHandler(AsyncHTTPHandler):
    """AsyncHTTPHandler that disables redirects on every client it creates (SSRF guard)."""

    def create_client(self, *args: object, **kwargs: object) -> httpx.AsyncClient:
        client: Final = super().create_client(*args, **kwargs)  # pyright: ignore[reportArgumentType]  # passthrough
        client.follow_redirects = False
        return client


class FocusTernaryDestination(FocusDestination):
    """Upload one UTC day of FOCUS CSV per delivery to Ternary through a signed GCS URL."""

    def __init__(
        self,
        *,
        prefix: str,
        config: dict[str, str] | None = None,  # mutable-ok: FocusDestination(config) factory contract
    ) -> None:
        resolved_config: Final = config or {}  # mutable-ok: read-only local; empty fallback for absent config
        api_key: Final = resolved_config.get("api_key")
        connection_id: Final = resolved_config.get("connection_id")
        base_url: Final = resolved_config.get("base_url")
        if not api_key:
            raise ValueError(
                "api_key must be provided for Ternary destination "
                "(set TERNARY_API_KEY env var or pass in destination_config)"
            )
        if not connection_id:
            raise ValueError(
                "connection_id must be provided for Ternary destination "
                "(set TERNARY_CONNECTION_ID env var or pass in destination_config)"
            )
        if "/" in connection_id or ".." in connection_id or any(c.isspace() for c in connection_id):
            raise ValueError(f"connection_id must not contain '/', '..', or whitespace (got {connection_id!r})")
        if not base_url:
            raise ValueError(
                "base_url must be provided for Ternary destination "
                "(set TERNARY_BASE_URL env var or pass in destination_config)"
            )
        _require_secure_base_url(str(base_url))
        self.api_key = api_key
        self.connection_id = connection_id
        self.base_url = str(base_url).rstrip("/")
        self.prefix = prefix

    async def deliver(
        self,
        *,
        content: bytes,
        time_window: FocusTimeWindow,
        filename: str,
    ) -> None:
        """Upload one UTC day: request a signed URL for the window's start day, then PUT the CSV to it."""
        if not content:
            verbose_logger.debug("Ternary destination: empty content, skipping upload")
            return

        day: Final = time_window.start_time.strftime("%Y-%m-%d")
        handler: Final = _NoRedirectHTTPHandler(timeout=TERNARY_UPLOAD_TIMEOUT_SECONDS)
        try:
            body: Final = await self._request_upload_url(handler.client, day)
            url: Final = body.get("url")
            headers: Final = body.get("headers")
            if not url or headers is None:
                raise RuntimeError(f"Ternary destination: upload-url response for {day} is missing 'url' or 'headers'")
            if body.get("method") != "PUT":
                raise RuntimeError(f"Ternary destination: unsupported upload method {body.get('method')!r} for {day}")
            _require_gcs_url(url)
            uploaded: Final = await handler.client.put(
                url, content=content, headers=headers, timeout=TERNARY_UPLOAD_TIMEOUT_SECONDS
            )
            if uploaded.status_code >= 400:
                raise RuntimeError(
                    f"Ternary destination: upload for {day} failed ({uploaded.status_code}): {uploaded.text[:200]}"
                )
        finally:
            await handler.close()

        verbose_logger.debug("Ternary destination: uploaded %d bytes for %s (%s)", len(content), day, filename)

    async def _request_upload_url(self, client: httpx.AsyncClient, day: str) -> TernaryUploadUrlBody:
        endpoint: Final = f"{self.base_url}/external-cost-sources/v1/{quote(self.connection_id, safe='')}/upload-url"
        response: Final = await client.post(
            endpoint,
            headers={"Authorization": f"Bearer {self.api_key}"},  # mutable-ok: request headers for the client
            json={"day": day},  # mutable-ok: JSON request body for the client
            timeout=TERNARY_UPLOAD_TIMEOUT_SECONDS,
        )
        if response.status_code >= 400:
            raise RuntimeError(
                f"Ternary destination: upload-url request for {day} failed ({response.status_code}): "
                f"{response.text[:200]}"
            )
        return _upload_url_body(response)
