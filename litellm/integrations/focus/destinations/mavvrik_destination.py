"""Mavvrik GCS destination for FOCUS export.

Flow:
  1. GET /metrics/agent/ai/{connection_id}/upload-url → GCS signed URL
  2. PUT <signed_url> with CSV content
  3. PATCH /metrics/agent/ai/{connection_id} → advance metricsMarker
"""

from __future__ import annotations

import gzip
from typing import Any, Optional
from urllib.parse import urlparse

from litellm._logging import verbose_logger
from litellm.llms.custom_httpx.http_handler import (
    AsyncHTTPHandler,
    get_async_httpx_client,
    httpxSpecialProvider,
)

from .base import FocusDestination, FocusTimeWindow

_MAVVRIK_ALLOWED_SUFFIXES = (".mavvrik.dev", ".mavvrik.ai", ".mavvrik.app")

# GCS requires intermediate chunks to be a multiple of 256 KB.
# 8 MB gives a good balance between round-trips and memory pressure.
_GCS_CHUNK_SIZE = 8 * 1024 * 1024  # 8 MB


def _validate_api_endpoint(api_endpoint: str) -> None:
    if not api_endpoint.startswith("https://"):
        raise ValueError("MAVVRIK_API_ENDPOINT must be an HTTPS URL")
    hostname = (urlparse(api_endpoint).hostname or "").lower()
    if not any(hostname.endswith(suffix) for suffix in _MAVVRIK_ALLOWED_SUFFIXES):
        raise ValueError(
            "MAVVRIK_API_ENDPOINT host must be a Mavvrik domain (e.g. https://api.mavvrik.dev/<tenant_id>)"
        )


def _validate_gcs_url(url: str, label: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme != "https":
        raise ValueError(f"Mavvrik FOCUS destination: {label} must be HTTPS, got scheme '{parsed.scheme}'")
    hostname = (parsed.hostname or "").lower()
    if not (hostname == "storage.googleapis.com" or hostname.endswith(".storage.googleapis.com")):
        raise ValueError(
            f"Mavvrik FOCUS destination: {label} must be a GCS endpoint (storage.googleapis.com), got '{hostname}'"
        )


class FocusMavvrikDestination(FocusDestination):
    """Upload FOCUS CSV exports to Mavvrik via GCS signed URL."""

    def __init__(
        self,
        *,
        prefix: str,
        config: Optional[dict[str, Any]] = None,
    ) -> None:
        config = config or {}
        api_key = config.get("api_key")
        api_endpoint = config.get("api_endpoint")
        connection_id = config.get("connection_id")

        if not api_key:
            raise ValueError(
                "MAVVRIK_API_KEY must be provided for Mavvrik FOCUS destination "
                "(set MAVVRIK_API_KEY env var or pass in destination_config)"
            )
        if not api_endpoint:
            raise ValueError(
                "MAVVRIK_API_ENDPOINT must be provided for Mavvrik FOCUS destination "
                "(set MAVVRIK_API_ENDPOINT env var or pass in destination_config)"
            )
        if not connection_id:
            raise ValueError(
                "MAVVRIK_CONNECTION_ID must be provided for Mavvrik FOCUS destination "
                "(set MAVVRIK_CONNECTION_ID env var or pass in destination_config)"
            )

        _validate_api_endpoint(api_endpoint)

        self.api_key = api_key
        self.api_endpoint = api_endpoint.rstrip("/")
        self.connection_id = connection_id
        self.prefix = prefix
        self._http: AsyncHTTPHandler = get_async_httpx_client(llm_provider=httpxSpecialProvider.LoggingCallback)
        self._registered = False

    @property
    def _agent_url(self) -> str:
        return f"{self.api_endpoint}/metrics/agent/ai/{self.connection_id}"

    @property
    def _upload_url_endpoint(self) -> str:
        return f"{self.api_endpoint}/metrics/agent/ai/{self.connection_id}/upload-url"

    @property
    def _auth_headers(self) -> dict[str, str]:
        return {"Content-Type": "application/json", "x-api-key": self.api_key}

    async def _ensure_registered(self) -> Optional[int]:
        """POST agent endpoint to register/initialize the connector (once per instance).

        Returns metricsMarker from the Mavvrik response — the last date index
        Mavvrik has successfully processed. Used by the logger to catch up any
        dates that were missed due to previous export failures.

        Returns None if the connector was already registered (cached).
        """
        if self._registered:
            return None
        resp = await self._http.client.request(
            method="POST",
            url=self._agent_url,
            headers=self._auth_headers,
            json={"name": self.connection_id},
            timeout=30.0,
        )
        if resp.status_code == 410:
            self._registered = False
            raise RuntimeError(
                "Mavvrik FOCUS destination: connector is disconnected (410). "
                "Re-enable the connection in the Mavvrik dashboard."
            )
        if resp.status_code >= 400:
            raise RuntimeError(f"Mavvrik FOCUS destination: register failed ({resp.status_code}): {resp.text[:200]}")
        self._registered = True
        metrics_marker = resp.json().get("metricsMarker", 0)
        verbose_logger.debug(
            "Mavvrik FOCUS destination: connector registered (metricsMarker=%s)",
            metrics_marker,
        )
        return metrics_marker

    async def _get_signed_url(self, date_str: str) -> str:
        """GET upload-url endpoint → GCS signed URL for the given date."""
        params = {"name": date_str, "type": "metrics", "datetime": date_str}
        resp = await self._http.client.request(
            method="GET",
            url=self._upload_url_endpoint,
            headers=self._auth_headers,
            params=params,
            timeout=30.0,
        )
        if resp.status_code >= 400:
            raise RuntimeError(
                f"Mavvrik FOCUS destination: failed to get signed URL ({resp.status_code}): {resp.text[:200]}"
            )
        signed_url = resp.json().get("url")
        if not signed_url:
            raise RuntimeError(f"Mavvrik FOCUS destination: response missing 'url' field: {resp.json()}")
        _validate_gcs_url(signed_url, "signed URL")
        verbose_logger.debug("Mavvrik FOCUS destination: got signed URL for date %s", date_str)
        return signed_url

    async def _upload_to_gcs(self, signed_url: str, content: bytes) -> None:
        """Upload gzip-compressed CSV to GCS via chunked resumable upload.

        The full CSV is gzip-compressed first, then uploaded in _GCS_CHUNK_SIZE
        chunks using the GCS resumable upload protocol. GCS assembles the chunks
        server-side into a single complete object — the bucket receives one file
        regardless of how many chunks were sent.

        Intermediate chunks: Content-Range: bytes X-Y/*   → expect 308
        Final chunk:         Content-Range: bytes X-Y/T   → expect 200/201

        This handles exports larger than available memory for a single PUT while
        keeping the destination code self-contained (no changes to the FOCUS
        pipeline upstream).
        """
        gzip_bytes = gzip.compress(content)
        total = len(gzip_bytes)

        # Step 1: initiate resumable upload session
        metadata = b'{"contentEncoding":"gzip","contentDisposition":"attachment"}'
        init_resp = await self._http.client.request(
            method="POST",
            url=signed_url,
            headers={
                "Content-Type": "application/gzip",
                "x-goog-resumable": "start",
            },
            content=metadata,
            timeout=30.0,
        )
        if init_resp.status_code not in (200, 201):
            raise RuntimeError(
                f"Mavvrik FOCUS destination: GCS session init failed ({init_resp.status_code}): {init_resp.text[:400]}"
            )

        session_uri = init_resp.headers.get("Location")
        if not session_uri:
            raise RuntimeError("Mavvrik FOCUS destination: GCS session init missing Location header")
        _validate_gcs_url(session_uri, "session URI")

        verbose_logger.debug(
            "Mavvrik FOCUS destination: GCS session started, uploading %d gzip bytes in %d chunk(s)",
            total,
            max(1, -(-total // _GCS_CHUNK_SIZE)),  # ceiling division
        )

        # Step 2: upload in chunks; cancel session on any failure to avoid
        # lingering GCS sessions (they stay open for ~1 week otherwise).
        offset = 0
        try:
            while offset < total:
                chunk = gzip_bytes[offset : offset + _GCS_CHUNK_SIZE]
                chunk_end = offset + len(chunk) - 1
                is_final = (offset + len(chunk)) >= total
                content_range = f"bytes {offset}-{chunk_end}/{total}" if is_final else f"bytes {offset}-{chunk_end}/*"
                expected_statuses = {200, 201} if is_final else {308}

                resp = await self._http.client.request(
                    method="PUT",
                    url=session_uri,
                    headers={
                        "Content-Type": "application/gzip",
                        "Content-Range": content_range,
                    },
                    content=chunk,
                    timeout=120.0,
                )
                if resp.status_code not in expected_statuses:
                    raise RuntimeError(
                        f"Mavvrik FOCUS destination: GCS chunk upload failed "
                        f"(chunk offset={offset}, expected={expected_statuses}, "
                        f"got={resp.status_code}): {resp.text[:400]}"
                    )
                offset += len(chunk)
                verbose_logger.debug(
                    "Mavvrik FOCUS destination: uploaded chunk offset=%d/%d",
                    offset,
                    total,
                )
        except Exception:
            # Cancel the open GCS session so it doesn't linger for up to 1 week.
            try:
                await self._http.client.request(method="DELETE", url=session_uri, timeout=10.0)
                verbose_logger.debug("Mavvrik FOCUS destination: cancelled GCS session after error")
            except Exception:
                pass
            raise

    async def _update_metrics_marker(self, date_epoch: int) -> None:
        """PATCH agent endpoint to advance metricsMarker after a successful upload."""
        resp = await self._http.client.request(
            method="PATCH",
            url=self._agent_url,
            headers=self._auth_headers,
            json={"metricsMarker": date_epoch},
            timeout=30.0,
        )
        if resp.status_code == 410:
            self._registered = False
            raise RuntimeError(
                "Mavvrik FOCUS destination: connector is disconnected (410). "
                "Re-enable the connection in the Mavvrik dashboard."
            )
        if resp.status_code >= 400:
            raise RuntimeError(
                f"Mavvrik FOCUS destination: failed to update metricsMarker ({resp.status_code}): {resp.text[:200]}"
            )
        verbose_logger.debug("Mavvrik FOCUS destination: metricsMarker advanced to %s", date_epoch)

    async def get_metrics_marker(self) -> Optional[int]:
        """Register with Mavvrik and return the current metricsMarker.

        Always calls the Mavvrik register API — unlike deliver() which skips
        registration once _registered is True, catch-up requires a fresh
        marker value on every run.
        """
        resp = await self._http.client.request(
            method="POST",
            url=self._agent_url,
            headers=self._auth_headers,
            json={"name": self.connection_id},
            timeout=30.0,
        )
        if resp.status_code == 410:
            self._registered = False
            raise RuntimeError(
                "Mavvrik FOCUS destination: connector is disconnected (410). "
                "Re-enable the connection in the Mavvrik dashboard."
            )
        if resp.status_code >= 400:
            raise RuntimeError(f"Mavvrik FOCUS destination: register failed ({resp.status_code}): {resp.text[:200]}")
        self._registered = True
        metrics_marker = resp.json().get("metricsMarker", 0)
        verbose_logger.debug("Mavvrik FOCUS destination: got metricsMarker=%s", metrics_marker)
        return metrics_marker

    async def deliver(
        self,
        *,
        content: bytes,
        time_window: FocusTimeWindow,
        filename: str,
    ) -> None:
        """Upload FOCUS CSV to Mavvrik via GCS signed URL.

        Uses the start date of the time window as the object date key.
        """
        date_str = time_window.start_time.strftime("%Y-%m-%d")
        date_epoch = int(time_window.start_time.timestamp())

        await self._ensure_registered()

        if not content:
            verbose_logger.debug(
                "Mavvrik FOCUS destination: empty content for date=%s, advancing marker",
                date_str,
            )
            await self._update_metrics_marker(date_epoch)
            return

        verbose_logger.debug(
            "Mavvrik FOCUS destination: uploading %d bytes for date=%s (%s)",
            len(content),
            date_str,
            filename,
        )

        signed_url = await self._get_signed_url(date_str)
        await self._upload_to_gcs(signed_url, content)
        await self._update_metrics_marker(date_epoch)

        verbose_logger.debug("Mavvrik FOCUS destination: upload complete for date=%s", date_str)
