"""Uploader — GCS resumable upload protocol.

Responsibility: receive a CSV string and upload it to GCS. Nothing else.

Upload flow:
  1. Compress   — gzip the CSV string → bytes
  2. Signed URL — GET from Mavvrik API via Client.get_signed_url()
  3. Initiate   — POST to signed URL → GCS session URI (Location header)
  4. Finalize   — PUT gzip bytes to session URI → upload complete

Steps 3 and 4 talk directly to GCS (no Mavvrik auth header).
Step 2 is delegated to Client which owns all Mavvrik API calls.

GCS resumable upload protocol reference:
  https://cloud.google.com/storage/docs/resumable-uploads
"""

import asyncio
import gzip
import io
from typing import TYPE_CHECKING, Any

import httpx

from litellm._logging import verbose_proxy_logger

if TYPE_CHECKING:
    from litellm.integrations.mavvrik.client import Client
else:
    Client = Any

_MAX_RETRIES = 3
_RETRY_BACKOFF_BASE = 1.0  # seconds; doubles each retry


class Uploader:
    """Upload gzip-compressed CSV data to GCS via the resumable upload protocol."""

    def __init__(self, client: "Client") -> None:
        self._client = client

    @property
    def client(self) -> "Client":
        return self._client

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def upload(self, csv_payload: str, date_str: str) -> None:
        """Compress and upload a CSV string to GCS for the given date.

        Re-uploading the same date overwrites the previous object — idempotent.

        Args:
            csv_payload: CSV string (header + rows).
            date_str:    Date in YYYY-MM-DD format.

        Raises:
            RuntimeError: if any upload step fails after retries.
        """
        if not csv_payload.strip():
            verbose_proxy_logger.debug("uploader: empty payload, skipping upload")
            return

        gzip_bytes = self._compress(csv_payload)
        signed_url = await self._client.get_signed_url(date_str)
        session_uri = await self._initiate_resumable_upload(signed_url)
        await self._finalize_upload(session_uri, gzip_bytes)

        verbose_proxy_logger.info(
            "uploader: uploaded %d bytes for date %s", len(gzip_bytes), date_str
        )

    # ------------------------------------------------------------------
    # GCS protocol steps
    # ------------------------------------------------------------------

    async def _initiate_resumable_upload(self, signed_url: str) -> str:
        """POST to the GCS signed URL to open a resumable upload session.

        Returns the session URI from the Location response header.
        Retries on 5xx; raises immediately on 4xx.
        """
        headers = {
            "Content-Type": "application/gzip",
            "x-goog-resumable": "start",
        }
        metadata = b'{"contentEncoding":"gzip","contentDisposition":"attachment"}'
        last_exc: Exception = RuntimeError("unknown error")

        async with httpx.AsyncClient() as http:
            for attempt in range(_MAX_RETRIES):
                try:
                    resp = await http.post(
                        signed_url, headers=headers, content=metadata, timeout=30.0
                    )
                    if resp.status_code == 201:
                        session_uri = resp.headers.get("Location")
                        if not session_uri:
                            raise RuntimeError(
                                "GCS initiate upload response missing Location header"
                            )
                        return session_uri

                    last_exc = RuntimeError(
                        f"GCS initiate upload failed: {resp.status_code} {resp.text[:200]}"
                    )
                    if resp.status_code < 500:
                        raise last_exc

                except httpx.RequestError as exc:
                    last_exc = exc

                if attempt < _MAX_RETRIES - 1:
                    wait = _RETRY_BACKOFF_BASE * (2**attempt)
                    verbose_proxy_logger.warning(
                        "uploader: initiate attempt %d/%d failed, retrying in %.1fs: %s",
                        attempt + 1,
                        _MAX_RETRIES,
                        wait,
                        last_exc,
                    )
                    await asyncio.sleep(wait)

        raise RuntimeError(
            f"GCS initiate upload failed after {_MAX_RETRIES} attempts: {last_exc}"
        )

    async def _finalize_upload(self, session_uri: str, gzip_bytes: bytes) -> None:
        """PUT gzip bytes to the GCS session URI to complete the upload.

        Retries on 5xx; raises immediately on 4xx.
        """
        headers = {
            "Content-Type": "application/gzip",
            "Content-Encoding": "gzip",
            "x-goog-resumable": "stop",
        }
        last_exc: Exception = RuntimeError("unknown error")

        async with httpx.AsyncClient() as http:
            for attempt in range(_MAX_RETRIES):
                try:
                    resp = await http.put(
                        session_uri, headers=headers, content=gzip_bytes, timeout=120.0
                    )
                    if resp.status_code in (200, 201):
                        verbose_proxy_logger.debug(
                            "uploader: finalize OK (%d)", resp.status_code
                        )
                        return

                    last_exc = RuntimeError(
                        f"GCS finalize upload failed: {resp.status_code} {resp.text[:200]}"
                    )
                    if resp.status_code < 500:
                        raise last_exc

                except httpx.RequestError as exc:
                    last_exc = exc

                if attempt < _MAX_RETRIES - 1:
                    wait = _RETRY_BACKOFF_BASE * (2**attempt)
                    verbose_proxy_logger.warning(
                        "uploader: finalize attempt %d/%d failed, retrying in %.1fs: %s",
                        attempt + 1,
                        _MAX_RETRIES,
                        wait,
                        last_exc,
                    )
                    await asyncio.sleep(wait)

        raise RuntimeError(
            f"GCS finalize upload failed after {_MAX_RETRIES} attempts: {last_exc}"
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compress(text: str) -> bytes:
        """GZIP-compress a UTF-8 string and return the raw bytes."""
        buf = io.BytesIO()
        with gzip.GzipFile(fileobj=buf, mode="wb") as gz:
            gz.write(text.encode("utf-8"))
        return buf.getvalue()
