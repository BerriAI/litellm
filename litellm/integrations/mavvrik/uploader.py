"""Uploader — GCS resumable upload protocol.

Responsibility: receive a CSV string and upload it to GCS. Nothing else.

Upload flow:
  1. Compress   — gzip the CSV string → bytes
  2. Signed URL — GET from Mavvrik API via Client.get_signed_url()
  3. Initiate   — POST to signed URL → GCS session URI (Location header)
  4. Finalize   — PUT gzip bytes to session URI → upload complete

Steps 3 and 4 talk directly to GCS (no Mavvrik auth header).
Step 2 is delegated to Client which owns all Mavvrik API calls.

Transport layer (shared by all GCS steps):
  http_request() from _http.py — single httpx call with retry + exponential
  backoff. Used by _initiate_resumable_upload, _finalize_upload, _put_chunk.

GCS resumable upload protocol reference:
  https://cloud.google.com/storage/docs/resumable-uploads
"""

import gzip
import io
from typing import TYPE_CHECKING, Any, AsyncIterator

from litellm._logging import verbose_proxy_logger
from litellm.integrations.mavvrik._http import http_request

if TYPE_CHECKING:
    from litellm.integrations.mavvrik.client import Client
else:
    Client = Any

# GCS requires intermediate chunks to be exactly this size (256 KB aligned).
# Only the final chunk can be smaller.
_GCS_CHUNK_SIZE = 256 * 1024


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
        """
        metadata = b'{"contentEncoding":"gzip","contentDisposition":"attachment"}'
        resp = await http_request(
            "POST",
            signed_url,
            headers={"Content-Type": "application/gzip", "x-goog-resumable": "start"},
            content=metadata,
            timeout=30.0,
            label="initiate",
        )
        if resp.status_code != 201:
            raise RuntimeError(
                f"GCS initiate upload failed: {resp.status_code} {resp.text[:200]}"
            )
        session_uri = resp.headers.get("Location")
        if not session_uri:
            raise RuntimeError("GCS initiate upload response missing Location header")
        return session_uri

    async def _finalize_upload(self, session_uri: str, gzip_bytes: bytes) -> None:
        """PUT gzip bytes to the GCS session URI to complete the bulk upload."""
        resp = await http_request(
            "PUT",
            session_uri,
            headers={
                "Content-Type": "application/gzip",
                "Content-Encoding": "gzip",
                "x-goog-resumable": "stop",
            },
            content=gzip_bytes,
            timeout=120.0,
            label="finalize",
        )
        if resp.status_code not in (200, 201):
            raise RuntimeError(
                f"GCS finalize upload failed: {resp.status_code} {resp.text[:200]}"
            )
        verbose_proxy_logger.debug("uploader: finalize OK (%d)", resp.status_code)

    async def _put_chunk(
        self,
        session_uri: str,
        chunk: bytes,
        offset: int,
        final: bool,
    ) -> None:
        """PUT one chunk to the GCS resumable session URI.

        Intermediate chunks: Content-Range: bytes X-Y/*  → expect 308
        Final chunk:         Content-Range: bytes X-Y/T  → expect 200/201
        """
        end = offset + len(chunk) - 1
        total_str = str(offset + len(chunk)) if final else "*"
        content_range = f"bytes {offset}-{end}/{total_str}"
        expected = {200, 201} if final else {308}

        resp = await http_request(
            "PUT",
            session_uri,
            headers={
                "Content-Type": "application/gzip",
                "Content-Range": content_range,
            },
            content=chunk,
            timeout=120.0,
            label="chunk",
        )
        if resp.status_code not in expected:
            raise RuntimeError(
                f"GCS PUT chunk failed: {resp.status_code} "
                f"(expected {expected}): {resp.text[:200]}"
            )

    # ------------------------------------------------------------------
    # Streaming upload
    # ------------------------------------------------------------------

    async def _stream_upload(
        self,
        pages: AsyncIterator[str],
        date_str: str,
    ) -> int:
        """Stream CSV pages to GCS using chunked resumable upload.

        Each intermediate chunk is exactly _GCS_CHUNK_SIZE bytes (256 KB aligned).
        The final chunk can be any size. Called exclusively by Orchestrator._export().

        Returns total compressed bytes uploaded (0 if pages is empty).
        """
        gz_buffer = bytearray()
        raw_buf = io.BytesIO()
        gz = gzip.GzipFile(fileobj=raw_buf, mode="wb")
        offset = 0
        session_uri: str = ""
        has_data = False

        async for csv_chunk in pages:
            if not csv_chunk:
                continue

            if not has_data:
                signed_url = await self._client.get_signed_url(date_str)
                session_uri = await self._initiate_resumable_upload(signed_url)
                has_data = True

            gz.write(csv_chunk.encode("utf-8"))
            gz.flush()
            gz_buffer.extend(raw_buf.getvalue())
            raw_buf.seek(0)
            raw_buf.truncate(0)

            while len(gz_buffer) >= _GCS_CHUNK_SIZE:
                chunk = bytes(gz_buffer[:_GCS_CHUNK_SIZE])
                gz_buffer = gz_buffer[_GCS_CHUNK_SIZE:]
                await self._put_chunk(session_uri, chunk, offset=offset, final=False)
                offset += len(chunk)

        if not has_data:
            verbose_proxy_logger.debug("uploader: no data to stream, skipping upload")
            return 0

        gz.close()
        gz_buffer.extend(raw_buf.getvalue())
        total = offset + len(gz_buffer)
        await self._put_chunk(session_uri, bytes(gz_buffer), offset=offset, final=True)

        verbose_proxy_logger.info(
            "uploader: stream upload complete — %d bytes for date %s", total, date_str
        )
        return total

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
