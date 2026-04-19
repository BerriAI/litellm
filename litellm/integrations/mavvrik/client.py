"""Mavvrik upload layer — 3-step signed URL upload pattern.

  Step 1 — GET signed URL from Mavvrik API
      GET {api_endpoint}/metrics/agent/ai/{connection_id}/upload-url
          ?name={date_str}&type=metrics
      Header: x-api-key: {api_key}
      Response: { "url": "https://..." }

  Step 2 — Initiate resumable upload (POST to signed URL)
      POST {signed_url}
      Headers: Content-Type: application/gzip, x-goog-resumable: start
      Response 201: Location header = session URI

  Step 3 — Upload CSV payload (PUT to session URI)
      PUT {session_uri}
      Headers: Content-Type: application/gzip
      Body: gzip-compressed CSV bytes
      Response 200/201: upload complete

Additionally implements registration and marker advance:

  Register — POST {api_endpoint}/metrics/agent/ai/{connection_id}
      Body: { "name": instance_id, "version": <litellm version>, "arch": <system arch> }
      Response: { "id": "...", "metricsMarker": <epoch_seconds> }

  Advance marker — PATCH {api_endpoint}/metrics/agent/ai/{connection_id}
      Body: { "metricsMarker": <epoch_seconds> }
      Response: 204 No Content
"""

import asyncio
import gzip
import io
import platform
from datetime import datetime as _dt
from datetime import timezone as _tz
from typing import Dict, Optional

import httpx
import litellm as _litellm

from litellm._logging import verbose_proxy_logger

_MAX_RETRIES = 3
_RETRY_BACKOFF_BASE = 1.0  # seconds; doubles each retry


class MavvrikClient:
    """Upload gzip-compressed CSV spend data to Mavvrik via a signed URL upload."""

    def __init__(self, api_key: str, api_endpoint: str, connection_id: str) -> None:
        self._api_key = api_key
        self._api_endpoint = api_endpoint.rstrip("/")
        self._connection_id = connection_id

    # ------------------------------------------------------------------
    # Read-only public accessors (for backward compatibility)
    # ------------------------------------------------------------------

    @property
    def api_key(self) -> str:
        return self._api_key

    @property
    def api_endpoint(self) -> str:
        return self._api_endpoint

    @property
    def connection_id(self) -> str:
        return self._connection_id

    # ------------------------------------------------------------------
    # URL properties
    # ------------------------------------------------------------------

    @property
    def agent_url(self) -> str:
        return f"{self._api_endpoint}/metrics/agent/ai/{self._connection_id}"

    @property
    def upload_url(self) -> str:
        return f"{self._api_endpoint}/metrics/agent/ai/{self._connection_id}/upload-url"

    @property
    def _auth_headers(self) -> Dict[str, str]:
        return {"Content-Type": "application/json", "x-api-key": self._api_key}

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def upload(self, csv_payload: str, date_str: str) -> None:
        """Upload a CSV string to Mavvrik for the given date.

        Args:
            csv_payload: CSV string (header + rows) from MavvrikTransformer.to_csv().
            date_str:    Date string "YYYY-MM-DD". Re-uploading the same date
                         overwrites the previous upload — idempotent.

        Raises:
            RuntimeError: if any upload step fails after retries.
        """
        if not csv_payload.strip():
            verbose_proxy_logger.debug(
                "Mavvrik uploader: empty payload, skipping upload"
            )
            return

        upload_data = self._compress(csv_payload)

        # Share one client across all three steps to reuse the TCP connection pool.
        # Use a generous overall timeout; each step applies its own timeout via the
        # per-request `timeout` arg so slow steps don't block unrelated ones.
        async with httpx.AsyncClient() as client:
            signed_url = await self._get_signed_url(date_str, client=client)
            session_uri = await self._initiate_resumable_upload(
                signed_url, client=client
            )
            await self._finalize_upload(session_uri, upload_data, client=client)

        verbose_proxy_logger.info(
            "Mavvrik uploader: successfully uploaded %d bytes for date %s",
            len(upload_data),
            date_str,
        )

    # ------------------------------------------------------------------
    # Registration — called once during POST /mavvrik/init
    # ------------------------------------------------------------------

    async def register(self) -> str:
        """POST to Mavvrik agent endpoint and return the initial marker as ISO-8601.

          POST {api_endpoint}/metrics/agent/ai/{connection_id}
          Body: { "name": instance_id, "version": <litellm version>, "arch": <system arch> }
          Response: { "id": "...", "metricsMarker": <epoch_seconds> }
          metricsMarker == 0 → default to first day of current month

        Returns:
            ISO-8601 UTC string for the initial export window start.

        Raises:
            RuntimeError: if the registration call fails.
        """
        headers = self._auth_headers
        body: dict = {
            "name": self._connection_id,
            "version": getattr(_litellm, "__version__", "0.0.0"),
            "arch": platform.machine(),
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(self.agent_url, headers=headers, json=body)

        if resp.status_code != 200:
            raise RuntimeError(
                f"Mavvrik registration failed: {resp.status_code} {resp.text[:200]}"
            )

        response_body = resp.json()
        epoch = response_body.get("metricsMarker", 0)

        if epoch:
            marker_dt = _dt.fromtimestamp(float(epoch), tz=_tz.utc)
        else:
            marker_dt = self._utc_now().replace(
                day=1, hour=0, minute=0, second=0, microsecond=0
            )

        marker_iso = marker_dt.isoformat()
        verbose_proxy_logger.info(
            "Mavvrik register: remote epoch=%s → initial marker %s",
            epoch,
            marker_iso,
        )
        return marker_iso

    # ------------------------------------------------------------------
    # Marker advance — called after each successful upload
    # ------------------------------------------------------------------

    async def advance_marker(self, epoch: int) -> None:
        """PATCH the Mavvrik agent endpoint to advance the export marker.

          PATCH {api_endpoint}/metrics/agent/ai/{connection_id}
          Body: { "metricsMarker": <epoch_seconds> }
          Response: 204 No Content

        Raises:
            RuntimeError: if the PATCH fails.
        """
        headers = self._auth_headers
        body = {"metricsMarker": epoch}

        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.patch(self.agent_url, headers=headers, json=body)

        if resp.status_code not in (200, 204):
            raise RuntimeError(
                f"Mavvrik advance_marker failed: {resp.status_code} {resp.text[:200]}"
            )

        verbose_proxy_logger.info(
            "Mavvrik uploader: marker advanced to epoch %d", epoch
        )

    async def report_error(self, error_message: str) -> None:
        """PATCH the Mavvrik agent endpoint to report an export error.

        Uses the same endpoint as advance_marker but sends errorMessage
        instead of metricsMarker so Mavvrik can track failures on their side.

          PATCH {api_endpoint}/metrics/agent/ai/{connection_id}
          Body: { "errorMessage": <str> }
          Response: 204 No Content

        This is best-effort — failures are logged but not raised.
        """
        headers = self._auth_headers
        body = {
            "errorMessage": error_message[:500]
        }  # truncate to avoid oversized payloads

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.patch(self.agent_url, headers=headers, json=body)

            if resp.status_code not in (200, 204):
                verbose_proxy_logger.warning(
                    "Mavvrik report_error PATCH returned %d: %s",
                    resp.status_code,
                    resp.text[:200],
                )
            else:
                verbose_proxy_logger.debug(
                    "Mavvrik report_error: error reported for connection %s",
                    self.connection_id,
                )
        except Exception as exc:
            verbose_proxy_logger.warning(
                "Mavvrik report_error failed (non-fatal): %s", exc
            )

    # ------------------------------------------------------------------
    # Step 1: Get signed URL from Mavvrik API
    # ------------------------------------------------------------------

    async def _get_signed_url(
        self, date_str: str, client: Optional[httpx.AsyncClient] = None
    ) -> str:
        params = {"name": date_str, "type": "metrics"}
        headers = {"Content-Type": "application/json", "x-api-key": self._api_key}

        last_exc: Exception = RuntimeError("unknown error")

        for attempt in range(_MAX_RETRIES):
            try:
                if client is not None:
                    resp = await client.get(
                        self.upload_url, headers=headers, params=params, timeout=30.0
                    )
                else:
                    async with httpx.AsyncClient(timeout=30.0) as _client:
                        resp = await _client.get(
                            self.upload_url, headers=headers, params=params
                        )

                if resp.status_code == 200:
                    body = resp.json()
                    signed_url = body.get("url")
                    if signed_url:
                        verbose_proxy_logger.debug(
                            "Mavvrik uploader: got signed URL for date %s", date_str
                        )
                        return signed_url
                    raise RuntimeError(
                        f"Mavvrik API response missing 'url' field: {body}"
                    )

                last_exc = RuntimeError(
                    f"Mavvrik signed URL request failed: {resp.status_code} {resp.text[:200]}"
                )
                if resp.status_code < 500:
                    raise last_exc

            except httpx.RequestError as exc:
                last_exc = exc

            wait = _RETRY_BACKOFF_BASE * (2**attempt)
            verbose_proxy_logger.warning(
                "Mavvrik uploader: signed URL attempt %d/%d failed for date %s, "
                "retrying in %.1fs: %s",
                attempt + 1,
                _MAX_RETRIES,
                date_str,
                wait,
                last_exc,
            )
            await asyncio.sleep(wait)

        raise RuntimeError(
            f"Mavvrik signed URL failed after {_MAX_RETRIES} attempts: {last_exc}"
        )

    # ------------------------------------------------------------------
    # Step 2: Initiate resumable upload
    # ------------------------------------------------------------------

    async def _initiate_resumable_upload(
        self, signed_url: str, client: Optional[httpx.AsyncClient] = None
    ) -> str:
        headers = {
            "Content-Type": "application/gzip",
            "x-goog-resumable": "start",
        }
        metadata = b'{"contentEncoding":"gzip","contentDisposition":"attachment"}'

        last_exc: Exception = RuntimeError("unknown error")
        for attempt in range(_MAX_RETRIES):
            try:
                if client is not None:
                    resp = await client.post(
                        signed_url, headers=headers, content=metadata, timeout=30.0
                    )
                else:
                    async with httpx.AsyncClient(timeout=30.0) as _client:
                        resp = await _client.post(
                            signed_url, headers=headers, content=metadata
                        )

                if resp.status_code == 201:
                    session_uri = resp.headers.get("Location")
                    if not session_uri:
                        raise RuntimeError(
                            "Mavvrik initiate upload response missing Location header"
                        )
                    verbose_proxy_logger.debug(
                        "Mavvrik uploader: resumable upload session created"
                    )
                    return session_uri

                last_exc = RuntimeError(
                    f"Mavvrik initiate upload failed: {resp.status_code} {resp.text[:200]}"
                )
                if resp.status_code < 500:
                    raise last_exc

            except httpx.RequestError as exc:
                last_exc = exc

            wait = _RETRY_BACKOFF_BASE * (2**attempt)
            verbose_proxy_logger.warning(
                "Mavvrik uploader: initiate attempt %d/%d failed, retrying in %.1fs: %s",
                attempt + 1,
                _MAX_RETRIES,
                wait,
                last_exc,
            )
            await asyncio.sleep(wait)

        raise RuntimeError(
            f"Mavvrik initiate upload failed after {_MAX_RETRIES} attempts: {last_exc}"
        )

    # ------------------------------------------------------------------
    # Step 3: Finalize upload — PUT gzip bytes to session URI
    # ------------------------------------------------------------------

    async def _finalize_upload(
        self,
        session_uri: str,
        csv_data: bytes,
        client: Optional[httpx.AsyncClient] = None,
    ) -> None:
        headers = {
            "Content-Type": "application/gzip",
            "Content-Encoding": "gzip",
            "x-goog-resumable": "stop",
        }

        last_exc: Exception = RuntimeError("unknown error")
        for attempt in range(_MAX_RETRIES):
            try:
                if client is not None:
                    resp = await client.put(
                        session_uri, headers=headers, content=csv_data, timeout=120.0
                    )
                else:
                    async with httpx.AsyncClient(timeout=120.0) as _client:
                        resp = await _client.put(
                            session_uri, headers=headers, content=csv_data
                        )

                if resp.status_code in (200, 201):
                    verbose_proxy_logger.debug(
                        "Mavvrik uploader: finalize upload OK (%d)", resp.status_code
                    )
                    return

                last_exc = RuntimeError(
                    f"Mavvrik finalize upload failed: {resp.status_code} {resp.text[:200]}"
                )
                if resp.status_code < 500:
                    raise last_exc

            except httpx.RequestError as exc:
                last_exc = exc

            wait = _RETRY_BACKOFF_BASE * (2**attempt)
            verbose_proxy_logger.warning(
                "Mavvrik uploader: finalize attempt %d/%d failed, retrying in %.1fs: %s",
                attempt + 1,
                _MAX_RETRIES,
                wait,
                last_exc,
            )
            await asyncio.sleep(wait)

        raise RuntimeError(
            f"Mavvrik finalize upload failed after {_MAX_RETRIES} attempts: {last_exc}"
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _utc_now() -> _dt:
        """Return current UTC datetime. Extracted for easy test patching."""
        return _dt.now(_tz.utc)

    @staticmethod
    def _compress(text: str) -> bytes:
        """GZIP-compress a UTF-8 string and return the raw bytes."""
        buf = io.BytesIO()
        with gzip.GzipFile(fileobj=buf, mode="wb") as gz:
            gz.write(text.encode("utf-8"))
        return buf.getvalue()
