"""Mavvrik API streaming layer.

Implements the 3-step upload pattern used by the k8s-appliance:

  Step 1 — GET signed URL from Mavvrik API
      GET {api_endpoint}/{tenant}/k8s/agent/{instance_id}/upload-url
          ?name={interval}&provider=k8s&type=metrics
      Header: x-api-key: {api_key}
      Response: { "url": "https://storage.googleapis.com/..." }

  Step 2 — Initiate resumable GCS upload (POST to signed URL)
      POST {signed_url}
      Headers: Content-Type: text/csv, x-goog-resumable: start
      Response 201: Location header = session URI

  Step 3 — Upload CSV payload (PUT to session URI)
      PUT {session_uri}
      Headers: Content-Type: text/csv
      Body: raw CSV bytes (UTF-8, no compression)
      Response 200/201: upload complete

The interval name (used as the GCS object name) is the ISO-8601 UTC
timestamp for the start of the export window, matching the k8s-appliance
convention so Mavvrik can partition objects by time.

Additionally implements the registration call (mirroring k8s-appliance):

  Register — POST {api_endpoint}/{tenant}/k8s/agent/{instance_id}
      Header: x-api-key: {api_key}
      Body: { "instanceId": instance_id, "provider": "litellm" }
      Response: { "id": "...", "metricsMarker": <epoch_seconds> }

  metricsMarker is the Unix epoch from which Mavvrik wants LiteLLM to
  start sending cost data.  It is stored as the initial marker in
  LiteLLM_Config so Mavvrik controls the data ingestion window.
  If metricsMarker == 0 or is absent, defaults to the first day of
  the current month (matching k8s-appliance behaviour).

Note: The endpoint path (/k8s/agent/) is a placeholder while the
litellm-specific Mavvrik endpoint is being built. Swap UPLOAD_URL_PATH
and AGENT_BASE_PATH when the real endpoints are ready.
"""

import gzip
import io

import httpx

from litellm._logging import verbose_proxy_logger

# TODO: swap these paths once Mavvrik builds the litellm-specific endpoints
AGENT_BASE_PATH = "/{tenant}/k8s/agent/{instance_id}"
UPLOAD_URL_PATH = "/{tenant}/k8s/agent/{instance_id}/upload-url"

_MAX_RETRIES = 3
_RETRY_BACKOFF_BASE = 1.0  # seconds; doubles each retry


class MavvrikStreamer:
    """Upload NDJSON+gzip cost data to GCS via a Mavvrik-issued signed URL."""

    def __init__(
        self, api_key: str, api_endpoint: str, tenant: str, instance_id: str
    ) -> None:
        self.api_key = api_key
        # Strip trailing slash for consistent URL construction
        self.api_endpoint = api_endpoint.rstrip("/")
        self.tenant = tenant
        self.instance_id = instance_id

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def upload(self, csv_payload: str, interval: str) -> None:
        """Upload a CSV string to GCS for the given interval.

        Args:
            csv_payload: CSV string (header + rows) from MavvrikTransformer.to_csv().
            interval:    ISO-8601 UTC string used as the GCS object name
                         (e.g. "2025-01-15T14:00:00Z").

        Raises:
            Exception: if any upload step fails after retries.
        """
        if not csv_payload.strip():
            verbose_proxy_logger.debug(
                "Mavvrik streamer: empty payload, skipping upload"
            )
            return

        upload_data = self._compress(csv_payload)

        signed_url = self._get_signed_url(interval)
        session_uri = self._initiate_resumable_upload(signed_url)
        self._finalize_upload(session_uri, upload_data)

        verbose_proxy_logger.info(
            "Mavvrik streamer: successfully uploaded %d bytes for interval %s",
            len(upload_data),
            interval,
        )

    # ------------------------------------------------------------------
    # Registration — called once during POST /mavvrik/init
    # ------------------------------------------------------------------

    def register(self) -> str:
        """POST to Mavvrik agent endpoint and return the initial marker as an ISO-8601 string.

        Mirrors the k8s-appliance RegisterCluster payload structure:
          POST {api_endpoint}/{tenant}/k8s/agent/{instance_id}
          Body: {
            "arch": <system arch>,
            "provider": "litellm",
            "hostProvider": "",
            "accountId": "",
            "location": "",
            "version": <litellm version>,
            "name": <instance_id>,
            "prometheusUrl": "",
            "meta": null,
            "invalidPermissions": []
          }
          Response: { "id": "...", "metricsMarker": <epoch_seconds> }
          metricsMarker == 0 → default to first day of current month

        Returns:
            ISO-8601 UTC string for the initial export window start.

        Raises:
            Exception: if the registration call fails.
        """
        import platform
        from datetime import datetime as _dt, timezone as _tz

        import litellm as _litellm

        path = AGENT_BASE_PATH.format(tenant=self.tenant, instance_id=self.instance_id)
        url = f"{self.api_endpoint}{path}"
        headers = {"Content-Type": "application/json", "x-api-key": self.api_key}
        body = {
            "arch": platform.machine(),
            "provider": "k8s",
            "hostProvider": "",
            "accountId": "",
            "location": "",
            "version": getattr(_litellm, "__version__", "0.0.0"),
            "name": self.instance_id,
            "prometheusUrl": f"{self.api_endpoint}/prometheus",
            "meta": {"node": [], "pvs": []},
            "invalidPermissions": [],
        }

        with httpx.Client(timeout=30.0) as client:
            resp = client.post(url, headers=headers, json=body)

        if resp.status_code != 200:
            raise Exception(
                f"Mavvrik registration failed: {resp.status_code} {resp.text}"
            )

        response_body = resp.json()
        epoch = response_body.get("metricsMarker", 0)

        if epoch:
            marker_dt = _dt.fromtimestamp(float(epoch), tz=_tz.utc)
        else:
            # Default: first day of current month (matching k8s-appliance behaviour)
            now = _dt.now(_tz.utc)
            marker_dt = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

        marker_iso = marker_dt.isoformat()
        verbose_proxy_logger.info(
            "Mavvrik register: metricsMarker=%s → initial marker %s",
            epoch,
            marker_iso,
        )
        return marker_iso

    # ------------------------------------------------------------------
    # Marker advance — called after each successful upload
    # ------------------------------------------------------------------

    def advance_marker(self, epoch: int) -> None:
        """PATCH the Mavvrik agent endpoint to advance the metricsMarker.

        Mirrors the k8s-appliance pattern:
          PATCH {api_endpoint}/{tenant}/k8s/agent/{instance_id}
          Body: { "metricsMarker": <epoch_seconds> }
          Response: 204 No Content

        Raises:
            Exception: if the PATCH fails.
        """
        path = AGENT_BASE_PATH.format(tenant=self.tenant, instance_id=self.instance_id)
        url = f"{self.api_endpoint}{path}"
        headers = {"Content-Type": "application/json", "x-api-key": self.api_key}
        body = {"metricsMarker": epoch}

        with httpx.Client(timeout=30.0) as client:
            resp = client.patch(url, headers=headers, json=body)

        if resp.status_code not in (200, 204):
            raise Exception(
                f"Mavvrik advance_marker failed: {resp.status_code} {resp.text}"
            )

        verbose_proxy_logger.info(
            "Mavvrik streamer: metricsMarker advanced to epoch %d", epoch
        )

    # ------------------------------------------------------------------
    # Step 1: Get signed URL from Mavvrik API
    # ------------------------------------------------------------------

    def _get_signed_url(self, interval: str) -> str:
        path = UPLOAD_URL_PATH.format(tenant=self.tenant, instance_id=self.instance_id)
        url = f"{self.api_endpoint}{path}"
        params = {"name": interval, "provider": "k8s", "type": "metrics"}
        headers = {"Content-Type": "application/json", "x-api-key": self.api_key}

        last_exc: Exception = Exception("unknown error")
        import time

        for attempt in range(_MAX_RETRIES):
            try:
                with httpx.Client(timeout=30.0) as client:
                    resp = client.get(url, headers=headers, params=params)

                if resp.status_code == 200:
                    body = resp.json()
                    signed_url = body.get("url")
                    if signed_url:
                        verbose_proxy_logger.debug(
                            "Mavvrik streamer: got signed URL for interval %s", interval
                        )
                        return signed_url
                    raise Exception(f"Mavvrik API response missing 'url' field: {body}")

                last_exc = Exception(
                    f"Mavvrik signed URL request failed: {resp.status_code} {resp.text}"
                )
                if resp.status_code < 500:
                    # 4xx errors won't improve with retries
                    raise last_exc

            except httpx.RequestError as exc:
                last_exc = exc

            wait = _RETRY_BACKOFF_BASE * (2**attempt)
            verbose_proxy_logger.warning(
                "Mavvrik streamer: signed URL attempt %d/%d failed, retrying in %.1fs: %s",
                attempt + 1,
                _MAX_RETRIES,
                wait,
                last_exc,
            )
            time.sleep(wait)

        raise Exception(
            f"Mavvrik signed URL failed after {_MAX_RETRIES} attempts: {last_exc}"
        )

    # ------------------------------------------------------------------
    # Step 2: Initiate resumable GCS upload
    # ------------------------------------------------------------------

    def _initiate_resumable_upload(self, signed_url: str) -> str:
        headers = {
            "Content-Type": "application/gzip",
            "x-goog-resumable": "start",
        }
        metadata = b'{"contentEncoding":"gzip","contentDisposition":"attachment"}'

        with httpx.Client(timeout=30.0) as client:
            resp = client.post(signed_url, headers=headers, content=metadata)

        if resp.status_code != 201:
            raise Exception(
                f"Mavvrik GCS initiate upload failed: {resp.status_code} {resp.text}"
            )

        session_uri = resp.headers.get("Location")
        if not session_uri:
            raise Exception("Mavvrik GCS initiate response missing Location header")

        verbose_proxy_logger.debug("Mavvrik streamer: resumable upload session created")
        return session_uri

    # ------------------------------------------------------------------
    # Step 3: Finalize upload — PUT gzip bytes to session URI
    # ------------------------------------------------------------------

    def _finalize_upload(self, session_uri: str, csv_data: bytes) -> None:
        headers = {
            "Content-Type": "application/gzip",
            "Content-Encoding": "gzip",
            "x-goog-resumable": "stop",
        }

        with httpx.Client(timeout=120.0) as client:
            resp = client.put(session_uri, headers=headers, content=csv_data)

        if resp.status_code not in (200, 201):
            raise Exception(
                f"Mavvrik GCS finalize upload failed: {resp.status_code} {resp.text}"
            )

        verbose_proxy_logger.debug(
            "Mavvrik streamer: GCS finalize OK (%d)", resp.status_code
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
