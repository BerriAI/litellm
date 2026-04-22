"""Mavvrik API client — all HTTP calls to the Mavvrik REST API.

Responsibility: talk to the Mavvrik API and nothing else.

Public methods (one per API endpoint):
  register()         POST  /metrics/agent/ai/{id}         → Optional[str] ISO marker
  advance_marker()   PATCH /metrics/agent/ai/{id}         → None
  report_error()     PATCH /metrics/agent/ai/{id}         → None (best-effort)
  get_signed_url()   GET   /metrics/agent/ai/{id}/upload-url → str

Transport layer (shared by all four methods):
  _request()    — single httpx call with retry + exponential backoff
  _assert_ok()  — raises RuntimeError on unexpected status; fast-fails on 4xx
"""

import asyncio
import platform
from datetime import datetime as _dt
from datetime import timezone as _tz
from typing import Any, Dict, List, Optional, Union

import httpx
import litellm as _litellm

from litellm._logging import verbose_proxy_logger

_MAX_RETRIES = 3
_RETRY_BACKOFF_BASE = 1.0  # seconds; doubles each retry


class Client:
    """HTTP client for the Mavvrik REST API."""

    def __init__(self, api_key: str, api_endpoint: str, connection_id: str) -> None:
        self._api_key = api_key
        self._api_endpoint = api_endpoint.rstrip("/")
        self._connection_id = connection_id

    # ------------------------------------------------------------------
    # Read-only properties
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
    # Public API methods
    # ------------------------------------------------------------------

    async def register(self) -> Optional[str]:
        """POST agent endpoint → return current metricsMarker as ISO-8601 string.

        Returns None when the remote marker is absent or zero (first run).
        Raises RuntimeError if the call fails.
        """
        body: dict = {
            "name": self._connection_id,
            "version": getattr(_litellm, "__version__", "0.0.0"),
            "arch": platform.machine(),
        }
        resp = await self._request(
            "POST", self.agent_url, headers=self._auth_headers, json=body
        )
        self._assert_ok(resp, expected={200})

        epoch = resp.json().get("metricsMarker", 0)
        if not epoch:
            verbose_proxy_logger.info(
                "register: no marker (first run), epoch=%s", epoch
            )
            return None

        marker_iso = _dt.fromtimestamp(float(epoch), tz=_tz.utc).isoformat()
        verbose_proxy_logger.info("register: epoch=%s → marker %s", epoch, marker_iso)
        return marker_iso

    async def advance_marker(self, epoch: int) -> None:
        """PATCH agent endpoint to advance the export cursor to the given epoch.

        Raises RuntimeError if the call fails.
        """
        resp = await self._request(
            "PATCH",
            self.agent_url,
            headers=self._auth_headers,
            json={"metricsMarker": epoch},
        )
        self._assert_ok(resp, expected={200, 204})
        verbose_proxy_logger.info("client: marker advanced to epoch %d", epoch)

    async def report_error(self, error_message: str) -> None:
        """PATCH agent endpoint to report an export failure to Mavvrik.

        Best-effort: exceptions are logged and swallowed so a reporting failure
        never masks the original error.
        """
        try:
            resp = await self._request(
                "PATCH",
                self.agent_url,
                headers=self._auth_headers,
                json={"error": error_message[:500]},
            )
            if resp.status_code in (200, 204):
                verbose_proxy_logger.debug(
                    "report_error: reported for connection %s", self._connection_id
                )
            else:
                verbose_proxy_logger.warning(
                    "report_error: unexpected status %d", resp.status_code
                )
        except Exception as exc:
            verbose_proxy_logger.warning("report_error failed (non-fatal): %s", exc)

    async def get_signed_url(self, date_str: str) -> str:
        """GET upload-url endpoint → return the GCS signed URL for the given date.

        Raises RuntimeError if the call fails or the response is missing the URL.
        """
        params = {"name": date_str, "type": "metrics"}
        resp = await self._request(
            "GET", self.upload_url, headers=self._auth_headers, params=params
        )
        self._assert_ok(resp, expected={200})

        signed_url = resp.json().get("url")
        if not signed_url:
            raise RuntimeError(
                f"Mavvrik API response missing 'url' field: {resp.json()}"
            )

        verbose_proxy_logger.debug("client: got signed URL for date %s", date_str)
        return signed_url

    # ------------------------------------------------------------------
    # Transport layer
    # ------------------------------------------------------------------

    async def _request(
        self,
        method: str,
        url: str,
        *,
        headers: Optional[Dict[str, str]] = None,
        json: Optional[Any] = None,
        params: Optional[Dict[str, str]] = None,
        content: Optional[bytes] = None,
        timeout: float = 30.0,
    ) -> httpx.Response:
        """Execute an HTTP request with retry and exponential backoff.

        Retries up to _MAX_RETRIES times on 5xx responses and network errors.
        4xx responses are not retried — they indicate a client-side problem.

        Returns the httpx.Response. Callers use _assert_ok() to check the status.
        """
        last_exc: Exception = RuntimeError("unknown error")

        async with httpx.AsyncClient() as http:
            for attempt in range(_MAX_RETRIES):
                try:
                    resp = await http.request(
                        method,
                        url,
                        headers=headers,
                        json=json,
                        params=params,
                        content=content,
                        timeout=timeout,
                    )

                    if resp.status_code < 500:
                        return resp

                    last_exc = RuntimeError(
                        f"{method} {url} → {resp.status_code} {resp.text[:200]}"
                    )

                except httpx.RequestError as exc:
                    last_exc = exc

                if attempt < _MAX_RETRIES - 1:
                    wait = _RETRY_BACKOFF_BASE * (2**attempt)
                    verbose_proxy_logger.warning(
                        "client: %s %s attempt %d/%d failed, retrying in %.1fs: %s",
                        method,
                        url,
                        attempt + 1,
                        _MAX_RETRIES,
                        wait,
                        last_exc,
                    )
                    await asyncio.sleep(wait)

        raise RuntimeError(
            f"client: {method} {url} failed after {_MAX_RETRIES} attempts: {last_exc}"
        )

    @staticmethod
    def _assert_ok(
        resp: httpx.Response,
        expected: Union[set, List[int]],
    ) -> None:
        """Raise RuntimeError when the response status is not in expected.

        Called immediately after _request() by each public method.
        """
        if resp.status_code not in expected:
            raise RuntimeError(
                f"unexpected status {resp.status_code}: {resp.text[:200]}"
            )
