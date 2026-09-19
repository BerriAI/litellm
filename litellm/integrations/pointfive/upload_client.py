"""
Uploads one batch to PointFive through a presigned URL.

The proxy holds no cloud credentials. For every batch it asks the PointFive API for a
single-use presigned URL and PUTs the bytes there, so the same plugin runs unchanged on
AWS, GCP, Azure or on-prem. The server picks the object key, so the proxy never chooses
where its data lands.
"""

import asyncio
from collections.abc import Awaitable, Callable
from types import MappingProxyType
from typing import Final

import httpx
from pydantic import BaseModel, Field, ValidationError

import litellm
from litellm._logging import verbose_logger
from litellm.litellm_core_utils.url_utils import SSRFError, validate_url
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.types.integrations.pointfive import (
    RETRYABLE_UPLOAD_STATUS_CODES,
    PointFiveUploadFailure,
    PointFiveUploadTarget,
)

UPLOAD_KIND: Final = "LITELLM"
UPLOAD_URL_PATH: Final = "/upload-url"
PING_PATH: Final = "/ping"
PUT_HEADERS: Final = MappingProxyType({"Content-Type": "application/x-ndjson", "Content-Encoding": "gzip"})


class _PresignRequest(BaseModel):
    kind: str = UPLOAD_KIND
    byte_count: int = Field(serialization_alias="byteCount")


class _PingRequest(BaseModel):
    kind: str = UPLOAD_KIND


class _TargetPayload(BaseModel):
    upload_url: str = Field(alias="uploadUrl")
    object_key: str = Field(alias="objectKey")


class _ErrorPayload(BaseModel):
    error: str = ""


class PointFiveUploadError(Exception):
    """A batch could not be uploaded and the failure is worth retrying."""


def _failure_for(response: httpx.Response, what: str) -> PointFiveUploadFailure:
    detail: Final = f"{what} returned {response.status_code}"
    reason: Final = _refusal_reason(response.text)
    return PointFiveUploadFailure(
        f"{detail}, {reason}" if reason else detail,
        retryable=response.status_code in RETRYABLE_UPLOAD_STATUS_CODES,
    )


def _refusal_reason(body: str) -> str:
    try:
        return _ErrorPayload.model_validate_json(body).error
    except ValidationError:
        return ""


def _parse_target(body: str) -> PointFiveUploadTarget | PointFiveUploadFailure:
    try:
        target: Final = _TargetPayload.model_validate_json(body)
    except ValidationError:
        return PointFiveUploadFailure("pointfive api returned an unreadable body", retryable=False)
    return PointFiveUploadTarget(upload_url=target.upload_url, object_key=target.object_key)


class PointFiveUploadClient:
    """Presigns and uploads one batch at a time."""

    def __init__(
        self,
        api_key: str,
        api_url: str,
        http_client: AsyncHTTPHandler,
        max_retries: int,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        validate_upload_url: Callable[[str], tuple[str, str]] = validate_url,
    ) -> None:
        self.api_key: Final = api_key
        self.api_url: Final = api_url.rstrip("/")
        self.http_client: Final = http_client
        self.max_retries: Final = max_retries
        self.sleep: Final = sleep
        self.validate_upload_url: Final = validate_upload_url

    async def upload(self, body: bytes) -> str | PointFiveUploadFailure:
        """
        Upload one gzipped batch, returning the object key it landed at.

        Every attempt presigns again, so a retry never reuses a URL that has expired or
        has already been consumed.
        """
        for attempt in range(self.max_retries):
            match await self._upload_once(body):
                case PointFiveUploadFailure(retryable=True) as failure:
                    if attempt + 1 >= self.max_retries:
                        return PointFiveUploadFailure(
                            f"{failure.detail}, gave up after {self.max_retries} attempts", retryable=True
                        )
                    await self.sleep(float(1 << attempt))
                case outcome:
                    return outcome
        return PointFiveUploadFailure("max_upload_retries must be at least 1", retryable=False)

    async def _upload_once(self, body: bytes) -> str | PointFiveUploadFailure:
        target: Final = await self._presign(len(body))
        if isinstance(target, PointFiveUploadFailure):
            return target

        rejection: Final = await self._put(target, body)
        if rejection is not None:
            return rejection

        verbose_logger.debug("pointfive: uploaded %s gzipped bytes to %s", len(body), target.object_key)
        return target.object_key

    async def ping(self) -> PointFiveUploadFailure | None:
        """Report that the proxy is alive when it has nothing to upload."""
        body: Final = await self._post(PING_PATH, _PingRequest())
        if isinstance(body, PointFiveUploadFailure):
            return body
        return None

    async def _presign(self, byte_count: int) -> PointFiveUploadTarget | PointFiveUploadFailure:
        """Ask the PointFive API for a presigned URL sized to this batch."""
        body: Final = await self._post(UPLOAD_URL_PATH, _PresignRequest(byte_count=byte_count))
        if isinstance(body, PointFiveUploadFailure):
            return body
        return _parse_target(body)

    async def _post(self, path: str, request: BaseModel) -> str | PointFiveUploadFailure:
        """POST one JSON request to the PointFive ingestion API and return its raw body."""
        try:
            response: Final = await self.http_client.post(
                self.api_url + path,
                json=request.model_dump(by_alias=True),
                headers={  # mutable-ok: AsyncHTTPHandler.post types headers as dict
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
            )
        except httpx.HTTPStatusError as e:
            return _failure_for(e.response, "pointfive api")
        except Exception as e:  # noqa: BLE001  # a transport fault is worth another attempt
            return PointFiveUploadFailure(f"pointfive api unreachable: {type(e).__name__}", retryable=True)
        return response.text

    async def _put(self, target: PointFiveUploadTarget, body: bytes) -> PointFiveUploadFailure | None:
        """
        PUT the batch to the presigned URL, which carries its own authorization.

        The server chose that URL, so it is treated like any other externally supplied
        destination: the host is checked against blocked networks before connecting, and
        a redirect is refused rather than followed. A presigned URL never legitimately
        redirects, and following one would let a compromised endpoint point the proxy at
        an internal service.
        """
        destination: Final = self._destination(target.upload_url)
        if isinstance(destination, PointFiveUploadFailure):
            return destination
        url, host = destination
        headers: Final = dict(PUT_HEADERS, Host=host) if host else dict(PUT_HEADERS)  # mutable-ok: put wants dict
        try:
            await self.http_client.put(url, data=body, headers=headers, follow_redirects=False)
        except httpx.HTTPStatusError as e:
            if e.response.is_redirect:
                return PointFiveUploadFailure(
                    f"presigned upload redirected with {e.response.status_code}, refusing to follow", retryable=False
                )
            return _failure_for(e.response, "presigned upload")
        except Exception as e:  # noqa: BLE001  # a transport fault is worth another attempt
            return PointFiveUploadFailure(f"presigned upload unreachable: {type(e).__name__}", retryable=True)
        return None

    def _destination(self, upload_url: str) -> tuple[str, str | None] | PointFiveUploadFailure:
        if not getattr(litellm, "user_url_validation", True):
            return upload_url, None
        try:
            return self.validate_upload_url(upload_url)
        except SSRFError as e:
            return PointFiveUploadFailure(f"presigned upload url refused: {e}", retryable=False)
