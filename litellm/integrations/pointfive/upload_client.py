"""
Uploads one batch to PointFive through a presigned URL.

The proxy holds no cloud credentials. For every batch it asks the PointFive API for a
single-use presigned URL and PUTs the bytes there, so the same plugin runs unchanged on
AWS, GCP, Azure or on-prem. The server picks the object key, so the proxy never chooses
where its data lands.
"""

import asyncio
from collections.abc import Awaitable, Callable, Mapping
from typing import Final

import httpx
from pydantic import BaseModel, Field, ValidationError

from litellm._logging import verbose_logger
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.types.integrations.pointfive import (
    RETRYABLE_UPLOAD_STATUS_CODES,
    PointFiveUploadFailure,
    PointFiveUploadTarget,
)

UPLOAD_URL_QUERY: Final = (
    "query UploadUrl($kind: UploadKind!, $byteCount: Int!) "
    "{ uploadUrl(kind: $kind, byteCount: $byteCount) { uploadUrl objectKey } }"
)
PING_QUERY: Final = "query IntegrationPing($kind: UploadKind!) { integrationPing(kind: $kind) }"
UPLOAD_KIND: Final = "LITELLM"


class _PresignVariables(BaseModel):
    kind: str = UPLOAD_KIND
    byte_count: int = Field(serialization_alias="byteCount")


class _PresignRequest(BaseModel):
    variables: _PresignVariables
    query: str = UPLOAD_URL_QUERY


class _PingVariables(BaseModel):
    kind: str = UPLOAD_KIND


class _PingRequest(BaseModel):
    variables: _PingVariables = _PingVariables()
    query: str = PING_QUERY


class _TargetPayload(BaseModel):
    upload_url: str = Field(alias="uploadUrl")
    object_key: str = Field(alias="objectKey")


class _DataPayload(BaseModel):
    upload_url: _TargetPayload = Field(alias="uploadUrl")


class _ErrorPayload(BaseModel):
    message: str = ""


class _ErrorsPayload(BaseModel):
    """Just the errors array, so it reads any operation's response, whatever its data shape."""

    errors: tuple[_ErrorPayload, ...] | None = None


class _EnvelopePayload(BaseModel):
    data: _DataPayload | None = None


class PointFiveUploadError(Exception):
    """A batch could not be uploaded and the failure is worth retrying."""


def _failure_for(status_code: int, detail: str) -> PointFiveUploadFailure:
    return PointFiveUploadFailure(detail, retryable=status_code in RETRYABLE_UPLOAD_STATUS_CODES)


def _graphql_error(body: str) -> PointFiveUploadFailure | None:
    """GraphQL reports failures inside a 200, so the body has to be read either way."""
    try:
        envelope: Final = _ErrorsPayload.model_validate_json(body)
    except ValidationError:
        return PointFiveUploadFailure("pointfive api returned an unreadable body", retryable=False)
    if not envelope.errors:
        return None
    joined: Final = ", ".join(error.message for error in envelope.errors)
    return PointFiveUploadFailure(f"pointfive api rejected the request: {joined}", retryable=False)


def _parse_target(body: str) -> PointFiveUploadTarget | PointFiveUploadFailure:
    """Read the presigned target out of a GraphQL response body."""
    error: Final = _graphql_error(body)
    if error is not None:
        return error

    try:
        envelope: Final = _EnvelopePayload.model_validate_json(body)
    except ValidationError:
        return PointFiveUploadFailure("pointfive api returned an unreadable body", retryable=False)
    if envelope.data is None:
        return PointFiveUploadFailure("pointfive api returned no upload url", retryable=False)

    target: Final = envelope.data.upload_url
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
    ) -> None:
        self.api_key: Final = api_key
        self.api_url: Final = api_url
        self.http_client: Final = http_client
        self.max_retries: Final = max_retries
        self.sleep: Final = sleep

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
        body: Final = await self._query(_PingRequest().model_dump(by_alias=True))
        if isinstance(body, PointFiveUploadFailure):
            return body
        return _graphql_error(body)

    async def _presign(self, byte_count: int) -> PointFiveUploadTarget | PointFiveUploadFailure:
        """Ask the PointFive API for a presigned URL sized to this batch."""
        body: Final = await self._query(
            _PresignRequest(variables=_PresignVariables(byte_count=byte_count)).model_dump(by_alias=True)
        )
        if isinstance(body, PointFiveUploadFailure):
            return body
        return _parse_target(body)

    async def _query(self, request: Mapping[str, object]) -> str | PointFiveUploadFailure:
        """POST one GraphQL request to the PointFive API and return its raw body."""
        try:
            response: Final = await self.http_client.post(
                self.api_url,
                json=dict(request),  # mutable-ok: AsyncHTTPHandler.post types json as dict
                headers={  # mutable-ok: AsyncHTTPHandler.post types headers as dict
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
            )
        except httpx.HTTPStatusError as e:
            return _failure_for(e.response.status_code, f"pointfive api returned {e.response.status_code}")
        except Exception as e:  # noqa: BLE001  # a transport fault is worth another attempt
            return PointFiveUploadFailure(f"pointfive api unreachable: {type(e).__name__}", retryable=True)

        if response is None:
            return PointFiveUploadFailure("pointfive api returned no response", retryable=True)
        return response.text

    async def _put(self, target: PointFiveUploadTarget, body: bytes) -> PointFiveUploadFailure | None:
        """PUT the batch to the presigned URL, which carries its own authorization."""
        try:
            await self.http_client.put(
                target.upload_url,
                data=body,
                headers={  # mutable-ok: AsyncHTTPHandler.put types headers as dict
                    "Content-Type": "application/x-ndjson",
                    "Content-Encoding": "gzip",
                },
            )
        except httpx.HTTPStatusError as e:
            return _failure_for(e.response.status_code, f"presigned upload returned {e.response.status_code}")
        except Exception as e:  # noqa: BLE001  # a transport fault is worth another attempt
            return PointFiveUploadFailure(f"presigned upload unreachable: {type(e).__name__}", retryable=True)
        return None
