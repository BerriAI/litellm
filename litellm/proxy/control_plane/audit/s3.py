"""Append-only audit objects on S3: one NDJSON object per delivered batch, written once with ``If-None-Match: *`` and,
when configured, held under Object Lock so nobody can shorten or delete it inside the retention period."""

import base64
import hashlib
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import timedelta
from typing import TYPE_CHECKING, Final, Protocol
from urllib.parse import quote

import httpx

from litellm.litellm_core_utils.aws_partition import get_aws_dns_suffix
from litellm.llms.bedrock.base_aws_llm import BaseAWSLLM, run_aws_signing
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.proxy.control_plane.audit.pipeline import Batch, Delivered, DeliveryResult, Undelivered
from litellm.proxy.control_plane.audit.sinks import ocsf_line
from litellm.proxy.control_plane.settings import S3SinkSettings

if TYPE_CHECKING:
    from botocore.credentials import Credentials

CONTENT_TYPE: Final = "application/x-ndjson"
RETAIN_UNTIL_FORMAT: Final = "%Y-%m-%dT%H:%M:%SZ"
_ERROR_CODE: Final = re.compile(r"<Code>([^<]+)</Code>")


class ObjectStore(Protocol):
    """The one call the sink makes. Raises only httpx errors and never retries on its own."""

    async def put(self, url: str, content: bytes, headers: Mapping[str, str], timeout: float) -> httpx.Response: ...


class Signer(Protocol):
    def __call__(self, url: str, content: bytes, headers: Mapping[str, str]) -> Mapping[str, str]: ...


@dataclass(frozen=True, slots=True)
class HttpxObjectStore:
    """Holds the handler, not its client: the shared cache may close and replace the client between batches."""

    handler: AsyncHTTPHandler

    async def put(self, url: str, content: bytes, headers: Mapping[str, str], timeout: float) -> httpx.Response:
        return await self.handler.client.put(url, content=content, headers=dict(headers), timeout=timeout)


@dataclass(frozen=True, slots=True)
class SigV4Signer:
    """SigV4 over the usual credential chain (env, profile, IRSA, IMDS), the same way the proxy's s3 logger signs.
    Runs in the signing executor because the credential chain may block on STS or IMDS."""

    region: str
    aws: BaseAWSLLM

    def __call__(self, url: str, content: bytes, headers: Mapping[str, str]) -> Mapping[str, str]:
        from botocore.auth import S3SigV4Auth
        from botocore.awsrequest import AWSRequest
        from botocore.credentials import RefreshableCredentials

        credentials: Final[Credentials] = self.aws.get_credentials(aws_region_name=self.region)
        frozen: Final = (
            credentials.get_frozen_credentials() if isinstance(credentials, RefreshableCredentials) else credentials
        )
        request: Final = AWSRequest(method="PUT", url=url, data=content, headers=dict(headers))
        S3SigV4Auth(frozen, "s3", self.region).add_auth(request)
        return dict(request.headers.items())


def ndjson(batch: Batch) -> bytes:
    return "".join(f"{ocsf_line(record)}\n" for record in batch).encode()


def object_key(prefix: str, batch: Batch) -> str:
    """Dated, ordered by sequence, suffixed with the last hash so an identical retry hits the same key."""
    first: Final = batch[0]
    last: Final = batch[-1]
    return f"{prefix}/{first.event.at:%Y/%m/%d/%H}/{first.sequence:012d}-{last.sequence:012d}-{last.hash[-12:]}.ndjson"


def object_url(settings: S3SinkSettings, key: str) -> str:
    if settings.endpoint_url is not None:
        return f"{settings.endpoint_url.rstrip('/')}/{settings.bucket}/{quote(key)}"
    suffix: Final = get_aws_dns_suffix(settings.region)
    host: Final = f"s3.{settings.region}.{suffix}" if settings.region else f"s3.{suffix}"
    return f"https://{settings.bucket}.{host}/{quote(key)}"


def put_headers(settings: S3SinkSettings, batch: Batch, content: bytes) -> Mapping[str, str]:
    """Integrity headers, the never-overwrite condition and, when a lock mode is set, retention counted from the
    newest event in the batch so the clock in the record is the clock of the lock."""
    base: Final = {
        "Content-Type": CONTENT_TYPE,
        "Content-MD5": base64.b64encode(hashlib.md5(content, usedforsecurity=False).digest()).decode(),
        "x-amz-content-sha256": hashlib.sha256(content).hexdigest(),
        "If-None-Match": "*",
    }
    if settings.object_lock_mode is None:
        return base
    retain_until: Final = max(record.event.at for record in batch) + timedelta(days=settings.retention_days)
    return {
        **base,
        "x-amz-object-lock-mode": settings.object_lock_mode,
        "x-amz-object-lock-retain-until-date": retain_until.strftime(RETAIN_UNTIL_FORMAT),
    }


def _error_code(response: httpx.Response) -> str:
    found: Final = _ERROR_CODE.search(response.text)
    return found.group(1) if found else "no error code"


@dataclass(frozen=True, slots=True)
class S3ObjectLockSink:
    """A 412 means the key already exists from an earlier attempt whose acknowledgement was lost. The key carries the
    last record's hash, which covers every record in the batch, so the stored object is this batch and it counts
    as delivered."""

    settings: S3SinkSettings
    store: ObjectStore
    sign: Signer
    name: str = "s3"

    async def deliver(self, batch: Batch) -> DeliveryResult:
        key: Final = object_key(self.settings.prefix, batch)
        url: Final = object_url(self.settings, key)
        content: Final = ndjson(batch)
        try:
            signed: Final = await run_aws_signing(self.sign, url, content, put_headers(self.settings, batch, content))
            response: Final = await self.store.put(url, content, signed, self.settings.timeout_seconds)
        except httpx.HTTPError as error:
            return Undelivered(f"{type(error).__name__} putting {key}")
        except Exception as error:  # noqa: BLE001  # a failing credential chain is a failed delivery, not a dead worker
            return Undelivered(f"{type(error).__name__} signing {key}")
        match response.status_code:
            case 200 | 412:
                return Delivered(len(batch))
            case status:
                return Undelivered(f"HTTP {status} {_error_code(response)} putting {key}")

    async def close(self) -> None:
        return None
