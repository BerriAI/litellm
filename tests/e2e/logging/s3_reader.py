"""Read-back for the S3 logging tests against the real bucket the proxy writes to.

Delivery is judged on what actually landed in S3: the proxy uploads with its own
AWS credentials exactly as in production (no local minio, no endpoint override),
and the tests fetch the object back with the official AWS SDK. Which bucket the
proxy is configured to write to is not discoverable over any proxy route, so the
test process is told through E2E_S3_LOG_BUCKET (plus E2E_S3_LOG_REGION,
E2E_S3_LOG_PATH when the deployment sets an s3_path prefix, and
E2E_S3_LOG_PROFILE to read back through a named AWS profile). A missing bucket is
a hard failure, never an empty result.

An object's key ends in the id litellm assigned the call - the completion id for
a successful call, the litellm call id (the x-litellm-call-id response header)
for a failed one - so a test locates its own object exactly, without scanning
anyone else's, and deletes just that key on teardown.

boto3 is the AWS SDK's own client and is deliberately not routed through
``e2e_http``: hand-signing SigV4 to reach S3 would be a worse test than using the
vendor client. Its s3 client is untyped, so it is confined to this module behind
the ``_S3Client`` protocol, its responses are narrowed here, and every payload is
validated into a pydantic model before a test sees it.
"""

from __future__ import annotations

import os
import time
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Protocol, cast

import boto3
import pytest
from pydantic import BaseModel, ConfigDict, ValidationError

from e2e_config import POLL_INTERVAL, POLL_TIMEOUT

S3_LOG_BUCKET = os.environ.get("E2E_S3_LOG_BUCKET", "").strip()
S3_LOG_REGION = os.environ.get("E2E_S3_LOG_REGION", "us-east-1").strip()
#: The deployment's litellm_settings.s3_callback_params.s3_path, if it sets one.
S3_LOG_PATH = os.environ.get("E2E_S3_LOG_PATH", "").strip()
#: Named AWS profile for the read-back; empty uses boto3's default credential chain.
S3_LOG_PROFILE = os.environ.get("E2E_S3_LOG_PROFILE", "").strip()


class _S3Body(Protocol):
    def read(self) -> bytes: ...


class _S3Client(Protocol):
    """The three S3 calls this module makes."""

    def list_objects_v2(self, **kwargs: str) -> object: ...

    def get_object(self, *, Bucket: str, Key: str) -> object: ...

    def delete_object(self, *, Bucket: str, Key: str) -> object: ...


class S3LogError(BaseModel):
    """The error_information block of a failed call's payload."""

    model_config = ConfigDict(extra="ignore")

    error_class: str = ""
    error_code: str = ""
    llm_provider: str = ""


class S3LogMessage(BaseModel):
    model_config = ConfigDict(extra="ignore")

    role: str
    content: str


class S3LogRecord(BaseModel):
    """The StandardLoggingPayload as the proxy wrote it into the bucket. Only the
    fields the delivery tests pin are modelled."""

    model_config = ConfigDict(extra="ignore")

    id: str
    status: str
    call_type: str
    model_group: str
    response_cost: float
    total_tokens: int
    messages: tuple[S3LogMessage, ...] = ()
    error_information: S3LogError = S3LogError()
    error_str: str | None = None


@dataclass(frozen=True, slots=True)
class S3LogObject:
    key: str
    record: S3LogRecord


@dataclass(frozen=True, slots=True)
class _ListPage:
    keys: tuple[str, ...]
    next_token: str | None


def _s3_client() -> _S3Client:
    session = boto3.Session(profile_name=S3_LOG_PROFILE) if S3_LOG_PROFILE else boto3.Session()
    client = session.client("s3", region_name=S3_LOG_REGION)  # pyright: ignore[reportUnknownMemberType]  # boto3 ships no types for the per-service clients; _S3Client pins the calls made here
    return cast(_S3Client, client)


def _response(raw: object, call: str) -> dict[str, object]:
    if not isinstance(raw, dict):
        pytest.fail(f"S3 {call} returned {type(raw).__name__}, not a response mapping")
    return cast(dict[str, object], raw)


def _list_page(raw: object) -> _ListPage:
    listing = _response(raw, "list_objects_v2")
    contents = listing.get("Contents", [])
    if not isinstance(contents, list):
        pytest.fail(f"S3 list_objects_v2 Contents was {type(contents).__name__}, not a list")
    keys = tuple(
        key
        for entry in cast(list[object], contents)
        if isinstance(entry, dict) and isinstance(key := cast(dict[str, object], entry).get("Key"), str)
    )
    token = listing.get("NextContinuationToken")
    truncated = listing.get("IsTruncated") is True
    return _ListPage(keys=keys, next_token=token if truncated and isinstance(token, str) else None)


def _body(raw: object) -> bytes:
    stream = _response(raw, "get_object").get("Body")
    if not hasattr(stream, "read"):
        pytest.fail(f"S3 get_object returned a Body of {type(stream).__name__}, which is not readable")
    return cast(_S3Body, stream).read()


@dataclass(frozen=True, slots=True)
class S3LogsReader:
    bucket: str
    path: str

    def _keys(self, client: _S3Client) -> Iterator[str]:
        page = _list_page(client.list_objects_v2(Bucket=self.bucket, Prefix=self.path))
        while True:
            yield from page.keys
            if page.next_token is None:
                return
            page = _list_page(
                client.list_objects_v2(Bucket=self.bucket, Prefix=self.path, ContinuationToken=page.next_token)
            )

    def _record(self, client: _S3Client, key: str) -> S3LogRecord:
        body = _body(client.get_object(Bucket=self.bucket, Key=key))
        try:
            return S3LogRecord.model_validate_json(body)
        except ValidationError as exc:
            pytest.fail(f"the object at {key} is not a StandardLoggingPayload: {exc}")

    def objects_for_call(self, call_id: str) -> tuple[S3LogObject, ...]:
        """Every object the proxy wrote for the call litellm identified as
        ``call_id``. More than one is a duplicate-delivery bug, so this never
        collapses to a single object."""
        client = _s3_client()
        suffix = f"_{call_id}.json"
        return tuple(
            S3LogObject(key=key, record=self._record(client, key))
            for key in self._keys(client)
            if key.endswith(suffix)
        )

    def poll_objects_for_call(self, call_id: str) -> tuple[S3LogObject, ...]:
        """Poll until the call's object is readable - the integration batches
        uploads behind a flush interval - and return whatever is there at the
        deadline, so the caller's assertion, not a timeout, reports the gap."""
        deadline = time.monotonic() + POLL_TIMEOUT
        while time.monotonic() < deadline:
            found = self.objects_for_call(call_id)
            if found:
                return found
            time.sleep(POLL_INTERVAL)
        return self.objects_for_call(call_id)

    def delete(self, key: str) -> None:
        _ = _s3_client().delete_object(Bucket=self.bucket, Key=key)


def build_s3_logs_reader() -> S3LogsReader:
    if not S3_LOG_BUCKET:
        pytest.fail(
            "E2E_S3_LOG_BUCKET must name the bucket the proxy's s3_callback_params write to: "
            "the s3 tests read delivery back out of the real bucket, and no proxy route reports "
            "which one is configured; missing it is a hard failure, not a skip"
        )
    return S3LogsReader(bucket=S3_LOG_BUCKET, path=S3_LOG_PATH)
