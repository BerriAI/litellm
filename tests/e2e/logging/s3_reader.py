"""Read-back for the s3 logging tests against the real S3 bucket the proxy
ships StandardLoggingPayload objects to (litellm_settings.callbacks: ["s3_v2"]).

Delivery is judged on what actually landed in the bucket: the proxy writes
with its own credentials exactly as in production, and the tests list and
download the objects back with boto3 (already a litellm proxy dependency, so
the e2e runner image carries it; it is an AWS SDK, not a raw HTTP client, so
the e2e_http-only transport rule is untouched). The bucket comes from
S3_LOGS_BUCKET_NAME - on the cluster the secret manager injects it, locally
tests/e2e/.env provides it. Missing configuration is a hard failure, never a
skip.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import boto3
import pytest
from pydantic import BaseModel, ConfigDict

from e2e_config import POLL_INTERVAL, POLL_TIMEOUT

if TYPE_CHECKING:
    from types_boto3_s3.client import S3Client

#: How long to keep re-reading after the first match before trusting the
#: exactly-one assertion: past one full s3_v2 flush interval (~10s), so a
#: duplicate shipped by a LATER flush is seen, plus listing-latency margin.
#: The DataDog reader settles the same way (DD_SETTLE_SECONDS).
S3_SETTLE_SECONDS = 25.0


class S3LogRecord(BaseModel):
    """The StandardLoggingPayload fields the s3 scenarios pin."""

    model_config = ConfigDict(extra="ignore")

    id: str
    status: str
    model_group: str | None = None
    response_cost: float | None = None
    total_tokens: int | None = None
    error_str: str | None = None


@dataclass(frozen=True, slots=True)
class S3LogReader:
    bucket: str
    client: S3Client

    def list_keys(self, prefix: str) -> list[str]:
        response = self.client.list_objects_v2(Bucket=self.bucket, Prefix=prefix)
        return [obj["Key"] for obj in response.get("Contents", []) if "Key" in obj]

    def read_record(self, key: str) -> S3LogRecord:
        body = self.client.get_object(Bucket=self.bucket, Key=key)["Body"].read()
        return S3LogRecord.model_validate_json(body)

    def records_matching(self, *, prefix: str, predicate: Callable[[S3LogRecord], bool]) -> list[S3LogRecord]:
        return [record for record in map(self.read_record, self.list_keys(prefix)) if predicate(record)]

    def poll_records(self, *, prefix: str, predicate: Callable[[S3LogRecord], bool]) -> list[S3LogRecord]:
        """Poll until at least one matching object is listed (the s3_v2
        callback flushes on a ~10s timer), then keep re-reading for
        S3_SETTLE_SECONDS - past a full flush interval - so a duplicate
        shipped by a later flush cannot hide from the exactly-one assertion.
        One blind spot is inherent: a duplicate write that reuses the exact
        same object key overwrites the first object and no listing can see
        it; distinct-key duplicates are what this catches. At the deadline an
        empty list is returned and the caller's assertion carries the failure
        message."""
        deadline = time.monotonic() + POLL_TIMEOUT
        while time.monotonic() < deadline:
            records = self.records_matching(prefix=prefix, predicate=predicate)
            if records:
                return self._settled_records(prefix=prefix, predicate=predicate, first=records)
            time.sleep(POLL_INTERVAL)
        return []

    def _settled_records(
        self, *, prefix: str, predicate: Callable[[S3LogRecord], bool], first: list[S3LogRecord]
    ) -> list[S3LogRecord]:
        """Re-read at every poll interval until the settle window closes; a
        duplicate ends the watch early because more waiting cannot clear it.
        A transiently empty re-read never downgrades what was already seen."""
        settle_deadline = time.monotonic() + S3_SETTLE_SECONDS
        latest = first
        while time.monotonic() < settle_deadline and len(latest) <= 1:
            time.sleep(POLL_INTERVAL)
            latest = self.records_matching(prefix=prefix, predicate=predicate) or latest
        return latest


def build_s3_reader() -> S3LogReader:
    bucket = os.environ.get("S3_LOGS_BUCKET_NAME", "")
    if not bucket:
        pytest.fail(
            "S3_LOGS_BUCKET_NAME must be set: the s3 tests read the proxy's s3_v2 "
            "delivery back from the real bucket (the cluster secret manager injects "
            "it; locally set it in tests/e2e/.env to the same bucket "
            "s3_callback_params.s3_bucket_name names)"
        )
    region = os.environ.get("AWS_REGION_NAME") or os.environ.get("AWS_REGION") or "us-east-1"
    return S3LogReader(
        bucket=bucket,
        # boto3.client's overload set covers every AWS service; the ones without
        # installed stubs type as Unknown, so the member is "partially unknown"
        # even though the s3 overload itself resolves to S3Client.
        client=boto3.client("s3", region_name=region),  # pyright: ignore[reportUnknownMemberType]
    )
