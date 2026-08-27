"""Read-back for the gcs_bucket logging test against the real GCS bucket.

The proxy ships StandardLoggingPayload objects with its own service account
(litellm_settings.callbacks: ["gcs_bucket"] + GCS_BUCKET_NAME), and the test
reads them back through the GCS JSON API. Auth is a self-signed service-account
JWT (RS256 via PyJWT + cryptography, both litellm proxy dependencies the
runner installs) minted per request and sent directly as the Bearer token -
Google accepts that for storage.googleapis.com with no token exchange, which
keeps every HTTP read inside ``e2e_http``.

The default gcs_bucket mode batches payloads into ``{date}/batch-{id}.ndjson``
objects; unbatched mode writes ``{date}/{response_id}`` per call. The reader
handles both: it polls the day's listing, downloads the direct object when
present, and otherwise scans batch objects fresh enough to hold the call.
Missing configuration is a hard failure, never a skip.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import quote

import jwt
import pytest
from pydantic import BaseModel, ConfigDict, Field

from e2e_config import POLL_INTERVAL, POLL_TIMEOUT
from e2e_http import URL, Headers, probe

_GCS_API = "https://storage.googleapis.com"
#: Tolerance for clock skew between this host and GCS object timestamps.
_SKEW = timedelta(seconds=120)
#: How long to keep re-reading after the first match before trusting the
#: exactly-one assertion: past one full gcs_bucket flush interval (~20s), so
#: a duplicate shipped by a later flush is seen, plus listing-latency margin.
GCS_SETTLE_SECONDS = 45.0


class _ServiceAccount(BaseModel):
    model_config = ConfigDict(extra="ignore")

    client_email: str
    private_key: str


class _GcsAuthHeaders(Headers):
    authorization: str = Field(serialization_alias="Authorization")


class _GcsObject(BaseModel):
    model_config = ConfigDict(extra="ignore")

    name: str
    updated: datetime | None = None


class _GcsListResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    items: list[_GcsObject] = []
    next_page_token: str | None = Field(default=None, validation_alias="nextPageToken")


class _GcsListParams(BaseModel):
    prefix: str
    max_results: int = Field(default=1000, serialization_alias="maxResults")
    page_token: str | None = Field(default=None, serialization_alias="pageToken")


class _GcsMediaParams(BaseModel):
    alt: str = "media"


class GcsLogRecord(BaseModel):
    """The StandardLoggingPayload fields the gcs scenario pins."""

    model_config = ConfigDict(extra="ignore")

    id: str
    status: str
    model_group: str | None = None
    response_cost: float | None = None
    total_tokens: int | None = None
    error_str: str | None = None


def _mint_bearer(account: _ServiceAccount) -> str:
    """Self-signed service-account JWT: for Google APIs a token whose ``aud``
    is the service endpoint authorizes directly, no oauth2 token exchange.
    Minted per request so a long session never outlives one token's expiry."""
    now = int(time.time())
    claims: dict[str, str | int] = {
        "iss": account.client_email,
        "sub": account.client_email,
        "aud": f"{_GCS_API}/",
        "iat": now,
        "exp": now + 3600,
    }
    return jwt.encode(claims, account.private_key, algorithm="RS256")


@dataclass(frozen=True, slots=True)
class GcsLogReader:
    bucket: str
    account: _ServiceAccount

    def _headers(self) -> _GcsAuthHeaders:
        return _GcsAuthHeaders(authorization=f"Bearer {_mint_bearer(self.account)}")

    def _list(self, prefix: str) -> list[_GcsObject]:
        """Every object under ``prefix``, following ``nextPageToken`` - the
        shared day prefix accumulates all of the proxy's traffic, and a fresh
        record past the 1000-object page cap must still be seen."""
        items: list[_GcsObject] = []
        page_token: str | None = None
        while True:
            result = probe(
                URL(f"{_GCS_API}/storage/v1/b/{self.bucket}/o"),
                headers=self._headers(),
                params=_GcsListParams(prefix=prefix, page_token=page_token),
            )
            if result.status_code != 200:
                pytest.fail(
                    f"GCS object listing for gs://{self.bucket}/{prefix} failed "
                    f"({result.status_code}): {result.body[:300]}"
                )
            page = _GcsListResponse.model_validate_json(result.body)
            items.extend(page.items)
            page_token = page.next_page_token
            if not page_token:
                return items

    def _download(self, name: str) -> str:
        result = probe(
            URL(f"{_GCS_API}/storage/v1/b/{self.bucket}/o/{quote(name, safe='')}"),
            headers=self._headers(),
            params=_GcsMediaParams(),
        )
        if result.status_code != 200:
            pytest.fail(
                f"GCS object download gs://{self.bucket}/{name} failed ({result.status_code}): {result.body[:300]}"
            )
        return result.body

    def records_for_response_id(self, response_id: str, *, since: datetime) -> list[GcsLogRecord]:
        """Every payload written for ``response_id``: the direct
        ``{date}/{response_id}`` object plus any hit inside batch NDJSON
        objects updated after ``since``. More than one hit is the
        duplicate-delivery bug, so this never collapses to a single record."""
        records: list[GcsLogRecord] = []
        window_start = since - _SKEW
        for day_offset in (0, 1):
            day = (since + timedelta(days=day_offset)).strftime("%Y-%m-%d")
            for obj in self._list(f"{day}/"):
                if obj.name == f"{day}/{response_id}":
                    records.append(GcsLogRecord.model_validate_json(self._download(obj.name)))
                    continue
                is_fresh_batch = f"{day}/batch-" in obj.name and obj.updated is not None and obj.updated >= window_start
                if is_fresh_batch:
                    records.extend(
                        GcsLogRecord.model_validate_json(line)
                        for line in self._download(obj.name).splitlines()
                        if response_id in line
                    )
        return records

    def poll_records_for_response_id(self, response_id: str, *, since: datetime) -> list[GcsLogRecord]:
        """Poll until the payload is readable (the gcs_bucket callback flushes
        on a ~20s timer), then keep re-reading for GCS_SETTLE_SECONDS - past a
        full flush interval - so a duplicate shipped by a later flush cannot
        hide from the exactly-one assertion. A duplicate ends the settle early
        because more waiting cannot clear it."""
        deadline = time.monotonic() + POLL_TIMEOUT
        while time.monotonic() < deadline:
            records = self.records_for_response_id(response_id, since=since)
            if records:
                return self._settled_records(response_id, since=since, first=records)
            time.sleep(POLL_INTERVAL)
        return []

    def _settled_records(self, response_id: str, *, since: datetime, first: list[GcsLogRecord]) -> list[GcsLogRecord]:
        """Re-read at every poll interval until the settle window closes; a
        transiently empty re-read never downgrades what was already seen."""
        settle_deadline = time.monotonic() + GCS_SETTLE_SECONDS
        latest = first
        while time.monotonic() < settle_deadline and len(latest) <= 1:
            time.sleep(POLL_INTERVAL)
            latest = self.records_for_response_id(response_id, since=since) or latest
        return latest


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def build_gcs_reader() -> GcsLogReader:
    bucket = os.environ.get("GCS_BUCKET_NAME", "")
    if not bucket:
        pytest.fail(
            "GCS_BUCKET_NAME must be set: the gcs test reads the proxy's gcs_bucket "
            "delivery back from the real bucket (the cluster secret manager injects "
            "it; locally set it in tests/e2e/.env)"
        )
    raw = ""
    credentials_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "")
    if credentials_path and Path(credentials_path).is_file():
        raw = Path(credentials_path).read_text()
    else:
        raw = os.environ.get("VERTEXAI_CREDENTIALS", "")
    if not raw:
        pytest.fail(
            "GCS read-back needs a service-account key: set "
            "GOOGLE_APPLICATION_CREDENTIALS (path) or VERTEXAI_CREDENTIALS (JSON), "
            "as the cluster secret manager does"
        )
    return GcsLogReader(bucket=bucket, account=_ServiceAccount.model_validate_json(raw))
