"""Read-back for the DataDog logging tests against the real DataDog Logs
Search API.

Delivery is judged on what DataDog itself ingested: the proxy ships logs with
DD_API_KEY exactly as in production (no base-URL override, no local sink), and
the tests search the ingested events back with POST /api/v2/logs/events/search,
authenticated with the same DD_API_KEY plus a DD_APP_KEY application key. On
the cluster the secret manager injects both keys; locally tests/e2e/.env
provides them. Missing keys or a failed search call are hard failures, never an
empty result. External reads go through ``e2e_http``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import pytest
from pydantic import BaseModel, ConfigDict, Field

from e2e_config import (
    DD_API_KEY,
    DD_APP_KEY,
    DD_SETTLE_SECONDS,
    DD_SITE,
    POLL_INTERVAL,
    POLL_TIMEOUT,
)
from e2e_http import URL, Headers, Success, post


class _DdAuthHeaders(Headers):
    api_key: str = Field(serialization_alias="DD-API-KEY")
    app_key: str = Field(serialization_alias="DD-APPLICATION-KEY")


class _SearchFilter(BaseModel):
    query: str
    #: Wide enough to cover a full suite run plus DataDog's ingestion lag;
    #: markers are unique per test, so a wide window cannot match foreign events.
    from_: str = Field(default="now-30m", serialization_alias="from")
    to: str = "now"


class _SearchPage(BaseModel):
    limit: int = 100


class _SearchRequest(BaseModel):
    filter: _SearchFilter
    page: _SearchPage = _SearchPage()
    sort: str = "timestamp"


class DdLogEvent(BaseModel):
    """One ingested log event as the search API returns it: the indexed
    envelope (service/status/tags) plus ``attributes`` - DataDog's parse of the
    JSON message the integration shipped, i.e. the StandardLoggingPayload
    fields."""

    model_config = ConfigDict(extra="ignore")

    service: str | None = None
    status: str | None = None
    tags: list[str] = []
    attributes: dict[str, object] = {}


class _SearchEvent(BaseModel):
    model_config = ConfigDict(extra="ignore")

    attributes: DdLogEvent


class _SearchResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    data: list[_SearchEvent] = []


@dataclass(frozen=True, slots=True)
class DdLogsReader:
    site: str
    api_key: str
    app_key: str

    def events_for_marker(self, marker: str) -> list[DdLogEvent]:
        """Every ingested event matching the marker (full-text, exact phrase).
        More than one hit for one call IS the duplicate-delivery bug, so this
        never collapses to a single event."""
        result = post(
            URL(f"https://api.{self.site}/api/v2/logs/events/search"),
            headers=_DdAuthHeaders(api_key=self.api_key, app_key=self.app_key),
            json=_SearchRequest(filter=_SearchFilter(query=f'"{marker}"')),
            response_type=_SearchResponse,
            timeout=30.0,
        )
        match result:
            case Success(data=page):
                return [event.attributes for event in page.data]
            case failure:
                pytest.fail(f"DataDog Logs Search API at api.{self.site} failed: {failure}")

    def poll_events_for_marker(self, marker: str) -> list[DdLogEvent]:
        """Poll until at least one matching event is searchable (the callback
        flushes in periodic batches and DataDog ingestion adds seconds of lag),
        then keep re-reading for DD_SETTLE_SECONDS so a late duplicate cannot
        hide from the exactly-one assertion - real-DataDog jitter can surface
        one call's two events tens of seconds apart. At the deadline the last
        result is returned as-is."""
        deadline = time.monotonic() + POLL_TIMEOUT
        while time.monotonic() < deadline:
            events = self.events_for_marker(marker)
            if events:
                return self._settled_events_for_marker(marker, events)
            time.sleep(POLL_INTERVAL)
        return self.events_for_marker(marker)

    def _settled_events_for_marker(
        self, marker: str, events: list[DdLogEvent]
    ) -> list[DdLogEvent]:
        """Re-read at every poll interval until the settle window closes; a
        duplicate ends the watch early because more waiting cannot clear it."""
        settle_deadline = time.monotonic() + DD_SETTLE_SECONDS
        while time.monotonic() < settle_deadline:
            time.sleep(POLL_INTERVAL)
            events = self.events_for_marker(marker)
            if len(events) > 1:
                return events
        return events


def build_dd_logs_reader() -> DdLogsReader:
    if not DD_API_KEY or not DD_APP_KEY:
        pytest.fail(
            "DD_API_KEY and DD_APP_KEY must be set: the DataDog tests deliver to and "
            "read back from the real DataDog API (on the cluster the secret manager "
            "injects them; locally set them in tests/e2e/.env)"
        )
    return DdLogsReader(site=DD_SITE, api_key=DD_API_KEY, app_key=DD_APP_KEY)
