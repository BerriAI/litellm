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

import math
import random
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Final

import pytest
from pydantic import BaseModel, ConfigDict, Field

from e2e_config import (
    DD_API_KEY,
    DD_APP_KEY,
    DD_SEARCH_FROM,
    DD_SEARCH_INTERVAL,
    DD_SETTLE_SECONDS,
    DD_SITE,
    POLL_TIMEOUT,
)
from e2e_http import URL, Headers, StreamingResponse, send

type SearchCall = Callable[[str, float], StreamingResponse]


def _seconds(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        seconds: Final = float(value)
    except ValueError:
        return None
    return seconds if math.isfinite(seconds) and seconds >= 0 else None


def _rate_limit_delay(headers: Mapping[str, str]) -> float:
    delays: Final = tuple(
        delay
        for name in ("x-ratelimit-reset", "retry-after")
        if (delay := _seconds(headers.get(name))) is not None
    )
    return max(1.0, max(delays, default=DD_SEARCH_INTERVAL))


class _DdAuthHeaders(Headers):
    api_key: str = Field(serialization_alias="DD-API-KEY", repr=False)
    app_key: str = Field(serialization_alias="DD-APPLICATION-KEY", repr=False)


class _SearchFilter(BaseModel):
    query: str
    #: Wide enough to cover a full suite run plus DataDog's ingestion lag;
    #: markers are unique per test, so a wide window cannot match foreign events.
    #: Override via E2E_DD_SEARCH_FROM when CI lookback needs more than the default.
    from_: str = Field(default_factory=lambda: DD_SEARCH_FROM, serialization_alias="from")
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
    api_key: str = field(repr=False)
    app_key: str = field(repr=False)
    search: SearchCall | None = field(default=None, repr=False)
    now: Callable[[], float] = field(default=time.monotonic, repr=False)
    sleep: Callable[[float], None] = field(default=time.sleep, repr=False)
    jitter: Callable[[], float] = field(default=random.random, repr=False)

    def events_for_marker(self, marker: str) -> list[DdLogEvent]:
        """Every ingested event whose attributes carry the marker. DataDog
        consumes the shipped JSON message into ``attributes`` and leaves the
        indexed ``message`` empty, so a plain full-text query matches nothing;
        ``*:`` extends the scan to every attribute (the marker sits in the
        prompt, e.g. ``messages.content``, wherever the route's payload puts
        it)."""
        return self.events_for_query(f"*:*{marker}*")

    def events_for_query(self, query: str) -> list[DdLogEvent]:
        """Every ingested event the search query matches (failure payloads
        carry no prompt to mark, so failure scenarios query indexed attributes
        like ``@model_group:...`` instead of a body marker). More than one hit
        for one call IS the duplicate-delivery bug, so this never collapses to
        a single event. A 429 backs off and retries - the search budget is
        org-wide, so another consumer can empty it under us - while any other
        failure stays a hard fail."""
        return self._events_for_query(query, self.now() + POLL_TIMEOUT)

    def _events_for_query(self, query: str, deadline: float) -> list[DdLogEvent]:
        search: Final = self.search or self._search_page
        while (remaining := deadline - self.now()) > 0:
            if (result := search(query, min(30.0, remaining))).ok:
                return [event.attributes for event in _SearchResponse.model_validate_json(result.body).data]
            if result.status_code != 429:
                pytest.fail(f"DataDog Logs Search API at api.{self.site} failed with HTTP {result.status_code}")
            if (delay := min(_rate_limit_delay(result.headers) + self.jitter(), deadline - self.now())) > 0:
                self.sleep(delay)
        pytest.fail(
            f"DataDog Logs Search API at api.{self.site} remained rate-limited for {POLL_TIMEOUT}s; "
            "the org-wide logs_public_search_api budget is exhausted"
        )

    def _search_page(self, query: str, timeout: float) -> StreamingResponse:
        return send(
            URL(f"https://api.{self.site}/api/v2/logs/events/search"),
            headers=_DdAuthHeaders(api_key=self.api_key, app_key=self.app_key),
            json=_SearchRequest(filter=_SearchFilter(query=query)),
            timeout=timeout,
        )

    def poll_events_for_marker(self, marker: str) -> list[DdLogEvent]:
        """``poll_events_for_query`` over the every-attribute marker scan."""
        return self.poll_events_for_query(f"*:*{marker}*")

    def poll_events_for_query(self, query: str) -> list[DdLogEvent]:
        """Poll until at least one matching event is searchable (the callback
        flushes in periodic batches and DataDog ingestion adds seconds of lag),
        then keep re-reading for DD_SETTLE_SECONDS so a late duplicate cannot
        hide from the exactly-one assertion - real-DataDog jitter can surface
        one call's two events tens of seconds apart. Searches pace at
        DD_SEARCH_INTERVAL, not POLL_INTERVAL, to respect the search API's
        request budget. Discovery, quota retries, and duplicate detection share
        one POLL_TIMEOUT deadline; an incomplete settle window fails closed."""
        deadline: Final = self.now() + POLL_TIMEOUT
        while (remaining := deadline - self.now()) > 0:
            events = self._events_for_query(query, deadline)
            if events:
                return self._settled_events_for_query(query, events, deadline)
            if (remaining := deadline - self.now()) > 0:
                self.sleep(min(DD_SEARCH_INTERVAL, remaining))
        return []

    def _settled_events_for_query(self, query: str, events: list[DdLogEvent], deadline: float) -> list[DdLogEvent]:
        """Re-read at every search interval until the settle window closes; a
        duplicate ends the watch early because more waiting cannot clear it.

        Keep the last non-empty result: a transient empty search (index lag)
        must not erase events already confirmed earlier in the settle window.
        A successful final search must reach the full settle window before the
        shared read-back deadline; otherwise duplicate detection is incomplete.
        """
        settle_deadline: Final = self.now() + DD_SETTLE_SECONDS
        last_nonempty = events
        if len(events) > 1:
            return events
        while (remaining := deadline - self.now()) > 0:
            self.sleep(min(DD_SEARCH_INTERVAL, remaining))
            if self.now() >= deadline:
                break
            latest = self._events_for_query(query, deadline)
            if len(latest) > 1:
                return latest
            if latest:
                last_nonempty = latest
            if self.now() >= settle_deadline:
                return last_nonempty
        pytest.fail(f"DataDog log delivery could not complete its duplicate-detection window within {POLL_TIMEOUT}s")


def build_dd_logs_reader() -> DdLogsReader:
    if not DD_API_KEY or not DD_APP_KEY:
        pytest.fail(
            "DD_API_KEY and DD_APP_KEY must be set: the DataDog tests deliver to and "
            "read back from the real DataDog API (on the cluster the secret manager "
            "injects them; locally set them in tests/e2e/.env)"
        )
    return DdLogsReader(site=DD_SITE, api_key=DD_API_KEY, app_key=DD_APP_KEY)
