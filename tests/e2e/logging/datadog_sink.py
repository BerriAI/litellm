"""Read-back for the DataDog logging tests: typed models over the compose
stack's dd-sink service, which records every logs-intake POST the datadog
callback sends (gunzipped) and replays them as JSON.

Delivery is judged on what the sink actually received, mirroring how the OTEL
tests read Jaeger; a failed sink query is a hard failure, never an empty
result. External reads go through ``e2e_http``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import pytest
from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError

from e2e_config import DD_SINK_URL, POLL_INTERVAL, POLL_TIMEOUT
from e2e_http import URL, NoBody, Success, get


class DdSinkRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    path: str
    body: str


class DdSinkRequests(BaseModel):
    model_config = ConfigDict(extra="ignore")

    requests: list[DdSinkRequest] = []


class DdLogEvent(BaseModel):
    model_config = ConfigDict(extra="ignore")

    message: str
    ddsource: str | None = None
    service: str | None = None
    status: str | None = None


_EVENT_BATCH: TypeAdapter[list[DdLogEvent]] = TypeAdapter(list[DdLogEvent])


@dataclass(frozen=True, slots=True)
class DdSinkReader:
    sink_url: str

    def _recorded_requests(self) -> list[DdSinkRequest]:
        result = get(
            URL(f"{self.sink_url}/requests"),
            headers=NoBody(),
            params=NoBody(),
            response_type=DdSinkRequests,
            timeout=30.0,
        )
        match result:
            case Success(data=page):
                return page.requests
            case failure:
                pytest.fail(f"dd-sink query at {self.sink_url} failed: {failure}")

    def events_for_marker(self, marker: str) -> list[DdLogEvent]:
        """Every log event across every recorded intake batch whose message
        carries the marker. More than one hit for one call IS the
        duplicate-delivery bug, so this never collapses to a single event."""
        events: list[DdLogEvent] = []
        for request in self._recorded_requests():
            if "/api/v2/logs" not in request.path:
                continue
            try:
                batch = _EVENT_BATCH.validate_json(request.body)
            except ValidationError:
                # the intake accepts a single event object as well as an array
                try:
                    batch = [DdLogEvent.model_validate_json(request.body)]
                except ValidationError:
                    pytest.fail(
                        f"dd-sink recorded a non-log body on {request.path}: {request.body[:200]}"
                    )
            events.extend(event for event in batch if marker in event.message)
        return events

    def poll_events_for_marker(self, marker: str) -> list[DdLogEvent]:
        """Poll until at least one matching event lands (the callback flushes
        in periodic batches), then re-read after one more interval so a late
        duplicate cannot hide from the exactly-one assertion. At the deadline
        the last result is returned as-is."""
        deadline = time.monotonic() + POLL_TIMEOUT
        while time.monotonic() < deadline:
            events = self.events_for_marker(marker)
            if events:
                time.sleep(POLL_INTERVAL)
                return self.events_for_marker(marker)
            time.sleep(POLL_INTERVAL)
        return self.events_for_marker(marker)


def build_dd_sink_reader() -> DdSinkReader:
    return DdSinkReader(sink_url=DD_SINK_URL)
