import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import Final

import pytest

from datadog_reader import DdLogsReader
from datadog_reader import _DdAuthHeaders  # pyright: ignore[reportPrivateUsage]  # verifies private auth-header serialization
from e2e_config import DD_SEARCH_INTERVAL, POLL_TIMEOUT
from e2e_http import StreamingResponse


def test_failure_diagnostics_hide_credentials_without_changing_auth_headers() -> None:
    api_key: Final = "test-datadog-api-secret"
    app_key: Final = "test-datadog-app-secret"
    reader: Final = DdLogsReader(site="datadoghq.com", api_key=api_key, app_key=app_key)
    headers: Final = _DdAuthHeaders(api_key=api_key, app_key=app_key)

    for value in (reader, headers):
        assert api_key not in repr(value)
        assert app_key not in repr(value)

    assert headers.model_dump(by_alias=True) == {
        "DD-API-KEY": api_key,
        "DD-APPLICATION-KEY": app_key,
    }


@dataclass
class Clock:
    elapsed: float = 0.0

    def now(self) -> float:
        return self.elapsed

    def sleep(self, seconds: float) -> None:
        self.elapsed += seconds


@dataclass
class Search:
    responses: Iterator[StreamingResponse]
    calls: list[tuple[str, float]] = field(default_factory=list)

    def __call__(self, query: str, timeout: float) -> StreamingResponse:
        self.calls.append((query, timeout))
        return next(self.responses)


def _page(*event_ids: str) -> StreamingResponse:
    return StreamingResponse(
        status_code=200,
        body=json.dumps({"data": [{"attributes": {"attributes": {"id": event_id}}} for event_id in event_ids]}),
    )


def _reader(responses: Sequence[StreamingResponse], clock: Clock) -> tuple[DdLogsReader, Search]:
    search: Final = Search(iter(responses))
    return DdLogsReader(
        site="us5.datadoghq.com",
        api_key="test-api-secret",
        app_key="test-app-secret",
        search=search,
        now=clock.now,
        sleep=clock.sleep,
        jitter=lambda: 0.25,
    ), search


def test_429_honors_server_reset_and_preserves_duplicate_events() -> None:
    clock: Final = Clock()
    reader, search = _reader(
        (StreamingResponse(status_code=429, body="", headers={"x-ratelimit-reset": "6"}), _page("first", "duplicate")),
        clock,
    )

    events: Final = reader.events_for_query("test-marker")

    assert tuple(event.attributes["id"] for event in events) == ("first", "duplicate")
    assert clock.elapsed == 6.25
    assert search.calls == [("test-marker", 30.0), ("test-marker", 30.0)]


@pytest.mark.parametrize("reset", ("", "invalid", "nan", "inf", "-1"))
def test_invalid_reset_uses_search_interval(reset: str) -> None:
    clock: Final = Clock()
    reader, _ = _reader(
        (StreamingResponse(status_code=429, body="", headers={"x-ratelimit-reset": reset}), _page()), clock
    )

    assert reader.events_for_query("test-marker") == []
    assert clock.elapsed == DD_SEARCH_INTERVAL + 0.25


def test_zero_reset_cannot_create_a_busy_retry_loop() -> None:
    clock: Final = Clock()
    reader, _ = _reader(
        (StreamingResponse(status_code=429, body="", headers={"x-ratelimit-reset": "0"}), _page()), clock
    )

    assert reader.events_for_query("test-marker") == []
    assert clock.elapsed == 1.25


def test_retry_after_is_not_shortened_by_an_earlier_reset() -> None:
    clock: Final = Clock()
    reader, _ = _reader(
        (StreamingResponse(status_code=429, body="", headers={"x-ratelimit-reset": "2", "retry-after": "8"}), _page()),
        clock,
    )

    assert reader.events_for_query("test-marker") == []
    assert clock.elapsed == 8.25


def test_rate_limit_wait_stops_at_deadline_without_issuing_another_request() -> None:
    clock: Final = Clock()
    reader, search = _reader(
        (StreamingResponse(status_code=429, body="", headers={"x-ratelimit-reset": str(POLL_TIMEOUT * 10)}),), clock
    )

    with pytest.raises(pytest.fail.Exception, match="remained rate-limited"):
        reader.events_for_query("test-marker")

    assert clock.elapsed == POLL_TIMEOUT
    assert search.calls == [("test-marker", 30.0)]


def test_late_retry_cannot_receive_a_fresh_request_timeout() -> None:
    clock: Final = Clock()
    reader, search = _reader(
        (StreamingResponse(status_code=429, body="", headers={"x-ratelimit-reset": str(POLL_TIMEOUT - 5)}), _page()),
        clock,
    )

    assert reader.events_for_query("test-marker") == []
    assert search.calls == [("test-marker", 30.0), ("test-marker", 4.75)]


@pytest.mark.parametrize("status", (-1, 401, 403, 500))
def test_non_quota_failures_are_not_retried_or_treated_as_empty_results(status: int) -> None:
    clock: Final = Clock()
    reader, search = _reader((StreamingResponse(status_code=status, body=""), _page()), clock)

    with pytest.raises(pytest.fail.Exception, match=f"failed with HTTP {status}"):
        reader.events_for_query("test-marker")

    assert search.calls == [("test-marker", 30.0)]
    assert clock.elapsed == 0
