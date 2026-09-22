"""Harness coverage for the per-replica RSS capture behind the memory tests.

No proxy needed and no ``e2e`` marker: this pins that a replica which gives no
usable /debug/memory/summary reading (unreachable, a non-2xx, or a summary with
no ram_usage_mb) surfaces as a named failure instead of silently dropping out of
the capture, since a budget check that only sees the replicas that answered would
pass on the ones it never measured.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Final

from e2e_http import NetworkError, Success, UnknownApiError
from memory_readings import RssReading, rss_capture
from models import MemorySummaryResponse, ProcessMemory


def _summary(
    pid: int, hostname: str | None, ram_usage_mb: float | None, error: str | None = None
) -> Success[MemorySummaryResponse]:
    return Success(
        status_code=200,
        data=MemorySummaryResponse(
            worker_pid=pid, hostname=hostname, status="ok", memory=ProcessMemory(ram_usage_mb=ram_usage_mb, error=error)
        ),
    )


class TestRssCapture:
    def test_every_answering_replica_becomes_a_reading_keyed_by_worker(self) -> None:
        capture: Final = rss_capture(
            MappingProxyType(
                {
                    "http://gateway-1:4000": _summary(7, "gateway-1", 512.5),
                    "http://gateway-2:4000": _summary(7, "gateway-2", 640.0),
                }
            )
        )
        assert capture.failures == ()
        assert capture.readings == (
            RssReading("http://gateway-1:4000", "gateway-1", 7, 512.5),
            RssReading("http://gateway-2:4000", "gateway-2", 7, 640.0),
        )
        assert len({reading.worker for reading in capture.readings}) == 2

    def test_replicas_without_a_usable_reading_are_named_failures_not_dropped(self) -> None:
        capture: Final = rss_capture(
            MappingProxyType(
                {
                    "http://gateway-1:4000": _summary(7, "gateway-1", 512.5),
                    "http://gateway-2:4000": NetworkError(message="connection refused"),
                    "http://gateway-3:4000": UnknownApiError(status_code=503, body="warming up"),
                    "http://gateway-4:4000": _summary(9, "gateway-4", None, error="psutil unavailable"),
                }
            )
        )
        assert tuple(reading.replica for reading in capture.readings) == ("http://gateway-1:4000",)
        assert len(capture.failures) == 3
        assert "http://gateway-2:4000" in capture.failures[0] and "connection refused" in capture.failures[0]
        assert "http://gateway-3:4000" in capture.failures[1] and "503" in capture.failures[1]
        assert "http://gateway-4:4000" in capture.failures[2] and "psutil unavailable" in capture.failures[2]

    def test_where_names_the_worker_for_a_failure_message(self) -> None:
        assert RssReading("http://gateway-1:4000", None, 7, 1.0).where == (
            "worker pid 7 on an unnamed host behind http://gateway-1:4000"
        )
