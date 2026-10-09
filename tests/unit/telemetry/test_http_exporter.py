import json
from collections.abc import Mapping, Sequence
from typing import Final

import httpx
import pytest

from litellm.telemetry.histogram import ATTEMPT_BOUNDS, BLOCK_COUNT_BOUNDS, LATENCY_BOUNDS_MS
from litellm.telemetry.records import InstanceInfo, RequestRecord, StatusClass, TelemetryGroup
from litellm.telemetry.report import Report, RequestKey, RequestMetrics
from litellm.telemetry.http_exporter import HttpExporter
from litellm.telemetry.sink import ExportOutcome

_RECORD: Final = RequestRecord(
    endpoint="/v1/messages",
    stream=True,
    litellm_status=StatusClass.SUCCESS,
    provider="anthropic",
    provider_status=StatusClass.SUCCESS,
    latency_to_first_byte_ms=120.0,
)
_REPORT: Final = Report(
    instance=InstanceInfo(instance_id="i", litellm_version="1.2.3", groups=frozenset(TelemetryGroup)),
    window_start=10.0,
    window_end=70.0,
    requests=((RequestKey.of(_RECORD), RequestMetrics.of(_RECORD)),),
)


def _histogram_json(bounds: tuple[float, ...], hit_index: int | None) -> Sequence[int]:
    return [int(index == hit_index) for index in range(len(bounds) + 1)]


@pytest.mark.asyncio
async def test_posts_the_report_as_json() -> None:
    seen: Final[list[httpx.Request]] = []  # mutable-ok: captures what the transport received

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(202)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        outcome: Final = await HttpExporter(client, "https://telemetry.example/v1/reports").export(_REPORT)

    assert outcome is ExportOutcome.SENT
    (request,) = seen
    assert request.url == "https://telemetry.example/v1/reports"
    assert request.headers["content-type"] == "application/json"
    assert json.loads(request.content) == {
        "schema_version": 1,
        "instance": {
            "instance_id": "i",
            "litellm_version": "1.2.3",
            "groups": sorted(group.value for group in TelemetryGroup),
            "config_keys": [],
        },
        "window_start": 10.0,
        "window_end": 70.0,
        "dropped_records": 0,
        "requests": [
            {
                "endpoint": "/v1/messages",
                "provider": "anthropic",
                "deployment_hash": None,
                "litellm_status": "2xx",
                "provider_status": "2xx",
                "litellm_cache_hit": False,
                "handled_by_rust": False,
                "provider_cache_hit": False,
                "stream": True,
                "request_count": 1,
                "input_tokens": 0,
                "output_tokens": 0,
                "cache_read_tokens": 0,
                "block_count": _histogram_json(BLOCK_COUNT_BOUNDS, None),
                "block_types": {},
                "header_keys": {},
                "provider_attempts": _histogram_json(ATTEMPT_BOUNDS, 0),
                "latency_to_headers_ms": _histogram_json(LATENCY_BOUNDS_MS, None),
                "latency_to_first_byte_ms": _histogram_json(LATENCY_BOUNDS_MS, 2),
            }
        ],
        "attempts": [],
        "ui_events": [],
    }


@pytest.mark.parametrize(
    ("status_code", "expected"),
    [
        (200, ExportOutcome.SENT),
        (400, ExportOutcome.REJECTED),
        (413, ExportOutcome.REJECTED),
        (429, ExportOutcome.RETRY),
        (503, ExportOutcome.RETRY),
    ],
)
@pytest.mark.asyncio
async def test_status_codes_map_to_outcomes(status_code: int, expected: ExportOutcome) -> None:
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(status_code))) as client:
        assert await HttpExporter(client, "https://telemetry.example").export(_REPORT) is expected


@pytest.mark.asyncio
async def test_a_transport_error_is_retryable_instead_of_raising() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("unreachable", request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        assert await HttpExporter(client, "https://telemetry.example").export(_REPORT) is ExportOutcome.RETRY
