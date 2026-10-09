from typing import Final

from litellm.telemetry.histogram import LATENCY_BOUNDS_MS, Histogram
from litellm.telemetry.records import (
    AttemptRecord,
    InstanceInfo,
    StatusClass,
    TelemetryGroup,
    UIAction,
    UIEvent,
)
from litellm.telemetry.report import AttemptKey, AttemptMetrics, Report, report_to_json


def test_attempt_and_ui_event_rows_serialize_with_merged_counts() -> None:
    first: Final = AttemptRecord(
        provider="bedrock",
        provider_status=StatusClass.SERVER_ERROR,
        stream=True,
        deployment_hash="abc",
        latency_to_first_token_ms=None,
    )
    second: Final = AttemptRecord(
        provider="bedrock",
        provider_status=StatusClass.SERVER_ERROR,
        stream=True,
        deployment_hash="abc",
        latency_to_first_token_ms=200.0,
    )
    report: Final = Report(
        instance=InstanceInfo(instance_id="i", litellm_version="1.0.0", groups=frozenset(TelemetryGroup)),
        window_start=0.0,
        window_end=60.0,
        attempts=((AttemptKey.of(first), AttemptMetrics.of(first).merge(AttemptMetrics.of(second))),),
        ui_events=((UIEvent(page="models", action=UIAction.CLICK, target="add_model"), 3),),
    )

    first_token: Final = Histogram.of(LATENCY_BOUNDS_MS, 200.0)
    json_report: Final = report_to_json(report)
    assert isinstance(json_report, dict)
    assert json_report["attempts"] == [
        {
            "provider": "bedrock",
            "deployment_hash": "abc",
            "provider_status": "5xx",
            "stream": True,
            "attempt_count": 2,
            "latency_to_first_token_ms": list(first_token.counts),
        }
    ]
    assert json_report["ui_events"] == [{"page": "models", "action": "click", "target": "add_model", "count": 3}]
