from typing import Final

from litellm.telemetry.histogram import LATENCY_BOUNDS_MS
from litellm.telemetry.records import (
    AttemptRecord,
    BlockCounts,
    BlockType,
    InstanceInfo,
    RequestRecord,
    StatusClass,
    TelemetryGroup,
    UIAction,
    UIEvent,
)
from litellm.telemetry.report import AttemptKey, AttemptMetrics, Report, RequestKey, RequestMetrics, report_to_json


def _metrics_of(*records: AttemptRecord) -> AttemptMetrics:
    metrics: Final = AttemptMetrics.empty()
    for record in records:
        metrics.add(record)
    return metrics


def test_attempt_and_ui_event_rows_serialize_with_merged_counts() -> None:
    first: Final = AttemptRecord(
        provider="bedrock",
        provider_status=StatusClass.SERVER_ERROR,
        stream=True,
        deployment_hash="abc",
        latency_to_first_token_ms=-1.0,
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
        attempts=((AttemptKey.of(first), _metrics_of(first, second)),),
        ui_events=((UIEvent(page="models", action=UIAction.CLICK, target="add_model"), 3),),
    )

    json_report: Final = report_to_json(report)
    assert isinstance(json_report, dict)
    assert json_report["attempts"] == [
        {
            "provider": "bedrock",
            "deployment_hash": "abc",
            "provider_status": "5xx",
            "stream": True,
            "attempt_count": 2,
            "latency_to_first_token_ms": {"counts": [0, 0, 1, *[0] * (len(LATENCY_BOUNDS_MS) - 2)], "invalid": 1},
        }
    ]
    assert json_report["ui_events"] == [{"page": "models", "action": "click", "target": "add_model", "count": 3}]


def test_block_types_and_header_keys_serialize_with_merged_counts() -> None:
    first: Final = RequestRecord(
        endpoint="/v1/messages",
        stream=False,
        litellm_status=StatusClass.SUCCESS,
        blocks=BlockCounts(total=3, by_type=((BlockType.TEXT, 2), (BlockType.IMAGE, 1))),
        header_keys=frozenset({"anthropic-beta", "x-custom"}),
    )
    second: Final = RequestRecord(
        endpoint="/v1/messages",
        stream=False,
        litellm_status=StatusClass.SUCCESS,
        blocks=BlockCounts(total=1, by_type=((BlockType.TEXT, 1),)),
        header_keys=frozenset({"anthropic-beta"}),
    )
    metrics: Final = RequestMetrics.empty()
    metrics.add(first)
    metrics.add(second)
    report: Final = Report(
        instance=InstanceInfo(instance_id="i", litellm_version="1.0.0", groups=frozenset(TelemetryGroup)),
        window_start=0.0,
        window_end=60.0,
        requests=((RequestKey.of(first), metrics),),
    )

    json_report: Final = report_to_json(report)
    assert isinstance(json_report, dict)
    (row,) = json_report["requests"]
    assert {key: row[key] for key in ("block_count", "block_types", "header_keys")} == {
        "block_count": {"counts": [1, 1, 0, 0, 0], "invalid": 0},
        "block_types": {"image": 1, "text": 3},
        "header_keys": {"anthropic-beta": 2, "x-custom": 1},
    }
