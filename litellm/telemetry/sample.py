"""A made-up report, built through the real gate and aggregator, that shows what each group adds"""

from typing import Final

from litellm.telemetry.aggregate import AggregatingSink
from litellm.telemetry.consent import ConsentGatedSink, TelemetryConsent
from litellm.telemetry.records import (
    AttemptRecord,
    BlockCounts,
    BlockType,
    InstanceInfo,
    RequestRecord,
    StatusClass,
    TokenCounts,
    UIAction,
    UIEvent,
)
from litellm.telemetry.report import Report
from litellm.telemetry.sink import ExportOutcome

SAMPLE_WINDOW_S: Final = 300.0

_REQUESTS: Final = (
    RequestRecord(
        endpoint="/chat/completions",
        stream=True,
        litellm_status=StatusClass.SUCCESS,
        provider="openai",
        deployment_hash="3f9a1c0d2b7e4a61",
        provider_status=StatusClass.SUCCESS,
        provider_cache_hit=True,
        provider_attempts=1,
        tokens=TokenCounts(input=1200, output=340, cache_read=1024),
        latency_to_headers_ms=210.0,
        latency_to_first_byte_ms=420.0,
        blocks=BlockCounts(total=3, by_type=((BlockType.TEXT, 2), (BlockType.IMAGE, 1))),
        header_keys=frozenset({"x-stainless-lang", "x-request-id"}),
    ),
    RequestRecord(
        endpoint="/messages",
        stream=False,
        litellm_status=StatusClass.SERVER_ERROR,
        provider="anthropic",
        deployment_hash="8c2e5b9f01d34a77",
        provider_status=StatusClass.SERVER_ERROR,
        handled_by_rust=True,
        provider_attempts=3,
        latency_to_headers_ms=9150.0,
        latency_to_first_byte_ms=9180.0,
        blocks=BlockCounts(total=2, by_type=((BlockType.TEXT, 1), (BlockType.TOOL_RESULT, 1))),
        header_keys=frozenset({"anthropic-version", "anthropic-beta"}),
    ),
)
_ATTEMPTS: Final = (
    AttemptRecord(
        provider="openai",
        provider_status=StatusClass.SUCCESS,
        stream=True,
        deployment_hash="3f9a1c0d2b7e4a61",
        latency_to_first_token_ms=380.0,
    ),
    *(
        AttemptRecord(
            provider="anthropic",
            provider_status=StatusClass.SERVER_ERROR,
            stream=False,
            deployment_hash="8c2e5b9f01d34a77",
        )
        for _ in range(3)
    ),
)
_UI_EVENTS: Final = (
    UIEvent(page="models", action=UIAction.VIEW),
    UIEvent(page="playground", action=UIAction.CLICK, target="tab=compare"),
)


class _Capture:
    def __init__(self) -> None:
        self.report: Report | None = None

    async def export(self, report: Report) -> ExportOutcome:
        self.report = report
        return ExportOutcome.SENT


async def sample_report(consent: TelemetryConsent, *, litellm_version: str) -> Report | None:
    """``None`` when the consent sends nothing at all"""
    capture: Final = _Capture()
    times: Final = iter((0.0, SAMPLE_WINDOW_S))
    sink: Final = ConsentGatedSink(AggregatingSink(capture, clock=lambda: next(times)), consent)
    sink.set_instance(
        InstanceInfo(
            instance_id="00000000000000000000000000000000",
            litellm_version=litellm_version,
            config_keys=frozenset({"litellm_settings.cache", "router_settings.routing_strategy"}),
        )
    )
    for request in _REQUESTS:
        sink.record_request(request)
    for attempt in _ATTEMPTS:
        sink.record_attempt(attempt)
    for event in _UI_EVENTS:
        sink.record_ui_event(event)
    await sink.flush()
    return capture.report
