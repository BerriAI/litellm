from typing import Final

import pytest

from litellm.telemetry.consent import OFF, REQUIRES, TelemetryConsent
from litellm.telemetry.records import TelemetryGroup
from litellm.telemetry.report import report_to_json
from litellm.telemetry.sample import sample_report

_REQUEST_FIELDS: Final = {
    TelemetryGroup.REQUEST_SUCCESS: {
        "endpoint",
        "litellm_status",
        "handled_by_rust",
        "latency_to_first_byte_ms",
        "request_count",
    },
    TelemetryGroup.TOKEN_INFO: {"provider_cache_hit", "input_tokens", "cache_read_tokens"},
    TelemetryGroup.REQUEST_TAXONOMY: {"provider", "deployment_hash"},
    TelemetryGroup.EVENT_DETAILS: {"block_count", "block_types", "header_keys"},
}


def _with_requirements(group: TelemetryGroup) -> frozenset[TelemetryGroup]:
    required: Final = REQUIRES[group]
    parents: Final[frozenset[TelemetryGroup]] = _with_requirements(required) if required is not None else frozenset()
    return frozenset({group}) | parents


@pytest.mark.asyncio
async def test_off_builds_no_report() -> None:
    assert await sample_report(OFF, litellm_version="1.0.0") is None


@pytest.mark.asyncio
@pytest.mark.parametrize("group", list(_REQUEST_FIELDS))
async def test_each_request_group_adds_exactly_its_own_fields(group: TelemetryGroup) -> None:
    groups: Final = _with_requirements(group)
    with_group: Final = await sample_report(TelemetryConsent(groups), litellm_version="1.0.0")
    without_group: Final = await sample_report(TelemetryConsent(groups - {group}), litellm_version="1.0.0")
    assert with_group is not None and without_group is not None
    rows_with: Final = report_to_json(with_group)["requests"]
    assert isinstance(rows_with, list) and isinstance(rows_with[0], dict)
    rows_without: Final = report_to_json(without_group).get("requests", [{}])
    assert isinstance(rows_without, list) and isinstance(rows_without[0], dict)
    assert _REQUEST_FIELDS[group] <= set(rows_with[0]) - set(rows_without[0])


@pytest.mark.asyncio
async def test_heartbeat_only_sample_is_just_the_instance_header() -> None:
    report: Final = await sample_report(TelemetryConsent(frozenset({TelemetryGroup.HEARTBEAT})), litellm_version="2.0")
    assert report is not None
    assert report_to_json(report) == {
        "schema_version": 1,
        "instance": {"instance_id": "0" * 32, "litellm_version": "2.0", "groups": ["heartbeat"]},
        "window_start": 0.0,
        "window_end": 300.0,
        "dropped_records": 0,
    }


@pytest.mark.asyncio
async def test_without_taxonomy_rows_from_different_providers_with_the_same_status_merge() -> None:
    consent: Final = TelemetryConsent(_with_requirements(TelemetryGroup.TOKEN_INFO))
    report: Final = await sample_report(consent, litellm_version="1.0.0")
    assert report is not None
    assert len(report.requests) == 2
    assert all(key.provider is None for key, _ in report.requests)
    assert report.attempts == ()


@pytest.mark.asyncio
async def test_instance_configuration_adds_config_keys_and_page_navigation_adds_ui_events() -> None:
    consent: Final = TelemetryConsent(
        frozenset({TelemetryGroup.HEARTBEAT, TelemetryGroup.INSTANCE_CONFIGURATION, TelemetryGroup.PAGE_NAVIGATION})
    )
    report: Final = await sample_report(consent, litellm_version="1.0.0")
    assert report is not None
    json_report: Final = report_to_json(report)
    instance: Final = json_report["instance"]
    assert isinstance(instance, dict) and instance["config_keys"]
    assert json_report["ui_events"]
    assert "requests" not in json_report
