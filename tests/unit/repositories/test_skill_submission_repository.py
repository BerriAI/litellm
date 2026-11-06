import json

import pytest

from litellm.repositories.skill_submission_repository import scan_passed_for_archive


@pytest.mark.parametrize("status,digest,report,expected", [
    ("completed", "current", {"complete": True, "risk_score": 0, "findings": []}, True),
    ("running", "current", {"complete": True, "risk_score": 0, "findings": []}, False),
    ("failed", "current", {"complete": True, "risk_score": 0, "findings": []}, False),
    ("completed", "previous", {"complete": True, "risk_score": 0, "findings": []}, False),
    ("completed", "current", {"complete": False, "risk_score": 0, "findings": []}, False),
    ("completed", "current", None, False),
    ("completed", "current", {"complete": True, "risk_score": 90, "findings": []}, False),
    ("completed", "current", {"complete": True, "risk_score": 0, "findings": [{"rule_id": "rule", "severity": "LOW", "message": "Issue", "file": "SKILL.md"}]}, False),
    ("completed", "current", {"complete": True, "risk_score": 0}, False),
])
def test_scan_pass_requires_clean_complete_report_for_published_archive(
    status: str, digest: str, report: dict[str, object] | None, expected: bool,
) -> None:
    assert scan_passed_for_archive(json.dumps({
        "id": "scan", "requested_by": "admin", "started_at": "2026-01-01T00:00:00Z",
        "expires_at": "2026-01-01T00:03:00Z", "status": status, "archive_sha256": digest,
        "report": None if report is None else {"scanner_version": "test", "scan_mode": "static-only", **report},
    }), "current") is expected


@pytest.mark.parametrize("record", [None, "", "{}", "null", "not-json"])
def test_missing_or_invalid_scan_is_not_reported_as_passed(record: str | None) -> None:
    assert scan_passed_for_archive(record, "current") is False
