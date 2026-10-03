import hashlib
import json
import os
from pathlib import Path
from typing import Final
from unittest.mock import Mock, patch

import pytest
from tests.integration._support.conformance import read_checks, verify_archive, prefixed_request
from pydantic import ValidationError


@pytest.mark.parametrize("explicit", (False, True))
def test_owned_proxy_isolates_automatic_coverage_unless_requested(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, explicit: bool
) -> None:
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2]))
    from integration._support.client import Gateway
    from integration._support.process import owned_proxy_process

    monkeypatch.setenv("COVERAGE_PROCESS_CONFIG", "parent-config")
    monkeypatch.setenv("COVERAGE_PROCESS_START", "parent.toml")
    overrides: Final = {"COVERAGE_PROCESS_CONFIG": "child-config"} if explicit else {}
    gateway: Final = Gateway(Mock(), "owner-key", "http://upstream")
    with (
        patch("integration._support.process.subprocess.Popen") as spawn,
        patch("integration._support.process.httpx.Client") as client,
        patch("integration._support.process.group_members", return_value=()),
    ):
        spawn.return_value.poll.return_value = None
        client.return_value.__enter__.return_value.get.return_value.status_code = 200
        with owned_proxy_process(gateway, tmp_path, overrides):
            child: Final = spawn.call_args.kwargs["env"]
            assert child["COVERAGE_PROCESS_CONFIG"] == ("child-config" if explicit else "")
            assert child["COVERAGE_PROCESS_START"] == ""
            assert child["LITELLM_MASTER_KEY"] == gateway.key
    assert os.environ["COVERAGE_PROCESS_CONFIG"] == "parent-config"


def test_changed_archive_is_rejected(tmp_path: Path) -> None:
    archive: Final = tmp_path / "reference.tar.gz"
    archive.write_bytes(b"changed reference")
    with pytest.raises(AssertionError, match="SHA-256"):
        verify_archive(archive, hashlib.sha256(b"approved reference").hexdigest())


def test_matching_archive_is_accepted(tmp_path: Path) -> None:
    archive: Final = tmp_path / "reference.tar.gz"
    archive.write_bytes(b"approved reference")
    assert verify_archive(archive, "5a870d3ee9520e3a912c735d007d7b40f6d3cbf8b61d909ebb8361f423d4ba1e") is None


@pytest.mark.parametrize(
    "checks",
    (
        [],
        [{"id": "tools-list", "status": "FAILURE"}],
        [{"id": "tools-list", "status": "WARNING"}],
        [{"id": "wire-schema", "status": "SUCCESS"}],
        [{"id": "tools-list", "status": "SUCCESS"}, {"id": "wire-schema", "status": "FAILURE"}],
        [{"id": "tools-list", "status": "SUCCESS"}, {"id": "tools-list", "status": "SUCCESS"}],
    ),
)
def test_incomplete_or_failed_scenario_cannot_pass(tmp_path: Path, checks: list[dict[str, str]]) -> None:
    (tmp_path / "checks.json").write_text(json.dumps(checks))
    with pytest.raises(AssertionError, match="conformance"):
        read_checks(tmp_path, "tools-list")


def test_skipped_scenario_without_a_report_cannot_pass(tmp_path: Path) -> None:
    with pytest.raises(AssertionError, match="conformance"):
        read_checks(tmp_path, "tools-list")


def test_multiple_reports_cannot_supply_a_stale_pass(tmp_path: Path) -> None:
    for name in ("previous", "current"):
        directory: Final = tmp_path / name
        directory.mkdir()
        (directory / "checks.json").write_text('[{"id":"tools-list","status":"SUCCESS"}]')
    with pytest.raises(AssertionError, match="conformance"):
        read_checks(tmp_path, "tools-list")


def test_passing_scenario_keeps_nonbinding_diagnostics(tmp_path: Path) -> None:
    (tmp_path / "checks.json").write_text(
        json.dumps(
            [
                {"id": identity, "status": "INFO" if identity == "advisory" else "SUCCESS"}
                for identity in ("tools-list", "tools-name-format", "wire-schema-valid", "advisory")
            ]
        )
    )
    checks: Final = read_checks(tmp_path, "tools-list")
    assert checks[-1].id == "advisory" and checks[-1].status == "INFO"


def test_unrecognized_result_status_cannot_pass(tmp_path: Path) -> None:
    (tmp_path / "checks.json").write_text('[{"id":"tools-list","status":"SKIPPED"}]')
    with pytest.raises(ValidationError, match="status"):
        read_checks(tmp_path, "tools-list")


@pytest.mark.parametrize(
    "result",
    (
        {"content": [{"type": "text", "text": "Tool not found"}], "isError": True},
        {"content": [{"type": "text", "text": "wrong upstream payload"}]},
    ),
)
def test_official_text_success_cannot_hide_an_error_or_wrong_payload(tmp_path: Path, result: dict[str, object]) -> None:
    (tmp_path / "checks.json").write_text(
        json.dumps([{"id": "tools-call-simple-text", "status": "SUCCESS", "details": {"result": result}}])
    )
    with pytest.raises(AssertionError, match="reference payload"):
        read_checks(tmp_path, "tools-call-simple-text")


def test_official_text_success_requires_the_reference_payload(tmp_path: Path) -> None:
    (tmp_path / "checks.json").write_text(
        json.dumps(
            [
                {
                    "id": "tools-call-simple-text",
                    "status": "SUCCESS",
                    "details": {
                        "result": {
                            "content": [{"type": "text", "text": "This is a simple text response for testing."}],
                            "isError": False,
                        }
                    },
                }
            ]
        )
    )
    assert read_checks(tmp_path, "tools-call-simple-text")[0].status == "SUCCESS"


@pytest.mark.parametrize("method", ("tools/call", "prompts/get"))
def test_only_fixture_name_is_translated_to_advertised_name(method: str) -> None:
    request: Final = {
        "jsonrpc": "2.0",
        "id": 4,
        "method": method,
        "params": {"name": "test_tool", "_meta": {"progressToken": "keep"}},
    }
    changed: Final = json.loads(prefixed_request(json.dumps(request).encode(), "official"))
    assert changed == {**request, "params": {"name": "official-test_tool", "_meta": {"progressToken": "keep"}}}
    assert "arguments" not in changed["params"]


@pytest.mark.parametrize(
    "body",
    (
        b"malformed JSON",
        b"[]",
        b'{"method":"initialize","params":{"protocolVersion":"2025-03-26"}}',
        b'{"method":"tools/call","params":{"name":17}}',
    ),
)
def test_name_integration_preserves_other_requests(body: bytes) -> None:
    assert prefixed_request(body, "official") == body


@pytest.mark.parametrize(
    "scenario,identities",
    (
        ("server-initialize", ("server-initialize", "server-session-id-visible-ascii", "wire-schema-valid")),
        ("tools-list", ("tools-list", "tools-name-format", "wire-schema-valid")),
        ("tools-call-image", ("tools-call-image", "wire-schema-valid")),
    ),
)
def test_missing_secondary_checks_cannot_report_complete_conformance(
    tmp_path: Path, scenario: str, identities: tuple[str, ...]
) -> None:
    report: Final = tmp_path / "checks.json"
    for missing in identities:
        report.write_text(
            json.dumps([{"id": identity, "status": "SUCCESS"} for identity in identities if identity != missing])
        )
        with pytest.raises(AssertionError, match="conformance"):
            read_checks(tmp_path, scenario)
    report.write_text(json.dumps([{"id": identity, "status": "SUCCESS"} for identity in identities]))
    assert len(read_checks(tmp_path, scenario)) == len(identities)
