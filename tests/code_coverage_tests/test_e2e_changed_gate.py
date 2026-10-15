import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Final

import pytest

GATE: Final = Path(__file__).resolve().parents[2] / ".github/e2e-stack/assert_tests_ran.py"
SELECTED: Final = ("tests/e2e/access_control/test_a.py", "tests/e2e/access_control/test_b.py")


@pytest.mark.parametrize(
    ("second_outcome", "expected_status"),
    (("passed", 0), ("skipped", 1), ("failure", 1), ("error", 1), ("deselected", 1)),
)
def test_each_changed_file_must_run(tmp_path: Path, second_outcome: str, expected_status: int) -> None:
    suite: Final = ET.Element("testsuite")
    _ = ET.SubElement(suite, "testcase", file=SELECTED[0])
    if second_outcome != "deselected":
        second: Final = ET.SubElement(suite, "testcase", file=SELECTED[1])
        if second_outcome != "passed":
            _ = ET.SubElement(second, second_outcome)
    report: Final = tmp_path / "report.xml"
    ET.ElementTree(suite).write(report)

    result: Final = subprocess.run([sys.executable, str(GATE), str(report), *SELECTED], capture_output=True, text=True)

    assert result.returncode == expected_status, result.stdout


@pytest.mark.parametrize("outcome", ("failure", "error"))
def test_passing_case_does_not_hide_a_failure_in_the_same_file(tmp_path: Path, outcome: str) -> None:
    suite: Final = ET.Element("testsuite")
    _ = ET.SubElement(suite, "testcase", file=SELECTED[0])
    failed: Final = ET.SubElement(suite, "testcase", file=SELECTED[0])
    _ = ET.SubElement(failed, outcome)
    report: Final = tmp_path / "report.xml"
    ET.ElementTree(suite).write(report)

    result: Final = subprocess.run(
        [sys.executable, str(GATE), str(report), SELECTED[0]], capture_output=True, text=True
    )

    assert result.returncode == 1


def test_failed_cases_are_named_per_selected_file(tmp_path: Path) -> None:
    suite: Final = ET.Element("testsuite")
    _ = ET.SubElement(suite, "testcase", file=SELECTED[0], classname="tests.e2e.access_control.test_a", name="test_ok")
    failed: Final = ET.SubElement(
        suite, "testcase", file=SELECTED[0], classname="tests.e2e.access_control.test_a", name="test_boom"
    )
    _ = ET.SubElement(failed, "failure", message="secret-bearing message")
    errored: Final = ET.SubElement(
        suite, "testcase", file=SELECTED[1], classname="tests.e2e.access_control.test_b", name="test_setup"
    )
    _ = ET.SubElement(errored, "error")
    report: Final = tmp_path / "report.xml"
    ET.ElementTree(suite).write(report)

    result: Final = subprocess.run([sys.executable, str(GATE), str(report), *SELECTED], capture_output=True, text=True)

    assert result.returncode == 1
    assert "  failed: tests.e2e.access_control.test_a::test_boom\n" in result.stdout
    assert "  failed: tests.e2e.access_control.test_b::test_setup\n" in result.stdout
    assert "test_ok" not in result.stdout
    assert "secret-bearing message" not in result.stdout


@pytest.mark.parametrize("contents", ("<testsuite/>", "<testsuite", '<testsuite><testcase name="a"/></testsuite>'))
def test_missing_execution_evidence_fails(tmp_path: Path, contents: str) -> None:
    report: Final = tmp_path / "report.xml"
    _ = report.write_text(contents)

    result: Final = subprocess.run([sys.executable, str(GATE), str(report), *SELECTED], capture_output=True, text=True)

    assert result.returncode == 1


@pytest.mark.parametrize("omitted_role", ("proxy_admin", "team_member", "internal_user_viewer"))
def test_one_passing_management_case_cannot_hide_a_missing_actor(tmp_path: Path, omitted_role: str) -> None:
    suite: Final = ET.Element("testsuite")
    path: Final = "tests/e2e/management/test_jwt_management_e2e.py"
    case: Final = ET.SubElement(suite, "testcase", file=path)
    properties: Final = ET.SubElement(case, "properties")
    _ = ET.SubElement(
        properties,
        "property",
        name="management_node",
        value=f"{path}::TestJwtManagement::test_actor_subject_and_database_role[proxy_admin_viewer]",
    )
    report: Final = tmp_path / "report.xml"
    ET.ElementTree(suite).write(report)
    result: Final = subprocess.run([sys.executable, str(GATE), str(report), path], capture_output=True, text=True)
    assert result.returncode == 1
    assert f"test_actor_subject_and_database_role[{omitted_role}]" in result.stdout


@pytest.mark.parametrize("phase", ("setup", "call", "teardown"))
@pytest.mark.parametrize("required_count", ("1", "4"))
def test_oauth_failure_diagnostics_do_not_publish_private_payloads(
    tmp_path: Path, phase: str, required_count: str
) -> None:
    suite: Final = ET.Element("testsuite")
    case: Final = ET.SubElement(suite, "testcase", file=SELECTED[0])
    private: Final = "private-token-in-exception-message"
    failure: Final = ET.SubElement(case, "failure", message=private)
    failure.text = private
    properties: Final = ET.SubElement(case, "properties")
    for name, value in (
        ("oauth_failure_phase", phase),
        ("oauth_exception_type", "AssertionError"),
        ("oauth_frame", "oauth_gateway.py:120:start"),
        ("oauth_frame", f"injected\\n{private}"),
        ("unrelated_property", private),
    ):
        _ = ET.SubElement(properties, "property", name=name, value=value)
    report: Final = tmp_path / "report.xml"
    ET.ElementTree(suite).write(report)
    result: Final = subprocess.run(
        [sys.executable, "-I", str(GATE), str(report), SELECTED[0]],
        capture_output=True,
        text=True,
        env={**os.environ, "E2E_REQUIRED_TEST_COUNT": required_count},
    )
    assert result.returncode == 1
    assert f"oauth_failure_phase: {phase}" in result.stdout
    assert "oauth_exception_type: AssertionError" in result.stdout
    assert "oauth_frame: oauth_gateway.py:120:start" in result.stdout
    assert private not in result.stdout + result.stderr


@pytest.mark.parametrize(
    ("count", "skip", "expected"), ((0, False, 1), (3, False, 1), (4, False, 0), (5, False, 1), (4, True, 1))
)
def test_required_count_reports_cases_before_rejecting(tmp_path: Path, count: int, skip: bool, expected: int) -> None:
    suite = ET.Element("testsuite")
    for index in range(count):
        case = ET.SubElement(suite, "testcase", file=SELECTED[0], classname="OAuth", name=f"variant{index}")
        if skip and index == 0:
            ET.SubElement(case, "skipped", message="private-skip-reason")
    report = tmp_path / "report.xml"
    ET.ElementTree(suite).write(report)
    result = subprocess.run(
        [sys.executable, "-I", str(GATE), str(report), SELECTED[0]],
        env={**os.environ, "E2E_REQUIRED_TEST_COUNT": "4"},
        capture_output=True,
        text=True,
    )
    assert result.returncode == expected
    assert f"{count} collected, {int(skip)} skipped" in result.stdout
    if skip:
        assert "skipped: OAuth::variant0" in result.stdout
    assert "private-skip-reason" not in result.stdout + result.stderr
