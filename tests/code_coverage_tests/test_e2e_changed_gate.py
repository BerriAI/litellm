import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Final

import pytest

GATE: Final = Path(__file__).resolve().parents[2] / ".github/e2e-stack/assert_tests_ran.py"
SECRETS_TO_ENV: Final = GATE.with_name("secrets_to_env.py")
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


@pytest.mark.parametrize("contents", ("<testsuite/>", "<testsuite", '<testsuite><testcase name="a"/></testsuite>'))
def test_missing_execution_evidence_fails(tmp_path: Path, contents: str) -> None:
    report: Final = tmp_path / "report.xml"
    _ = report.write_text(contents)

    result: Final = subprocess.run([sys.executable, str(GATE), str(report), *SELECTED], capture_output=True, text=True)

    assert result.returncode == 1


def test_short_values_are_written_without_masking_every_digit_in_the_log(tmp_path: Path) -> None:
    env_path: Final = tmp_path / ".env"

    result: Final = subprocess.run(
        [sys.executable, str(SECRETS_TO_ENV), str(env_path)],
        input='{"FLAG": "1", "API_KEY": "sk-0123456789abcdef"}',
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout == "::add-mask::sk-0123456789abcdef\n"
    assert env_path.read_text() == "FLAG='1'\nAPI_KEY='sk-0123456789abcdef'\n"
