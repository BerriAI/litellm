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


@pytest.mark.parametrize("contents", ("<testsuite/>", "<testsuite", '<testsuite><testcase name="a"/></testsuite>'))
def test_missing_execution_evidence_fails(tmp_path: Path, contents: str) -> None:
    report: Final = tmp_path / "report.xml"
    _ = report.write_text(contents)

    result: Final = subprocess.run([sys.executable, str(GATE), str(report), *SELECTED], capture_output=True, text=True)

    assert result.returncode == 1
