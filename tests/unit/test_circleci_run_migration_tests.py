import importlib.util
from pathlib import Path
from typing import Final

import pytest

SCRIPT: Final = Path(__file__).resolve().parents[2] / ".circleci/scripts/run_migration_tests.py"
SPEC: Final = importlib.util.spec_from_file_location("migration_ci", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE: Final = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


@pytest.mark.parametrize(
    "xml,expected,exit_code,passed",
    (
        ('<testsuites><testsuite><testcase name="a"/><testcase name="b"/></testsuite></testsuites>', 2, 0, True),
        ('<testsuites><testsuite><testcase name="a"/></testsuite></testsuites>', 2, 0, False),
        ("<testsuite><testcase><failure/></testcase></testsuite>", 1, 0, False),
        ("<testsuite><testcase><error/></testcase></testsuite>", 1, 0, False),
        ("<testsuite><testcase><skipped/></testcase></testsuite>", 1, 0, False),
        ("<testsuite><testcase/></testsuite>", 1, 1, False),
        ("<testsuite><testcase/></testsuite>", 1, 5, False),
        ("<testsuite/>", 1, 0, False),
        ("<broken", 1, 0, False),
        (None, 1, 0, False),
        ('<testsuite><testcase name="a"/><testcase name="a"/></testsuite>', 2, 0, False),
    ),
)
def test_only_a_complete_passing_suite_can_certify_an_image(
    tmp_path: Path, xml: str | None, expected: int, exit_code: int, passed: bool
) -> None:
    path: Final = tmp_path / "results.xml"
    if xml is not None:
        path.write_text(xml)
    assert MODULE.successful_junit(path, expected, exit_code) is passed
