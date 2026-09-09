import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Final

import pytest

GATE: Final = Path(__file__).resolve().parents[2] / ".github/e2e-stack/assert_tests_ran.py"
SECRETS_TO_ENV: Final = GATE.with_name("secrets_to_env.py")
SELECT_TESTS: Final = GATE.with_name("select_tests.py")
CANARY: Final = ("tests/e2e/access_control/test_a.py", "tests/e2e/access_control/test_b.py")
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


def select_tests(changed: tuple[str, ...]) -> tuple[str, ...]:
    result: Final = subprocess.run(
        [sys.executable, str(SELECT_TESTS), *CANARY],
        input="".join(f"{path}\n" for path in changed),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    return tuple(result.stdout.split())


@pytest.mark.parametrize(
    ("changed", "expected"),
    (
        (("tests/e2e/logging/test_datadog_e2e.py", "litellm/router.py"), ("tests/e2e/logging/test_datadog_e2e.py",)),
        (("tests/e2e/ui/test_keys.py", "tests/e2e/claude_code/test_cli.py", "tests/e2e/load/test_burst.py"), ()),
        (("tests/e2e/batches/test_managed_files_enforcement_e2e.py",), ()),
        (("tests/e2e/guardrails/test_presidio_masking_e2e.py",), ()),
        (("tests/e2e/llm_translation/realtime/test_realtime_pipecat_audio_e2e.py",), ()),
        (
            ("tests/e2e/llm_translation/realtime/test_realtime_e2e.py",),
            ("tests/e2e/llm_translation/realtime/test_realtime_e2e.py",),
        ),
        (
            ("tests/e2e/guardrails/test_bedrock_guardrail_e2e.py",),
            ("tests/e2e/guardrails/test_bedrock_guardrail_e2e.py",),
        ),
        (("tests/e2e/logging/helpers.py", "docs/my-website/docs/index.md", "tests/e2e/CLAUDE.md"), ()),
        (
            ("tests/e2e/logging/test_datadog_e2e.py", "tests/e2e/logging/test_datadog_e2e.py"),
            ("tests/e2e/logging/test_datadog_e2e.py",),
        ),
    ),
)
def test_changed_suite_files_are_selected_unless_the_stack_cannot_run_them(
    changed: tuple[str, ...], expected: tuple[str, ...]
) -> None:
    assert select_tests(changed) == expected


@pytest.mark.parametrize(
    "harness_file",
    (
        "tests/e2e/proxy_client.py",
        "tests/e2e/conftest.py",
        "tests/e2e/pytest.ini",
        "tests/e2e/gateway/stage_mirror_ci_config.yml",
        ".github/e2e-stack/up.sh",
        ".github/workflows/test-e2e-changed.yml",
    ),
)
def test_harness_changes_run_the_canary_suite(harness_file: str) -> None:
    assert select_tests((harness_file, "litellm/router.py")) == CANARY


def test_a_changed_canary_file_is_selected_once_alongside_a_harness_change() -> None:
    assert select_tests((CANARY[1], "tests/e2e/proxy_client.py")) == CANARY


def test_the_canary_joins_directly_selected_files_in_sorted_order() -> None:
    assert select_tests(("tests/e2e/logging/test_datadog_e2e.py", ".github/e2e-stack/up.sh")) == (
        *CANARY,
        "tests/e2e/logging/test_datadog_e2e.py",
    )


def test_a_harness_unit_test_change_runs_itself_and_the_canary() -> None:
    assert select_tests(("tests/e2e/test_proxy_client.py",)) == (*CANARY, "tests/e2e/test_proxy_client.py")


def test_a_canary_argument_the_shell_never_expanded_fails_the_selector() -> None:
    result: Final = subprocess.run(
        [sys.executable, str(SELECT_TESTS), "tests/e2e/access_control/test_*.py"],
        input="tests/e2e/proxy_client.py\n",
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "tests/e2e/access_control/test_*.py" in result.stderr
    assert result.stdout == ""


@pytest.mark.parametrize(
    ("secrets", "offender", "unprintable"),
    (
        ('{"AWS_ACCESS_KEY_ID": "AKIAEXAMPLE", "BAD-NAME": "shibboleth"}', "BAD-NAME", "shibboleth"),
        ("""{"AWS_SECRET_ACCESS_KEY": "quote'shibboleth"}""", "AWS_SECRET_ACCESS_KEY", "shibboleth"),
        ('{"DD_API_KEY": "line\\nshibboleth"}', "DD_API_KEY", "shibboleth"),
    ),
)
def test_an_unusable_secret_is_named_without_printing_its_value(
    tmp_path: Path, secrets: str, offender: str, unprintable: str
) -> None:
    env_path: Final = tmp_path / ".env"

    result: Final = subprocess.run(
        [sys.executable, str(SECRETS_TO_ENV), str(env_path)], input=secrets, capture_output=True, text=True
    )

    assert result.returncode == 1
    assert offender in result.stderr
    assert unprintable not in result.stderr
    assert result.stdout == ""
    assert not env_path.exists()
