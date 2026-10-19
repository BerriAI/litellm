from __future__ import annotations

import html
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Final

import pytest
from pydantic import BaseModel, ConfigDict, TypeAdapter

ROOT: Final = Path(__file__).resolve().parents[2]
RENDERER: Final = ROOT / ".github/scripts/render_pytest_report.py"
SUITE: Final = """
from pathlib import Path
import pytest

def test_pass(ci_metric):
    ci_metric("Size", 41.2, "MB")

def test_fail():
    assert False, "permanent failure"

def test_skip():
    pytest.skip("not available")

@pytest.mark.xfail(reason="known bug")
def test_xfail():
    assert False

@pytest.mark.xfail(reason="fixed bug")
def test_xpass():
    pass

def test_retry():
    marker = Path(__file__).with_name("attempted")
    first = not marker.exists()
    marker.touch()
    assert not first, "retry me"

@pytest.fixture
def teardown_error():
    yield
    raise RuntimeError("cleanup failed")

def test_teardown(teardown_error):
    pass
"""


class ReportOutcome(BaseModel):
    model_config = ConfigDict(frozen=True)

    nodeid: str
    when: str
    category: str
    reason: str


class ReportMetric(BaseModel):
    model_config = ConfigDict(frozen=True)

    nodeid: str
    name: str
    value: str | float
    unit: str


class ReportData(BaseModel):
    model_config = ConfigDict(frozen=True)

    version: int
    outcomes: tuple[ReportOutcome, ...]
    metrics: tuple[ReportMetric, ...]


REPORT_DATA: Final = TypeAdapter(ReportData)


def run_suite(directory: Path, source: str = SUITE, *args: str) -> subprocess.CompletedProcess[str]:
    directory.mkdir(exist_ok=True)
    _ = (directory / "test_sample.py").write_text(source)
    return subprocess.run(
        (
            sys.executable,
            "-P",
            "-m",
            "pytest",
            "-c",
            "/dev/null",
            f"--rootdir={directory}",
            "--noconftest",
            "-p",
            "pytest_ci_report",
            "-p",
            "no:cacheprovider",
            "--color=no",
            "-rA",
            *args,
            "test_sample.py",
        ),
        cwd=directory,
        env={**os.environ, "PYTHONPATH": str(ROOT / "scripts"), "PYTEST_ADDOPTS": ""},
        capture_output=True,
        text=True,
        check=False,
        timeout=90,
    )


def render(path: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        (sys.executable, "-P", str(RENDERER), "--title", "CI outcomes", str(path)),
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )


@pytest.mark.parametrize("workers", ((), ("-n", "2")), ids=("serial", "xdist"))
def test_outcomes_match_pytest_and_metrics_survive_workers(tmp_path: Path, workers: tuple[str, ...]) -> None:
    report: Final = tmp_path / "report.json"
    result: Final = run_suite(
        tmp_path, SUITE, "--ci-report", str(report), "--reruns", "1", "--only-rerun", "retry me", *workers
    )
    assert result.returncode == 1, result.stdout + result.stderr
    data: Final = REPORT_DATA.validate_json(report.read_text())
    assert data.version == 1
    assert sorted((row.nodeid.split("::")[-1], row.when, row.category, row.reason) for row in data.outcomes) == [
        ("test_fail", "call", "failed", ""),
        ("test_pass", "call", "passed", ""),
        ("test_retry", "call", "passed", ""),
        ("test_skip", "call", "skipped", "Skipped: not available"),
        ("test_teardown", "call", "passed", ""),
        ("test_teardown", "teardown", "error", ""),
        ("test_xfail", "call", "xfailed", "known bug"),
        ("test_xpass", "call", "xpassed", "fixed bug"),
    ]
    assert data.metrics == (ReportMetric(nodeid="test_sample.py::test_pass", name="Size", value=41.2, unit="MB"),)
    assert tuple(tmp_path.glob("*.json")) == (report,)
    rendered: Final = render(report)
    assert rendered.returncode == 0, rendered.stderr
    terminal: Final = re.findall(
        r"\d+ (?:failed|passed|skipped|xfailed|xpassed|errors?)\b", result.stdout.splitlines()[-1]
    )
    assert ", ".join(terminal) in rendered.stdout
    assert "| test_sample.py | 3 | 1 | 1 | 1 | 1 | 1 |" in html.unescape(rendered.stdout)
    assert rendered.stdout.index("fixed bug") < rendered.stdout.index("known bug")


def test_maxfail_still_writes_report(tmp_path: Path) -> None:
    result: Final = run_suite(
        tmp_path, "def test_failure():\n    assert False\n", "--ci-report=report.json", "--maxfail=1"
    )
    assert result.returncode == 1
    data: Final = REPORT_DATA.validate_json((tmp_path / "report.json").read_text())
    assert data.outcomes == (
        ReportOutcome(nodeid="test_sample.py::test_failure", when="call", category="failed", reason=""),
    )


def test_report_is_opt_in(tmp_path: Path) -> None:
    result: Final = run_suite(tmp_path, "def test_pass(ci_metric):\n    ci_metric('Size', 1)\n")
    assert result.returncode == 0, result.stdout + result.stderr
    assert not tuple(tmp_path.glob("*.json"))


def test_unwritable_report_does_not_fail_tests(tmp_path: Path) -> None:
    result: Final = run_suite(tmp_path, "def test_pass():\n    pass\n", f"--ci-report={tmp_path}")
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("content", (None, "{", '{"version": 2}', '{"version": 1, "outcomes": null}'))
def test_unavailable_or_unknown_reports_are_non_gating(tmp_path: Path, content: str | None) -> None:
    report: Final = tmp_path / "report.json"
    if content is not None:
        _ = report.write_text(content)
    result: Final = render(report)
    assert result.returncode == 0, result.stderr
    assert "No report was produced" in result.stdout or "Unsupported report version" in result.stdout


@pytest.mark.parametrize("reason", ("known bug", "<tag>|\n" * 10000), ids=("entry-limit", "byte-limit"))
def test_details_and_large_cells_fit_summary_limit(tmp_path: Path, reason: str) -> None:
    report: Final = tmp_path / "report.json"
    _ = report.write_text(
        json.dumps(
            {
                "version": 1,
                "metrics": [],
                "outcomes": [
                    {"nodeid": f"test_sample.py::test_{index}", "when": "call", "category": "xfailed", "reason": reason}
                    for index in range(600)
                ],
            }
        )
    )
    result: Final = render(report)
    assert result.returncode == 0
    visible: Final = result.stdout.count("- xfailed:")
    assert visible <= 500
    assert f"and {600 - visible} more" in result.stdout
    assert len(result.stdout.encode()) < 1_000_000
    assert "<tag>" not in result.stdout


def test_metrics_are_compatible_with_junit(tmp_path: Path) -> None:
    result: Final = run_suite(
        tmp_path,
        "def test_pass(ci_metric):\n    ci_metric('Size', 1, 'MB')\n",
        "--ci-report=report.json",
        "--junitxml=report.xml",
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'name="ci_metric"' in (tmp_path / "report.xml").read_text()
    data: Final = REPORT_DATA.validate_json((tmp_path / "report.json").read_text())
    assert data.metrics[0].value == 1
