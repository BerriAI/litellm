"""Tests for the standard-reporter signal plumbing.

Two layers: the pure package/covers helpers, and an end-to-end check that the two
custom properties actually land in a pytest JUnit XML report via the real
`conftest.py::pytest_collection_modifyitems` hook. No `e2e` marker, so these run
without a proxy.
"""

from __future__ import annotations

import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from junit_properties import dedupe_covers, package_from_nodeid

E2E_DIR = Path(__file__).resolve().parent


@pytest.mark.parametrize(
    ("nodeid", "expected"),
    [
        pytest.param("logging/test_langfuse_e2e.py::TestX::test_y", "logging", id="suite-cwd"),
        pytest.param("tests/e2e/logging/test_langfuse_e2e.py::test_y", "logging", id="repo-root-prefix-stripped"),
        pytest.param("quota_management/spend_tracking/test_x.py::t", "quota_management", id="nested-uses-top-dir"),
        pytest.param("test_top.py::test_y", "root", id="top-level-file-is-root"),
        pytest.param("tests/e2e/test_top.py::test_y", "root", id="top-level-after-strip-is-root"),
        pytest.param("logging\\test_langfuse_e2e.py::t", "logging", id="windows-separators"),
        pytest.param("./logging/test_x.py::t", "logging", id="dot-segments-ignored"),
    ],
)
def test_package_from_nodeid(nodeid: str, expected: str) -> None:
    assert package_from_nodeid(nodeid) == expected


def test_dedupe_covers_flattens_dedupes_and_preserves_order() -> None:
    marker_args: tuple[tuple[object, ...], ...] = (("c.d", "a.b"), ("a.b",))
    assert dedupe_covers(marker_args) == ("c.d", "a.b")


def test_dedupe_covers_drops_empty_and_non_strings() -> None:
    marker_args: tuple[tuple[object, ...], ...] = (("a.b", "", 5, None, "c.d"),)
    assert dedupe_covers(marker_args) == ("a.b", "c.d")


def test_dedupe_covers_empty_input_is_empty() -> None:
    assert dedupe_covers(()) == ()


def _run_junit(tmp_path: Path, applications: int = 1) -> dict[str, ET.Element]:
    """Run a throwaway suite whose conftest wires attach_result_properties into
    user_properties exactly as tests/e2e/conftest.py does, then return its JUnit
    <testcase> elements keyed by test name. `applications` runs the collection
    hook that many times to exercise the idempotency guard."""
    (tmp_path / "pytest.ini").write_text("[pytest]\n")
    suite = tmp_path / "logging"
    suite.mkdir()
    (suite / "conftest.py").write_text(
        "from junit_properties import attach_result_properties\n"
        "\n"
        "def pytest_collection_modifyitems(items):\n"
        f"    for _ in range({applications}):\n"
        "        for item in items:\n"
        "            attach_result_properties(item)\n"
    )
    (suite / "test_sample.py").write_text(
        "import pytest\n"
        "\n"
        "@pytest.mark.covers('logging.langfuse.success.logs_spend', 'logging.langfuse.success.logs_spend')\n"
        "def test_pass():\n"
        "    pass\n"
        "\n"
        "@pytest.mark.covers('logging.s3.success.writes_object')\n"
        "def test_fail():\n"
        "    assert False\n"
        "\n"
        "@pytest.mark.skip(reason='demo')\n"
        "@pytest.mark.covers('logging.datadog.success.logs_spend')\n"
        "def test_skipped():\n"
        "    pass\n"
        "\n"
        "@pytest.fixture\n"
        "def boom():\n"
        "    raise RuntimeError('setup boom')\n"
        "\n"
        "@pytest.mark.covers('logging.otel.success.exports_metric')\n"
        "def test_setup_error(boom):\n"
        "    pass\n"
        "\n"
        "def test_no_covers():\n"
        "    pass\n"
    )
    report = tmp_path / "report.xml"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            str(suite),
            "-p",
            "no:cacheprovider",
            "-o",
            "addopts=",
            f"--junitxml={report}",
        ],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(E2E_DIR)},
        capture_output=True,
        text=True,
    )
    assert report.exists(), f"junit report not produced:\n{result.stdout}\n{result.stderr}"
    tree = ET.parse(report)
    return {testcase.attrib["name"]: testcase for testcase in tree.iter("testcase")}


def _properties(testcase: ET.Element) -> dict[str, str]:
    return {prop.attrib["name"]: prop.attrib["value"] for prop in testcase.iter("property")}


def _property_names(testcase: ET.Element) -> list[str]:
    return [prop.attrib["name"] for prop in testcase.iter("property")]


ALL_CASES = ("test_pass", "test_fail", "test_skipped", "test_setup_error", "test_no_covers")


def test_junit_report_carries_package_on_every_outcome(tmp_path: Path) -> None:
    cases = _run_junit(tmp_path)
    assert set(ALL_CASES) <= set(cases)
    for name in ALL_CASES:
        assert _properties(cases[name])["package"] == "logging", name


def test_junit_report_carries_deduped_covers(tmp_path: Path) -> None:
    cases = _run_junit(tmp_path)
    assert _properties(cases["test_pass"])["covers"] == "logging.langfuse.success.logs_spend"
    assert _properties(cases["test_fail"])["covers"] == "logging.s3.success.writes_object"
    assert _properties(cases["test_skipped"])["covers"] == "logging.datadog.success.logs_spend"
    assert _properties(cases["test_setup_error"])["covers"] == "logging.otel.success.exports_metric"
    assert _properties(cases["test_no_covers"])["covers"] == ""


def test_junit_report_records_standard_outcome(tmp_path: Path) -> None:
    cases = _run_junit(tmp_path)
    assert cases["test_pass"].find("failure") is None
    assert cases["test_pass"].find("skipped") is None
    assert cases["test_fail"].find("failure") is not None
    assert cases["test_skipped"].find("skipped") is not None
    assert cases["test_setup_error"].find("error") is not None


def test_properties_not_duplicated_when_hook_reapplied(tmp_path: Path) -> None:
    cases = _run_junit(tmp_path, applications=2)
    for name in ALL_CASES:
        names = _property_names(cases[name])
        assert names.count("package") == 1, (name, names)
        assert names.count("covers") == 1, (name, names)
