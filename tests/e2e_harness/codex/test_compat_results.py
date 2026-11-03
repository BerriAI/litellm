from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Final

import pytest
from codex.compat_results import RESULTS_PATH_ENV, Failed, KnownGap, NotTested, Outcome, Passed, read_results
from codex.matrix_builder import build_matrix

pytest_plugins = ("pytester",)

E2E_DIR: Final = Path(__file__).resolve().parents[2] / "e2e"
COLUMNS: Final = ("openai", "vertex_ai", "azure_openai", "bedrock_runtime")
CONFTEST: Final = """
import pytest
from codex.compat_results import pytest_runtest_makereport as pytest_runtest_makereport


@pytest.fixture
def setup(deployment):
    if deployment == "unregistered":
        raise RuntimeError("could not register the model")
    return deployment
"""
ROW_FILES: Final = MappingProxyType(
    {
        "test_alpha_row_e2e.py": """
import pytest


@pytest.mark.parametrize(
    "deployment",
    (
        pytest.param("passes", id="openai"),
        pytest.param("breaks", id="vertex_ai", marks=pytest.mark.xfail(strict=True, reason="the gateway drops it")),
        pytest.param("passes", id="azure_openai", marks=pytest.mark.xfail(strict=True, reason="fixed upstream")),
        pytest.param("skips", id="bedrock_runtime"),
    ),
)
class TestCodexAlphaRow:
    def test_cell(self, deployment):
        if deployment == "breaks":
            raise AssertionError("no web_search item")
        if deployment == "skips":
            pytest.skip("no AWS credentials")
""",
        "test_beta_e2e.py": """
import pytest


@pytest.mark.parametrize(
    "deployment", (pytest.param("answers", id="openai"), pytest.param("unregistered", id="vertex_ai"))
)
class TestCodexBeta:
    def test_cell(self, setup):
        assert setup == "the marker", "the answer lacked the marker"
""",
        "test_flaky_e2e.py": """
import os
from pathlib import Path

import pytest


@pytest.mark.parametrize("deployment", (pytest.param("flaky", id="openai"),))
class TestCodexFlaky:
    def test_cell(self, deployment):
        first_attempt = Path(os.environ["FIRST_ATTEMPT_FLAG"])
        if not first_attempt.exists():
            first_attempt.write_text("done")
            raise AssertionError("the first attempt failed")
""",
        "test_helper.py": """
import pytest


@pytest.mark.parametrize("deployment", (pytest.param("passes", id="openai"),))
class TestCodexHelper:
    def test_cell(self, deployment):
        pass
""",
        "test_unparametrized_e2e.py": """
class TestCodexUnparametrized:
    def test_cell(self):
        pass


def test_outside_a_class():
    pass
""",
    }
)
EXPECTED_CELLS: Final[MappingProxyType[tuple[str, str], Outcome]] = MappingProxyType(
    {
        ("alpha_row", "openai"): Passed(),
        ("alpha_row", "vertex_ai"): KnownGap(reason="the gateway drops it"),
        ("alpha_row", "azure_openai"): Failed(error="[XPASS(strict)] fixed upstream"),
        ("alpha_row", "bedrock_runtime"): NotTested(reason="no AWS credentials"),
        ("beta", "openai"): Failed(error="AssertionError: the answer lacked the marker"),
        ("beta", "vertex_ai"): Failed(error="RuntimeError: could not register the model"),
        ("beta", "azure_openai"): NotTested(),
        ("beta", "bedrock_runtime"): NotTested(),
        ("flaky", "openai"): Passed(),
        ("flaky", "vertex_ai"): NotTested(),
        ("flaky", "azure_openai"): NotTested(),
        ("flaky", "bedrock_runtime"): NotTested(),
    }
)
RECORDED_PHASES: Final = 8
INNER_PLUGINS_OFF: Final = ("-p", "no:cacheprovider", "-p", "no:asyncio")


def _suite(pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch) -> Path:
    results: Final = pytester.path / "results" / "codex.jsonl"
    monkeypatch.setenv(RESULTS_PATH_ENV, str(results))
    monkeypatch.setenv("FIRST_ATTEMPT_FLAG", str(pytester.path / "first-attempt"))
    _ = pytester.makeconftest(CONFTEST)
    for name, source in ROW_FILES.items():
        _ = (pytester.path / name).write_text(source)
    return results


def _cells(results: Path) -> MappingProxyType[tuple[str, str], Outcome]:
    matrix: Final = build_matrix(
        read_results(results),
        columns=COLUMNS,
        litellm_version="1.0.0",
        codex_version="0.1.0",
        generated_at=datetime(2026, 1, 2, tzinfo=timezone.utc),
    )
    return MappingProxyType(
        {(row.id, column): outcome for row in matrix.features for column, outcome in row.providers.items()}
    )


def test_every_row_cell_records_its_outcome_and_a_rerun_keeps_the_last_attempt(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
) -> None:
    results: Final = _suite(pytester, monkeypatch)

    run: Final = pytester.runpytest_inprocess(*INNER_PLUGINS_OFF, "--reruns", "1", "--only-rerun", "first")

    run.assert_outcomes(passed=5, failed=2, errors=1, skipped=1, xfailed=1)
    assert run.parseoutcomes()["rerun"] == 1
    assert _cells(results) == EXPECTED_CELLS
    assert next(result.row_name for result in read_results(results) if result.row == "alpha_row") == "Alpha row"


def test_xdist_workers_record_each_attempt_once(pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch) -> None:
    results: Final = _suite(pytester, monkeypatch)
    monkeypatch.setenv("PYTHONPATH", str(E2E_DIR))

    run: Final = pytester.runpytest_subprocess(*INNER_PLUGINS_OFF, "-n", "2", "--reruns", "1", "--only-rerun", "first")

    run.assert_outcomes(passed=5, failed=2, errors=1, skipped=1, xfailed=1)
    assert run.parseoutcomes()["rerun"] == 1
    assert len(read_results(results)) == RECORDED_PHASES
    assert _cells(results) == EXPECTED_CELLS


def test_nothing_is_written_without_the_results_path(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
) -> None:
    results: Final = _suite(pytester, monkeypatch)
    monkeypatch.delenv(RESULTS_PATH_ENV)

    _ = pytester.runpytest_inprocess(*INNER_PLUGINS_OFF)

    assert not results.parent.exists()
