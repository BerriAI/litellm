from __future__ import annotations

from pathlib import Path
from typing import Final

import pytest

pytest_plugins = ("pytester",)

E2E_DIR: Final = Path(__file__).resolve().parents[2] / "e2e"

CONFTEST: Final = """
import pytest


@pytest.fixture
def setup(request, deployment):
    match request.node.callspec.id:
        case "vertex_ai":
            raise RuntimeError("could not register the model")
        case "azure_openai":
            raise AssertionError("the model never became servable")
    return deployment
"""
ROW: Final = """
import pytest
from codex.setups import require_feature, with_expected_failures

GAPS = {setup_id: "drops the tool" for setup_id in ("openai", "vertex_ai", "azure_openai", "claude_bedrock")}


@pytest.mark.parametrize("deployment", with_expected_failures(GAPS))
class TestCodexRow:
    def test_cell(self, request, setup):
        match request.node.callspec.id:
            case "openai":
                require_feature(False, "no web_search item")
            case "claude_bedrock":
                raise AssertionError("codex exited 101")
"""


def test_an_expected_failure_swallows_only_the_missing_feature(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PYTHONPATH", str(E2E_DIR))
    _ = pytester.makeconftest(CONFTEST)
    _ = (pytester.path / "test_row_e2e.py").write_text(ROW)

    run: Final = pytester.runpytest_subprocess("-p", "no:cacheprovider", "-p", "no:asyncio", "-rA")

    run.assert_outcomes(passed=1, failed=1, errors=2, skipped=1, xfailed=1)
    run.stdout.fnmatch_lines_random(
        [
            "XFAIL *test_cell?openai? - drops the tool",
            "FAILED *test_cell?claude_bedrock? - Assertion*",
            "ERROR *test_cell?azure_openai? - AssertionError*",
            "ERROR *test_cell?vertex_ai? - RuntimeError*",
        ]
    )
