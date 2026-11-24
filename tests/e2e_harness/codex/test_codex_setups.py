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
    if request.node.callspec.id == "vertex_ai":
        raise RuntimeError("could not register the model")
    return deployment
"""
ROW: Final = """
import pytest
from codex.setups import with_expected_failures


@pytest.mark.parametrize(
    "deployment", with_expected_failures({"openai": "drops the tool", "vertex_ai": "drops the tool"})
)
class TestCodexRow:
    def test_cell(self, request, setup):
        if request.node.callspec.id == "openai":
            raise AssertionError("no web_search item")
"""


def test_an_expected_failure_swallows_only_a_failed_assertion(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PYTHONPATH", str(E2E_DIR))
    _ = pytester.makeconftest(CONFTEST)
    _ = (pytester.path / "test_row_e2e.py").write_text(ROW)

    run: Final = pytester.runpytest_subprocess("-p", "no:cacheprovider", "-p", "no:asyncio", "-rA")

    run.assert_outcomes(passed=3, errors=1, skipped=1, xfailed=1)
    run.stdout.fnmatch_lines(["ERROR *test_cell?vertex_ai? - RuntimeError*"])
