from __future__ import annotations

from pathlib import Path
from typing import Final

import pytest
from codex.compat_results import RESULTS_PATH_ENV

pytest_plugins = ("pytester",)

E2E_DIR: Final = Path(__file__).resolve().parents[2] / "e2e"
CODEX_CONFTEST: Final = E2E_DIR / "codex" / "conftest.py"
E2E_PYTEST_INI: Final = E2E_DIR / "pytest.ini"
FAILS_ONCE: Final = """
from pathlib import Path


def test_cell():
    attempted = Path(__file__).with_suffix(".attempted")
    if not attempted.exists():
        attempted.write_text("done")
        raise {error}
"""
SUITE_FILES: Final = (
    ("suite/test_assertion_e2e.py", 'AssertionError("the answer lacked the marker")'),
    ("suite/test_gateway_error_e2e.py", 'RuntimeError("status_code=502 from the gateway")'),
    ("suite/test_crash_e2e.py", 'RuntimeError("codex crashed")'),
    ("outside/test_outside_e2e.py", 'AssertionError("the answer lacked the marker")'),
)


def test_a_codex_test_is_rerun_once_after_a_failed_assertion_or_a_configured_error(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(RESULTS_PATH_ENV, raising=False)
    monkeypatch.setenv("PYTHONPATH", str(E2E_DIR))
    (pytester.path / "suite").mkdir()
    (pytester.path / "outside").mkdir()
    _ = (pytester.path / "pytest.ini").write_text(E2E_PYTEST_INI.read_text())
    _ = (pytester.path / "suite" / "conftest.py").write_text(CODEX_CONFTEST.read_text())
    for name, error in SUITE_FILES:
        _ = (pytester.path / name).write_text(FAILS_ONCE.format(error=error))

    run: Final = pytester.runpytest_subprocess(
        "-p", "no:cacheprovider", "-p", "no:asyncio", "-p", "no:pytest-retry", "-rA"
    )

    run.assert_outcomes(passed=2, failed=2)
    assert run.parseoutcomes()["rerun"] == 2
    run.stdout.fnmatch_lines_random(
        [
            "PASSED suite/test_assertion_e2e.py::test_cell",
            "PASSED suite/test_gateway_error_e2e.py::test_cell",
            "FAILED suite/test_crash_e2e.py::test_cell - RuntimeError*",
            "FAILED outside/test_outside_e2e.py::test_cell - AssertionError*",
        ]
    )
