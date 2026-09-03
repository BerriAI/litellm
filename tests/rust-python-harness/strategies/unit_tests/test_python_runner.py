from __future__ import annotations

import os
from pathlib import Path
from typing import Final

from .python_runner import BackendSpec, compare_python_runs, run_python_tests

HARNESS_ROOT: Final = Path(__file__).resolve().parents[4]


def _suite(root: Path, *, mismatch: bool = False) -> BackendSpec:
    (root / "pytest.ini").write_text("[pytest]\n")
    (root / "backend_probe.py").write_text(
        "import os\ndef selected():\n    return 'rust' if os.environ.get('TEST_USE_RUST') == '1' else 'python'\n"
    )
    (root / "test_backend.py").write_text(
        "import os\nfrom pathlib import Path\nfrom backend_probe import selected\n"
        "def test_backend():\n    backend = selected()\n"
        "    Path(backend + '.pid').write_text(str(os.getpid()))\n"
        + ("    assert backend == 'python'\n" if mismatch else "    assert backend in {'python', 'rust'}\n")
    )
    return BackendSpec(environment_variable="TEST_USE_RUST", probe="backend_probe:selected")


def test_runs_existing_python_tests_in_separate_verified_backends(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("PYTHONPATH", str(HARNESS_ROOT))
    monkeypatch.setenv("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
    spec: Final = _suite(tmp_path)
    python: Final = run_python_tests(("test_backend.py",), tmp_path, "python", spec)
    rust: Final = run_python_tests(("test_backend.py",), tmp_path, "rust", spec)
    assert compare_python_runs(python, rust) == ()
    assert python.tests == ("test_backend.py::test_backend",)
    assert (tmp_path / "python.pid").read_text() != (tmp_path / "rust.pid").read_text()
    assert (tmp_path / "python.pid").read_text() != str(os.getpid())


def test_rejects_wrong_backend_and_different_test_results(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("PYTHONPATH", str(HARNESS_ROOT))
    monkeypatch.setenv("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
    spec: Final = _suite(tmp_path, mismatch=True)
    python: Final = run_python_tests(("test_backend.py",), tmp_path, "python", spec)
    rust: Final = run_python_tests(("test_backend.py",), tmp_path, "rust", spec)
    assert "Python/Rust test outcomes differ" in compare_python_runs(python, rust)
    wrong: Final = run_python_tests(
        ("test_backend.py",),
        tmp_path,
        "rust",
        BackendSpec(environment_variable="WRONG_FLAG", probe=spec.probe),
    )
    assert wrong.exit_code == 1
    assert not wrong.verified
    assert "backend probe did not select rust" in wrong.problems[0]
