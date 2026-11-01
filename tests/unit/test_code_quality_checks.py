import importlib.util
from pathlib import Path

_MODULE_PATH = Path(__file__).resolve().parents[2] / "scripts" / "code_quality_checks.py"
_spec = importlib.util.spec_from_file_location("code_quality_checks", _MODULE_PATH)
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)

WORKFLOW_SOURCE = """
jobs:
  python:
    steps:
      - run: uv run --no-sync python ./tests/code_coverage_tests/from_another_job.py
  code-quality:
    steps:
      - uses: actions/checkout@v4
      - name: recursive_detector
        run: uv run --no-sync python ./tests/code_coverage_tests/recursive_detector.py
      - run: uv run --no-sync pytest -q tests/code_coverage_tests/test_thing.py
      - run: uv run --no-sync python ./tests/documentation_tests/test_env_keys.py
      - run: uv run --no-sync python ./tests/code_coverage_tests/check_licenses.py
      - run: |
          uv run --no-sync python ./tests/code_coverage_tests/router_code_coverage.py
"""


def test_selects_the_code_quality_jobs_scripts_that_leave_the_tree_untouched():
    assert mod.scripts(WORKFLOW_SOURCE) == (
        "./tests/code_coverage_tests/recursive_detector.py",
        "./tests/code_coverage_tests/router_code_coverage.py",
    )


def test_every_script_the_real_code_quality_job_names_exists():
    selected = mod.scripts(mod.WORKFLOW.read_text())
    assert selected
    assert [script for script in selected if not (mod.REPO_ROOT / script).is_file()] == []


def _workspace(tmp_path: Path, bad_exit: int) -> Path:
    scripts_dir = tmp_path / "tests" / "code_coverage_tests"
    scripts_dir.mkdir(parents=True)
    (scripts_dir / "passes.py").write_text("print('fine')\n")
    (scripts_dir / "flags.py").write_text(f"import sys\nprint('helper calls itself')\nsys.exit({bad_exit})\n")
    workflow = tmp_path / "test-linting.yml"
    workflow.write_text(
        "jobs:\n  code-quality:\n    steps:\n"
        "      - run: uv run --no-sync python ./tests/code_coverage_tests/passes.py\n"
        "      - run: uv run --no-sync python ./tests/code_coverage_tests/flags.py\n"
    )
    return workflow


def test_fails_and_shows_the_output_of_a_failing_script(tmp_path, capsys):
    workflow = _workspace(tmp_path, bad_exit=3)

    assert mod.main(workflow, tmp_path) == 1
    captured = capsys.readouterr()
    assert "./tests/code_coverage_tests/flags.py exited 3" in captured.err
    assert "helper calls itself" in captured.err
    assert "passes.py" not in captured.err
    assert "1 of 2 scripts passed" in captured.out


def test_passes_when_every_script_passes(tmp_path):
    workflow = _workspace(tmp_path, bad_exit=0)

    assert mod.main(workflow, tmp_path) == 0


def test_fails_when_the_job_names_no_scripts(tmp_path):
    workflow = tmp_path / "test-linting.yml"
    workflow.write_text("jobs:\n  python:\n    steps:\n      - run: make lint\n")

    assert mod.main(workflow, tmp_path) == 1
