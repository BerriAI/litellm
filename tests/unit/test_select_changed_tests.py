import importlib.util
import sys
from pathlib import Path
from typing import Final

import yaml

_REPO_ROOT: Final = Path(__file__).resolve().parents[2]
_MODULE_PATH: Final = _REPO_ROOT / ".github" / "scripts" / "ci" / "select_changed_tests.py"
_spec = importlib.util.spec_from_file_location("select_changed_tests", _MODULE_PATH)
selector = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = selector
_spec.loader.exec_module(selector)

WORKFLOW: Final = {
    "jobs": {
        "unit-core": {
            "strategy": {
                "matrix": {
                    "include": [
                        {"shard": "core-auth", "test-path": "tests/unit/proxy/auth/test_core_auth.py"},
                        {
                            "shard": "core-router",
                            "test-path": "tests/unit/router\n--ignore=tests/unit/router/test_slow.py",
                        },
                    ]
                }
            }
        },
        "unit": {
            "strategy": {
                "matrix": {
                    "include": [
                        {
                            "shard": "serial",
                            "test-path": "tests/unit/db/test_serial.py",
                            "workers": 0,
                            "python-version": "3.12",
                        },
                        {"shard": "old-python", "test-path": "tests/unit/lens", "workers": 4, "python-version": "3.10"},
                        {"shard": "plain", "test-path": "tests/unit/proxy", "workers": 4, "python-version": "3.12"},
                    ]
                }
            }
        },
    }
}


def _tree(root: Path, *paths: str) -> Path:
    for path in paths:
        (root / path).parent.mkdir(parents=True, exist_ok=True)
        (root / path).write_text("")
    return root


def _select(root: Path, *changed: str) -> tuple[str, ...]:
    return selector.select(changed, selector.exclusions(WORKFLOW, "3.12"), root)


def test_a_changed_unit_test_file_is_selected(tmp_path):
    root: Final = _tree(tmp_path, "tests/unit/proxy/test_spend.py")
    assert _select(root, "tests/unit/proxy/test_spend.py") == ("tests/unit/proxy/test_spend.py",)


def test_a_changed_source_module_selects_its_mirrored_tests_only(tmp_path):
    root: Final = _tree(
        tmp_path,
        "litellm/proxy/hooks/budget.py",
        "tests/unit/proxy/hooks/test_budget.py",
        "tests/unit/proxy/hooks/test_budget_windows.py",
        "tests/unit/proxy/hooks/test_budgeting.py",
        "tests/unit/proxy/test_budget.py",
    )
    assert _select(root, "litellm/proxy/hooks/budget.py") == (
        "tests/unit/proxy/hooks/test_budget.py",
        "tests/unit/proxy/hooks/test_budget_windows.py",
    )


def test_files_the_core_shards_run_are_left_out(tmp_path):
    root: Final = _tree(
        tmp_path,
        "tests/unit/proxy/auth/test_core_auth.py",
        "tests/unit/router/test_slow.py",
        "tests/unit/router/nested/test_deep.py",
        "tests/unit/proxy/auth/test_other_auth.py",
    )
    assert _select(
        root,
        "tests/unit/proxy/auth/test_core_auth.py",
        "tests/unit/router/test_slow.py",
        "tests/unit/router/nested/test_deep.py",
        "tests/unit/proxy/auth/test_other_auth.py",
    ) == ("tests/unit/proxy/auth/test_other_auth.py",)


def test_files_needing_serial_runs_or_another_python_are_left_out(tmp_path):
    root: Final = _tree(
        tmp_path, "tests/unit/db/test_serial.py", "tests/unit/lens/test_lens.py", "tests/unit/db/test_pool.py"
    )
    assert _select(
        root, "tests/unit/db/test_serial.py", "tests/unit/lens/test_lens.py", "tests/unit/db/test_pool.py"
    ) == ("tests/unit/db/test_pool.py",)


def test_deleted_files_helpers_and_package_inits_select_nothing(tmp_path):
    root: Final = _tree(
        tmp_path,
        "tests/unit/proxy/conftest.py",
        "tests/unit/proxy/helpers.py",
        "litellm/proxy/__init__.py",
        "tests/unit/proxy/test___init__.py",
    )
    assert (
        _select(
            root,
            "tests/unit/proxy/test_removed.py",
            "tests/unit/proxy/conftest.py",
            "tests/unit/proxy/helpers.py",
            "litellm/proxy/__init__.py",
            "docs/README.md",
            "",
        )
        == ()
    )


def test_the_real_unit_workflow_yields_core_exclusions_that_exist():
    workflow: Final = yaml.safe_load((_REPO_ROOT / ".github/workflows/test-unit.yml").read_text(encoding="utf-8"))
    core_paths: Final = selector.exclusions({"jobs": {"unit-core": workflow["jobs"]["unit-core"]}}, "3.12")
    assert core_paths
    assert all((_REPO_ROOT / path).exists() for path in core_paths)
