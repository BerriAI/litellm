import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Final

import pytest
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


def test_include_core_selects_core_tests_but_still_excludes_incompatible_shards(tmp_path):
    paths: Final = (
        "tests/unit/proxy/auth/test_core_auth.py",
        "tests/unit/router/nested/test_deep.py",
        "tests/unit/db/test_serial.py",
        "tests/unit/lens/test_lens.py",
    )
    root: Final = _tree(tmp_path, *paths)
    excluded: Final = selector.exclusions(WORKFLOW, "3.12", include_core=True)
    assert selector.select(paths, excluded, root) == paths[:2]


def test_node_selected_core_tests_exclude_their_file_from_changed_tests(tmp_path):
    root: Final = _tree(tmp_path, "tests/unit/test_router.py", "tests/unit/test_other.py")
    workflow: Final = {
        "jobs": {
            "unit-core": {
                "strategy": {
                    "matrix": {
                        "include": [{"test-path": "tests/unit/test_router.py::TestRetries::test_retry[timeout]"}]
                    }
                }
            }
        }
    }
    assert selector.select(
        ("tests/unit/test_router.py", "tests/unit/test_other.py"), selector.exclusions(workflow, "3.12"), root
    ) == ("tests/unit/test_other.py",)


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


@pytest.mark.parametrize(
    "changed",
    (
        "litellm/proxy-extras/litellm_proxy_extras/migrations/20990101000000_add_column/migration.sql",
        "litellm/proxy-extras/litellm_proxy_extras/schema.prisma",
        "litellm/proxy-extras/litellm_proxy_extras/utils.py",
    ),
)
def test_a_proxy_extras_change_selects_every_proxy_extras_test(tmp_path, changed):
    root: Final = _tree(
        tmp_path,
        "tests/unit/litellm_proxy_extras/test_litellm_proxy_extras_utils.py",
        "tests/unit/litellm_proxy_extras/test_migration_dry_run.py",
        "tests/unit/litellm_proxy_extras/helpers.py",
        "tests/unit/db_scripts/test_db_scripts.py",
    )
    assert _select(root, changed) == (
        "tests/unit/litellm_proxy_extras/test_litellm_proxy_extras_utils.py",
        "tests/unit/litellm_proxy_extras/test_migration_dry_run.py",
    )


def test_a_change_outside_proxy_extras_selects_no_proxy_extras_test(tmp_path):
    root: Final = _tree(tmp_path, "tests/unit/litellm_proxy_extras/test_litellm_proxy_extras_utils.py")
    assert _select(root, "litellm/proxy/schema.prisma", "litellm/proxy-extras-notes.md") == ()


@pytest.mark.parametrize(
    "changed", ("model_prices_and_context_window.json", "litellm/model_prices_and_context_window_backup.json")
)
def test_a_cost_map_change_selects_the_cost_map_tests_and_every_root_test_naming_a_cost_map_file(tmp_path, changed):
    root: Final = _tree(
        tmp_path,
        "tests/unit/test_model_prices_and_context_window.py",
        "tests/unit/test_reads_main_model_metadata.py",
        "tests/unit/test_reads_backup_model_metadata.py",
        "tests/unit/test_utils.py",
        "tests/unit/helpers.py",
        "tests/unit/llms/test_nested_reader.py",
        "tests/unit/litellm_proxy_extras/test_litellm_proxy_extras_utils.py",
    )
    main_reader: Final = 'json.loads(Path("model_prices_and_context_window.json").read_text())'
    backup_reader: Final = 'open("litellm/model_prices_and_context_window_backup.json")'
    (root / "tests/unit/test_reads_main_model_metadata.py").write_text(main_reader, encoding="utf-8")
    (root / "tests/unit/test_reads_backup_model_metadata.py").write_text(backup_reader, encoding="utf-8")
    (root / "tests/unit/helpers.py").write_text(main_reader, encoding="utf-8")
    (root / "tests/unit/llms/test_nested_reader.py").write_text(main_reader, encoding="utf-8")
    assert _select(root, changed) == (
        "tests/unit/test_model_prices_and_context_window.py",
        "tests/unit/test_reads_backup_model_metadata.py",
        "tests/unit/test_reads_main_model_metadata.py",
    )


def test_a_change_outside_the_cost_map_selects_no_root_test_naming_a_cost_map_file(tmp_path):
    root: Final = _tree(tmp_path, "tests/unit/test_reads_main_model_metadata.py", "tests/unit/proxy/test_utils.py")
    (root / "tests/unit/test_reads_main_model_metadata.py").write_text(
        'open("model_prices_and_context_window.json")', encoding="utf-8"
    )
    assert _select(root, "litellm/proxy/utils.py", "litellm/cost.json") == ("tests/unit/proxy/test_utils.py",)


def test_a_file_named_like_the_cost_map_elsewhere_selects_no_cost_map_test(tmp_path):
    root: Final = _tree(tmp_path, "tests/unit/test_model_prices_and_context_window.py")
    assert (
        _select(root, "litellm/model_prices_and_context_window.json", "docs/model_prices_and_context_window.json") == ()
    )


@pytest.mark.parametrize(
    "changed", ("model_prices_and_context_window.json", "litellm/model_prices_and_context_window_backup.json")
)
def test_the_real_unit_workflow_lets_a_cost_map_change_run_the_cost_map_and_model_metadata_tests(changed):
    workflow: Final = yaml.safe_load((_REPO_ROOT / ".github/workflows/test-unit.yml").read_text(encoding="utf-8"))
    selected: Final = selector.select((changed,), selector.exclusions(workflow, "3.12"), _REPO_ROOT)
    assert "tests/unit/test_model_prices_and_context_window.py" in selected
    assert "tests/unit/test_together_ai_model_metadata.py" in selected
    assert all(Path(path).parent.as_posix() == "tests/unit" for path in selected)


def test_the_real_unit_workflow_lets_a_migration_change_run_the_migration_ddl_tests():
    workflow: Final = yaml.safe_load((_REPO_ROOT / ".github/workflows/test-unit.yml").read_text(encoding="utf-8"))
    changed: Final = ("litellm/proxy-extras/litellm_proxy_extras/migrations/20990101000000_add_column/migration.sql",)
    assert "tests/unit/litellm_proxy_extras/test_litellm_proxy_extras_utils.py" in selector.select(
        changed, selector.exclusions(workflow, "3.12"), _REPO_ROOT
    )


def test_the_real_unit_workflow_yields_core_exclusions_that_exist():
    workflow: Final = yaml.safe_load((_REPO_ROOT / ".github/workflows/test-unit.yml").read_text(encoding="utf-8"))
    core_paths: Final = selector.exclusions({"jobs": {"unit-core": workflow["jobs"]["unit-core"]}}, "3.12")
    assert core_paths
    assert all((_REPO_ROOT / path).exists() for path in core_paths)


@pytest.mark.parametrize(
    ("changed", "classifier", "tests_only", "selected"),
    (
        ("tests/unit/test_utils.py", None, "true", "tests/unit/test_utils.py"),
        ("tests/unit/router/test_removed.py", None, "true", ""),
        ("tests/unit/test_utils.py\ntests/conftest.py", None, "false", ""),
        ("tests/unit/test_utils.py", "exit 1", "false", ""),
        ("tests/unit/test_utils.py", "echo maybe", "false", ""),
    ),
)
def test_plan_skips_core_only_for_successful_tests_only_classification(
    tmp_path, changed, classifier, tests_only, selected
):
    workflow: Final = yaml.safe_load((_REPO_ROOT / ".github/workflows/test-unit.yml").read_text())
    script_dir: Final = tmp_path / ".github/scripts/ci"
    script_dir.mkdir(parents=True)
    shutil.copy(_MODULE_PATH, script_dir)
    classifier_path: Final = script_dir / "classify_changes.sh"
    if classifier is None:
        shutil.copy(_MODULE_PATH.with_name("classify_changes.sh"), classifier_path)
    else:
        classifier_path.write_text(classifier)
    workflow_path: Final = tmp_path / ".github/workflows/test-unit.yml"
    workflow_path.parent.mkdir()
    workflow_path.write_text(yaml.safe_dump(workflow))
    _tree(tmp_path, "tests/unit/test_utils.py")
    bin_dir: Final = tmp_path / "bin"
    bin_dir.mkdir()
    gh: Final = bin_dir / "gh"
    gh.write_text('#!/bin/bash\ncase "$2" in */files) printf "%s\\n" "$CHANGED" ;; esac\n')
    gh.chmod(0o755)
    output: Final = tmp_path / "output"
    result: Final = subprocess.run(
        ["bash", "-e", "-c", workflow["jobs"]["plan"]["steps"][-1]["run"]],
        cwd=tmp_path,
        env={
            **os.environ,
            "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
            "PR_NUMBER": "1",
            "BASE_REF": "main",
            "GITHUB_REPOSITORY": "BerriAI/litellm",
            "GITHUB_OUTPUT": str(output),
            "CHANGED": changed,
        },
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    outputs: Final = dict(line.split("=", 1) for line in output.read_text().splitlines())
    assert outputs["mode"] == "core"
    assert outputs["tests-only"] == tests_only
    assert outputs["changed-tests"] == selected
    assert outputs["lint-runtime"] == "true"
    skipped: Final = json.loads(outputs["skipped"])
    assert ("unit-core" in skipped) == (tests_only == "true")
    assert ("unit-changed" in skipped) == (not selected)
    needs: Final = {
        job: {"result": "skipped" if job in skipped else "success"} for job in workflow["jobs"]["unit-passed"]["needs"]
    }
    aggregate: Final = subprocess.run(
        ["bash", "-e", "-c", workflow["jobs"]["unit-passed"]["steps"][0]["run"]],
        env={**os.environ, "NEEDS": json.dumps({**needs, "plan": {"result": "success", "outputs": outputs}})},
        capture_output=True,
        text=True,
    )
    assert aggregate.returncode == 0, aggregate.stdout + aggregate.stderr
