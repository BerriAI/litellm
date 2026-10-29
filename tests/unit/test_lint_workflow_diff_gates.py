from __future__ import annotations

import fnmatch
import re
import shlex
import subprocess
from pathlib import Path
from typing import Final

import pytest
import yaml

WORKFLOW: Final = Path(__file__).resolve().parents[2] / ".github" / "workflows" / "test-linting.yml"
DIFF_GATE: Final = re.compile(r'git diff --name-only --diff-filter=\w+ "\$GATE_BASE_SHA" HEAD -- (.+?) \|')
GATES: Final = tuple(tuple(shlex.split(gate.group(1))) for gate in DIFF_GATE.finditer(WORKFLOW.read_text()))
PYTHON_GATES: Final = tuple(gate for gate in GATES if gate[0].startswith(":(glob)"))


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout


def _scoped_root(pathspec: str) -> str:
    return re.sub(r"^:\([^)]*\)", "", pathspec).split("*", 1)[0]


def _gate_rooted_at(root: str) -> tuple[str, ...]:
    return next(gate for gate in GATES if _scoped_root(gate[0]) == root)


def _changed_files_selected_by(tmp_path: Path, pathspecs: tuple[str, ...], files: tuple[str, ...]) -> frozenset[str]:
    _git(tmp_path, "init", "-q", "-b", "main")
    _git(tmp_path, "config", "user.email", "t@t")
    _git(tmp_path, "config", "user.name", "t")
    _git(tmp_path, "commit", "-q", "--allow-empty", "-m", "base")
    for name in files:
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("x = 1\n")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-qm", "change")
    return frozenset(
        _git(tmp_path, "diff", "--name-only", "--diff-filter=ACMRD", "HEAD~1", "HEAD", "--", *pathspecs).split()
    )


def test_workflow_still_carries_the_ruff_format_e2e_basedpyright_and_e2e_harness_diff_gates() -> None:
    assert len(GATES) == 3
    assert frozenset(gate[0] for gate in GATES) == frozenset(
        {":(glob)litellm/**/*.py", ":(glob)tests/e2e/**/*.py", "tests/e2e"}
    )


@pytest.mark.parametrize("pathspecs", PYTHON_GATES, ids=" ".join)
def test_diff_gate_selects_top_level_and_nested_python_files_only(tmp_path: Path, pathspecs: tuple[str, ...]) -> None:
    root = _scoped_root(pathspecs[0])
    top_level = f"{root}top_level_module.py"
    nested = f"{root}pkg/sub/nested_module.py"
    selected = _changed_files_selected_by(
        tmp_path,
        pathspecs,
        (top_level, nested, f"{root}notes.md", "elsewhere/top_level_module.py", "elsewhere/pkg/nested_module.py"),
    )
    assert selected == frozenset({top_level, nested})


@pytest.mark.parametrize(
    "trigger",
    (
        "tests/e2e_harness/top_level_module.py",
        "tests/e2e_harness/pkg/sub/nested_module.py",
        "pyrightconfig.json",
    ),
)
def test_e2e_basedpyright_gate_also_fires_on_harness_python_and_pyrightconfig(tmp_path: Path, trigger: str) -> None:
    selected = _changed_files_selected_by(
        tmp_path,
        _gate_rooted_at("tests/e2e/"),
        (trigger, "tests/e2e_harness/notes.md", "elsewhere/pyrightconfig.json", "tests/e2e_harnessish/module.py"),
    )
    assert selected == frozenset({trigger})


@pytest.mark.parametrize(
    "trigger",
    (
        "tests/e2e/claude_code/cron_vm/install_claude_code.sh",
        "tests/e2e/notes.md",
        "tests/e2e/pkg/sub/nested_module.py",
        "tests/e2e_harness/claude_code/test_driver.py",
        "pyproject.toml",
        "uv.lock",
        ".github/workflows/test-linting.yml",
    ),
)
def test_e2e_harness_gate_fires_on_any_e2e_or_harness_file_its_dependency_manifests_and_workflow(
    tmp_path: Path, trigger: str
) -> None:
    selected = _changed_files_selected_by(
        tmp_path,
        _gate_rooted_at("tests/e2e"),
        (trigger, "tests/e2e/ui/spec.ts", "tests/e2e/ui/pkg/page.py", "elsewhere/pyproject.toml", "tests/e2e_other/module.py"),
    )
    assert selected == frozenset({trigger})


UI_API_TYPES_DETECT: Final[str] = next(
    step["run"]
    for step in yaml.safe_load(WORKFLOW.read_text())["jobs"]["ui-api-types"]["steps"]
    if step.get("id") == "changes"
)
PRE_COMMIT_LINT: Final = WORKFLOW.parents[2] / "scripts" / "pre_commit_lint.sh"
TERRAFORM_PROVIDER_WORKFLOW: Final = WORKFLOW.with_name("test-terraform-provider.yml")


def _ui_api_types_relevant(tmp_path: Path, changed: str) -> bool:
    _git(tmp_path, "init", "-q", "-b", "main")
    _git(tmp_path, "config", "user.email", "t@t")
    _git(tmp_path, "config", "user.name", "t")
    _git(tmp_path, "commit", "-q", "--allow-empty", "-m", "base")
    _git(tmp_path, "switch", "-q", "-c", "pr")
    target: Final = tmp_path / changed
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("x = 1\n")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-qm", "change")
    _git(tmp_path, "switch", "-q", "main")
    _git(tmp_path, "commit", "-q", "--allow-empty", "-m", "main moved")
    _git(tmp_path, "merge", "-q", "--no-ff", "--no-edit", "pr")
    output: Final = tmp_path / "github_output"
    subprocess.run(
        ["bash", "-c", UI_API_TYPES_DETECT],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        env={"PATH": "/usr/bin:/bin", "GITHUB_OUTPUT": str(output)},
    )
    return output.read_text().strip() == "relevant=true"


@pytest.mark.parametrize(
    "changed",
    (
        "litellm/proxy/proxy_server.py",
        "litellm/types/router.py",
        "litellm/tracing/generated/models.py",
        "litellm/router_strategy/complexity_router/config.py",
        "litellm/models/credentials.py",
        "enterprise/litellm_enterprise/types/proxy/audit_logging_endpoints.py",
        "pyproject.toml",
        "uv.lock",
        "ui/litellm-dashboard/scripts/gen-api-types.mjs",
        "ui/litellm-dashboard/package-lock.json",
        "ui/litellm-dashboard/src/lib/http/schema.d.ts",
    ),
)
def test_ui_api_types_check_runs_on_a_pull_request_touching_any_spec_input(tmp_path: Path, changed: str) -> None:
    assert _ui_api_types_relevant(tmp_path, changed)


@pytest.mark.parametrize(
    "changed",
    (
        "litellm-rust/crates/traces/src/lib.rs",
        "litellm-proxy-extras/litellm_proxy_extras/utils.py",
        "tests/unit/test_router.py",
        "ui/litellm-dashboard/src/app/page.tsx",
        "docs/my-website/docs/proxy/config.md",
        "litellm-dashboard/pyproject.toml",
    ),
)
def test_ui_api_types_check_skips_a_pull_request_that_cannot_change_the_spec(tmp_path: Path, changed: str) -> None:
    assert not _ui_api_types_relevant(tmp_path, changed)


def _first_single_quoted_capture(pattern: str, text: str) -> str:
    match: Final = re.search(pattern, text, re.MULTILINE)
    assert match is not None, pattern
    return match.group(1)


def _tracked_paths_matching(extended_regex: str) -> frozenset[str]:
    tracked: Final = _git(WORKFLOW.parents[2], "ls-files")
    selected: Final = subprocess.run(["grep", "-E", extended_regex], input=tracked, capture_output=True, text=True)
    return frozenset(selected.stdout.splitlines())


def test_make_check_runs_the_api_type_sync_on_the_same_tracked_files_as_ci() -> None:
    ci_trigger: Final = _first_single_quoted_capture(r"grep -Eq '([^']+)'", UI_API_TYPES_DETECT)
    make_check_trigger: Final = _first_single_quoted_capture(r"^spec_pattern='([^']+)'$", PRE_COMMIT_LINT.read_text())
    ci_selected: Final = _tracked_paths_matching(ci_trigger)
    assert "pyproject.toml" in ci_selected
    assert _tracked_paths_matching(make_check_trigger) == ci_selected - {".github/workflows/test-linting.yml"}


def test_a_pull_request_changing_any_route_file_runs_the_terraform_provider_endpoint_audit() -> None:
    triggers: Final = yaml.safe_load(TERRAFORM_PROVIDER_WORKFLOW.read_text())[True]["pull_request"]["paths"]
    route_files: Final = _git(
        WORKFLOW.parents[2], "grep", "-l", "-E", r"APIRouter\(", "--", "*.py", ":(exclude)tests/**"
    ).splitlines()
    assert route_files
    assert [name for name in route_files if not any(fnmatch.fnmatch(name, pattern) for pattern in triggers)] == []
