from __future__ import annotations

import fnmatch
import itertools
import os
import re
import shlex
import subprocess
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Final, TypeAlias

import pytest
import yaml
from pydantic import BaseModel, ConfigDict, Field

WORKFLOW: Final = Path(__file__).resolve().parents[2] / ".github" / "workflows" / "test-linting.yml"
WORKFLOWS_DIR: Final = WORKFLOW.parent
REPO_ROOT: Final = WORKFLOW.parents[2]
DIFF_GATE: Final = re.compile(r'git diff --name-only --diff-filter=\w+ "\$GATE_BASE_SHA" HEAD -- (.+?) \|')
GATES: Final = tuple(tuple(shlex.split(gate.group(1))) for gate in DIFF_GATE.finditer(WORKFLOW.read_text()))
PYTHON_GATES: Final = tuple(gate for gate in GATES if gate[0].startswith(":(glob)"))
REPOSITORY: Final = "BerriAI/litellm"
UI_DASHBOARD: Final = "ui/litellm-dashboard"
STACKED_BASE_REF: Final = "litellm_/stacked"
RELEASE_BASE_REF: Final = "stable/next"
BASE_REFS: Final = ("main", STACKED_BASE_REF, RELEASE_BASE_REF)
RENAMED_BASE_REF: Final = "litellm_renamed_before_the_rerun"
GITHUB_EXPRESSION: Final = re.compile(r"\$\{\{\s*(.+?)\s*\}\}")
FAKE_GH: Final = r"""#!/usr/bin/env bash
set -euo pipefail
endpoint=${2%%\?*}
endpoint=${endpoint#"repos/$GITHUB_REPOSITORY/"}
commit() { git -C "$UPSTREAM" rev-parse --verify --quiet "$1^{commit}"; }
renamed() { git -C "$UPSTREAM" config --get "renamed.$1.to" || printf '%s\n' "$1"; }
case "$1 $endpoint ${3-} ${4-}" in
  "api compare/"*"..."*" --jq .merge_base_commit.sha")
    basehead=${endpoint#compare/}
    git -C "$UPSTREAM" merge-base "$(commit "${basehead%%...*}")" "$(commit "${basehead#*...}")" ;;
  "api pulls/1 --jq .base.ref")
    printf '%s\n' "$LIVE_BASE_REF" ;;
  "api branches/"*)
    branch=$(renamed "${endpoint#branches/}")
    sha=$(commit "refs/heads/$branch") || { [ "${3-}" = --silent ] || printf '{"message":"Not Found"}\n'; exit 1; }
    case "${3-} ${4-}" in
      "--silent ") ;;
      "--jq .name") printf '%s\n' "$branch" ;;
      "--jq .commit.sha") printf '%s\n' "$sha" ;;
      *) echo "fake gh cannot answer: $*" >&2; exit 64 ;;
    esac ;;
  "api pulls/1/reviews --paginate --slurp")
    printf '[[]]\n' ;;
  *)
    echo "fake gh cannot answer: $*" >&2
    exit 64 ;;
esac
"""
FAKE_ARGV_RECORDER: Final = r"""#!/usr/bin/env bash
printf '%s\n' "$@" > "$RUNNER_TEMP/$(basename "$0").argv"
"""


class WorkflowStep(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str = ""
    name: str = ""
    uses: str = ""
    with_: Mapping[str, str] = Field(default_factory=dict, alias="with")
    env: Mapping[str, str] = Field(default_factory=dict)
    run: str = ""


class RunDefaults(BaseModel):
    model_config = ConfigDict(frozen=True)

    working_directory: str = Field(default=".", alias="working-directory")


class JobDefaults(BaseModel):
    model_config = ConfigDict(frozen=True)

    run: RunDefaults = Field(default_factory=RunDefaults)


class WorkflowJob(BaseModel):
    model_config = ConfigDict(frozen=True)

    defaults: JobDefaults = Field(default_factory=JobDefaults)
    steps: tuple[WorkflowStep, ...] = ()


class PullRequestTrigger(BaseModel):
    model_config = ConfigDict(frozen=True)

    branches: tuple[str, ...] = ("**",)


class WorkflowTriggers(BaseModel):
    model_config = ConfigDict(frozen=True)

    pull_request: PullRequestTrigger = Field(default_factory=PullRequestTrigger)


class Workflow(BaseModel):
    model_config = ConfigDict(frozen=True)

    on: WorkflowTriggers
    jobs: Mapping[str, WorkflowJob]


@dataclass(frozen=True, slots=True)
class MergeBaseStep:
    workflow: str
    job_id: str
    job: WorkflowJob
    step: WorkflowStep
    base_branches: tuple[str, ...]

    @property
    def job_key(self) -> str:
        return f"{self.workflow}:{self.job_id}"


@dataclass(frozen=True, slots=True)
class StaleBaseUpstream:
    path: Path
    stale_base: str
    head: str
    tip_by_ref: Mapping[str, str]
    merge_base_by_ref: Mapping[str, str]


@dataclass(frozen=True, slots=True)
class Runner:
    root: Path
    upstream: StaleBaseUpstream
    live_base_ref: str
    context: Mapping[str, str]

    @property
    def checkout(self) -> Path:
        return self.root / "checkout"

    @property
    def tools(self) -> Path:
        return self.root / "bin"

    @property
    def temp(self) -> Path:
        return self.root / "runner_temp"

    @property
    def github_output(self) -> Path:
        return self.root / "github_output"

    @property
    def github_env(self) -> Path:
        return self.root / "github_env"


@dataclass(frozen=True, slots=True)
class StepRun:
    step: MergeBaseStep
    runner: Runner


Observation: TypeAlias = str | frozenset[str] | tuple[str, str]


@dataclass(frozen=True, slots=True)
class Consumer:
    observe: Callable[[StepRun], Observation]
    expect: Callable[[StaleBaseUpstream, str], Observation]


def _workflow(path: Path) -> Workflow:
    return Workflow.model_validate(yaml.load(path.read_text(), Loader=yaml.BaseLoader))


def _merge_base_steps_in(path: Path) -> Iterator[MergeBaseStep]:
    workflow: Final = _workflow(path)
    for job_id, job in workflow.jobs.items():
        yield from (
            MergeBaseStep(path.name, job_id, job, step, workflow.on.pull_request.branches)
            for step in job.steps
            if "merge_base_commit" in step.run
        )


MERGE_BASE_STEPS: Final = tuple(
    itertools.chain.from_iterable(map(_merge_base_steps_in, sorted(WORKFLOWS_DIR.glob("*.y*ml"))))
)


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


def _event_context(pull_request: Mapping[str, str]) -> Mapping[str, str]:
    return MappingProxyType(
        {
            "github.repository": REPOSITORY,
            "github.token": "fake-token",
            **{f"github.event.pull_request.{field}": value for field, value in pull_request.items()},
        }
    )


def _context_value(context: Mapping[str, str], expression: str) -> str:
    if expression not in context:
        pytest.fail(f"the fake runner has no value for ${{{{ {expression} }}}}")
    return context[expression]


def _resolved(text: str, context: Mapping[str, str]) -> str:
    return GITHUB_EXPRESSION.sub(lambda expression: _context_value(context, expression.group(1)), text)


def _install_tool(path: Path, script: str) -> None:
    path.write_text(script)
    path.chmod(0o755)


def _runner(tmp_path: Path, upstream: StaleBaseUpstream, pull_request: Mapping[str, str], live_base_ref: str) -> Runner:
    runner: Final = Runner(tmp_path, upstream, live_base_ref, _event_context(pull_request))
    runner.tools.mkdir()
    runner.temp.mkdir()
    for name, script in (("gh", FAKE_GH), ("npm", FAKE_ARGV_RECORDER), ("uv", FAKE_ARGV_RECORDER)):
        _install_tool(runner.tools / name, script)
    return runner


def _check_out_like_the_job(job: WorkflowJob, runner: Runner) -> None:
    checkout: Final = next(step for step in job.steps if step.uses.startswith("actions/checkout@"))
    ref: Final = _resolved(checkout.with_.get("ref", "${{ github.event.pull_request.head.sha }}"), runner.context)
    _git(runner.root, "init", "-q", str(runner.checkout))
    _git(runner.checkout, "remote", "add", "origin", f"file://{runner.upstream.path}")
    _git(runner.checkout, "fetch", "-q", "--no-tags", "--depth=1", "origin", ref)
    _git(runner.checkout, "checkout", "-q", "--detach", "FETCH_HEAD")


def _run_step(step: WorkflowStep, job: WorkflowJob, context: Mapping[str, str], runner: Runner) -> None:
    result: Final = subprocess.run(
        ["bash", "--noprofile", "--norc", "-eo", "pipefail", "-c", _resolved(step.run, context)],
        cwd=runner.checkout / job.defaults.run.working_directory,
        env={
            **{name: _resolved(value, context) for name, value in step.env.items()},
            "PATH": f"{runner.tools}{os.pathsep}{os.environ['PATH']}",
            "HOME": str(runner.root),
            "GITHUB_REPOSITORY": REPOSITORY,
            "GITHUB_WORKSPACE": str(REPO_ROOT),
            "GITHUB_OUTPUT": str(runner.github_output),
            "GITHUB_ENV": str(runner.github_env),
            "RUNNER_TEMP": str(runner.temp),
            "UPSTREAM": str(runner.upstream.path),
            "LIVE_BASE_REF": runner.live_base_ref,
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def _key_values(path: Path) -> Mapping[str, str]:
    return MappingProxyType(
        {key: value for key, _, value in (line.partition("=") for line in path.read_text().splitlines())}
    )


def _argv_recorded(runner: Runner, tool: str) -> tuple[str, ...]:
    return tuple((runner.temp / f"{tool}.argv").read_text().splitlines())


def _gate_base_exported(run: StepRun) -> str:
    return _key_values(run.runner.github_env)["GATE_BASE_SHA"]


def _ui_files_queued_for_prettier(run: StepRun) -> frozenset[str]:
    return frozenset((run.runner.temp / "prettier_files.txt").read_text().splitlines())


def _ui_files_passed_to_vitest_related(run: StepRun) -> frozenset[str]:
    argv: Final = _argv_recorded(run.runner, "npm")
    return frozenset(argv[argv.index("related") + 1 : argv.index("--run")])


def _guard_base_argument_and_checkout(run: StepRun) -> tuple[str, str]:
    guard: Final = next(step for step in run.step.job.steps if step.name == "Run the guard")
    outputs: Final = {
        f"steps.{run.step.step.id}.outputs.{name}": value
        for name, value in _key_values(run.runner.github_output).items()
    }
    _run_step(guard, run.step.job, MappingProxyType({**run.runner.context, **outputs}), run.runner)
    argv: Final = _argv_recorded(run.runner, "uv")
    return argv[argv.index("--base") + 1], _sha(run.runner.checkout, "HEAD")


def _merge_base(upstream: StaleBaseUpstream, ref: str) -> str:
    return upstream.merge_base_by_ref[ref]


def _ui_files_changed_since_merge_base(upstream: StaleBaseUpstream, ref: str) -> frozenset[str]:
    merge_base: Final = upstream.merge_base_by_ref[ref]
    changed: Final = _git(upstream.path, "diff", "--name-only", f"--relative={UI_DASHBOARD}", merge_base, upstream.head)
    return frozenset(changed.split())


def _merge_base_and_tip(upstream: StaleBaseUpstream, ref: str) -> tuple[str, str]:
    return upstream.merge_base_by_ref[ref], upstream.tip_by_ref[ref]


CONSUMERS: Final = MappingProxyType(
    {
        "test-linting.yml:python": Consumer(_gate_base_exported, _merge_base),
        "test-linting.yml:ui": Consumer(_ui_files_queued_for_prettier, _ui_files_changed_since_merge_base),
        "test-unit.yml:ui-unit": Consumer(_ui_files_passed_to_vitest_related, _ui_files_changed_since_merge_base),
        "spend-logs-guard.yml:spend-logs-guard": Consumer(_guard_base_argument_and_checkout, _merge_base_and_tip),
        "cost-map-guard.yml:cost-map-guard": Consumer(_guard_base_argument_and_checkout, _merge_base_and_tip),
    }
)


def _sha(repo: Path, revision: str) -> str:
    return _git(repo, "rev-parse", revision).strip()


def _commit_ui_file(repo: Path, name: str) -> None:
    target: Final = repo / UI_DASHBOARD / "src" / f"{name}.ts"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(f"export const {name} = 1;\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", name)


@pytest.fixture(scope="module")
def upstream(tmp_path_factory: pytest.TempPathFactory) -> StaleBaseUpstream:
    path: Final = tmp_path_factory.mktemp("upstream")
    _git(path, "init", "-q", "-b", "main")
    _git(path, "config", "user.email", "t@t")
    _git(path, "config", "user.name", "t")
    _commit_ui_file(path, "main_when_the_pull_request_was_opened")
    _git(path, "branch", "feature")
    _git(path, "branch", RELEASE_BASE_REF)
    _commit_ui_file(path, "main_since")
    _git(path, "checkout", "-q", "-b", STACKED_BASE_REF)
    _commit_ui_file(path, "stacked_base_since")
    _git(path, "checkout", "-q", "feature")
    _commit_ui_file(path, "pull_request_work")
    _git(path, "merge", "-q", "--no-ff", "-m", "merge the newer base", STACKED_BASE_REF)
    _git(path, "config", f"renamed.{RENAMED_BASE_REF}.to", STACKED_BASE_REF)
    return StaleBaseUpstream(
        path=path,
        stale_base=_sha(path, "main~1"),
        head=_sha(path, "feature"),
        tip_by_ref=MappingProxyType({ref: _sha(path, ref) for ref in BASE_REFS}),
        merge_base_by_ref=MappingProxyType(
            {ref: _git(path, "merge-base", ref, "feature").strip() for ref in BASE_REFS}
        ),
    )


def test_every_workflow_that_reads_a_merge_base_has_its_steps_discovered() -> None:
    assert MERGE_BASE_STEPS
    assert frozenset(step.workflow for step in MERGE_BASE_STEPS) == frozenset(
        path.name for path in WORKFLOWS_DIR.glob("*.y*ml") if "merge_base_commit" in path.read_text()
    )


def _runs_for(step: MergeBaseStep, base_ref: str) -> bool:
    return any(fnmatch.fnmatchcase(base_ref, pattern) for pattern in step.base_branches)


BASE_REF_CASES: Final = (
    ("main", "main", "main"),
    ("stacked", STACKED_BASE_REF, STACKED_BASE_REF),
    ("rerun_after_the_stacked_base_was_deleted", "litellm_deleted_after_merge", "main"),
    ("rerun_after_the_stacked_base_was_renamed", RENAMED_BASE_REF, STACKED_BASE_REF),
    ("rerun_after_a_retarget_off_main_to_a_stacked_branch", "main", STACKED_BASE_REF),
    ("rerun_after_a_retarget_off_main_to_a_release_branch", "main", RELEASE_BASE_REF),
)
MERGE_BASE_CASES: Final = tuple(
    pytest.param(step, event_base_ref, live_base_ref, id=f"{step.job_key}:{step.step.name}-{case}")
    for step, (case, event_base_ref, live_base_ref) in itertools.product(MERGE_BASE_STEPS, BASE_REF_CASES)
    if _runs_for(step, event_base_ref)
)


@pytest.mark.parametrize(("step", "event_base_ref", "live_base_ref"), MERGE_BASE_CASES)
def test_merge_base_step_hands_its_consumer_the_events_base_unless_that_branch_is_gone(
    upstream: StaleBaseUpstream, tmp_path: Path, step: MergeBaseStep, event_base_ref: str, live_base_ref: str
) -> None:
    pull_request: Final = MappingProxyType(
        {
            "number": "1",
            "base.ref": event_base_ref,
            "base.sha": upstream.stale_base,
            "head.sha": upstream.head,
            "head.ref": "feature",
            "user.type": "User",
        }
    )
    runner: Final = _runner(tmp_path, upstream, pull_request, live_base_ref)
    _check_out_like_the_job(step.job, runner)
    _run_step(step.step, step.job, runner.context, runner)
    consumer: Final = CONSUMERS[step.job_key]
    measured_ref: Final = event_base_ref if event_base_ref in upstream.tip_by_ref else live_base_ref
    assert consumer.observe(StepRun(step, runner)) == consumer.expect(upstream, measured_ref)


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
