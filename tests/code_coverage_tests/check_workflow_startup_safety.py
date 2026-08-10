"""Catch workflow mistakes that GitHub reports as nothing at all.

A workflow whose YAML is valid but whose expressions are not fails at *startup*:
the run is marked failed, no jobs are created, and no check run is ever posted.
Nothing turns red on the PR, so an entire test suite can silently stop running
while the checks list stays green. These invariants have to be enforced here
because CI cannot enforce them on itself.

1. No arithmetic inside ``${{ }}``. GitHub expressions support grouping, index,
   dereference, ``!``, the comparisons, ``&&`` and ``||``, and nothing else. A
   ``${{ a + b }}`` is a startup failure, not a value. Only ``+`` and ``*`` are
   flagged: ``-`` appears in hyphenated input names like ``inputs.timeout-minutes``
   and ``/`` inside ref strings, so neither can be told apart from arithmetic by
   inspection alone.
2. Callers of the reusable unit-test workflow keep the job timeout at or above
   the test budget plus the setup ceilings plus the runner overhead below.
   Otherwise the job deadline preempts pytest inside its own advertised budget,
   which is the failure the split timeouts exist to prevent, and it shows up as
   a cancelled shard whose tests were passing.
"""

import re
import sys
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Final

import yaml
from pydantic import BaseModel, Field, ValidationError

REPO_ROOT: Final = Path(__file__).resolve().parent.parent.parent
WORKFLOWS_DIR: Final = REPO_ROOT / ".github" / "workflows"
BASE_WORKFLOW: Final = "./.github/workflows/_test-unit-base.yml"
BASE_WORKFLOW_PATH: Final = WORKFLOWS_DIR / "_test-unit-base.yml"

# Runner time the job clock charges but no step owns: job init, the gaps between
# steps, and post-job cleanup. Without it a job capped at exactly test + setup
# would still preempt pytest inside its own budget.
JOB_OVERHEAD_MINUTES: Final = 5

EXPRESSION: Final = re.compile(r"\$\{\{(?P<body>.*?)\}\}", re.DOTALL)
QUOTED: Final = re.compile(r"'[^']*'")
ARITHMETIC: Final = re.compile(r"[+*]")
MATRIX_REF: Final = re.compile(r"^\$\{\{\s*matrix\.(?P<key>[\w-]+)\s*\}\}$")


class WorkflowStartupError(Exception):
    pass


class ReusableCall(BaseModel):
    uses: str | None = None
    with_: Mapping[str, object] = Field(default_factory=dict, alias="with")
    strategy: Mapping[str, object] = Field(default_factory=dict)
    steps: tuple[Mapping[str, object], ...] = ()

    model_config = {"populate_by_name": True}


class WorkflowFile(BaseModel):
    jobs: Mapping[str, ReusableCall] = Field(default_factory=dict)


def parse_workflow(text: str) -> WorkflowFile | str:
    parsed: Final = yaml.safe_load(text)
    try:
        return WorkflowFile.model_validate(parsed if isinstance(parsed, dict) else {})
    except ValidationError as exc:
        return f"does not parse as a workflow: {exc.error_count()} schema error(s)"


def arithmetic_expressions(text: str) -> Iterator[str]:
    for match in EXPRESSION.finditer(text):
        body: Final = match.group("body")
        if ARITHMETIC.search(QUOTED.sub("", body)):
            yield body.strip()


def setup_ceiling_minutes(base_text: str) -> int:
    """Sum the per-step timeouts on everything the base workflow runs before pytest."""
    base: Final = yaml.safe_load(base_text)
    steps: Final = base["jobs"]["run"]["steps"]
    return sum(
        s["timeout-minutes"]
        for s in steps
        if s.get("name") != "Run tests" and isinstance(s.get("timeout-minutes"), int)
    )


def base_default(base_text: str, name: str) -> int:
    base: Final = yaml.safe_load(base_text)
    return base[True]["workflow_call"]["inputs"][name]["default"]


def budget_source(job: ReusableCall, key: str, fallback: int) -> int | str | None:
    """A caller passes a literal, or `${{ matrix.x }}` naming a column of its matrix."""
    value: Final = job.with_.get(key)
    if value is None:
        return fallback
    if isinstance(value, int):
        return value

    matrix_ref: Final = MATRIX_REF.match(str(value))
    return matrix_ref.group("key") if matrix_ref else None


def matrix_rows(job: ReusableCall) -> Sequence[Mapping[str, object]]:
    matrix: Final = job.strategy.get("matrix", {})
    entries: Final = matrix.get("include", ()) if isinstance(matrix, dict) else ()
    return tuple(e for e in entries if isinstance(e, dict))


def budget_pairs(job: ReusableCall, test_source: int | str, job_source: int | str) -> Iterator[tuple[int, int]]:
    """Pair each shard's test budget with the job budget of that same shard.

    Matrix-sourced budgets resolve per `include` row, so two matrix columns are
    read off the same row rather than cross-producted across rows.
    """
    if isinstance(test_source, int) and isinstance(job_source, int):
        yield test_source, job_source
        return

    for row in matrix_rows(job):
        test_budget = row.get(test_source) if isinstance(test_source, str) else test_source
        job_budget = row.get(job_source) if isinstance(job_source, str) else job_source
        if isinstance(test_budget, int) and isinstance(job_budget, int):
            yield test_budget, job_budget


def timeout_contract_errors(rel: Path, workflow: WorkflowFile, ceiling: int, base_text: str) -> Iterator[str]:
    for job_name, job in workflow.jobs.items():
        if job.uses != BASE_WORKFLOW:
            continue

        test_source: Final = budget_source(job, "timeout-minutes", base_default(base_text, "timeout-minutes"))
        job_source: Final = budget_source(job, "job-timeout-minutes", base_default(base_text, "job-timeout-minutes"))
        if test_source is None or job_source is None:
            continue

        for test_budget, job_budget in budget_pairs(job, test_source, job_source):
            required = test_budget + ceiling + JOB_OVERHEAD_MINUTES
            if job_budget < required:
                yield (
                    f"{rel}: job `{job_name}` gives pytest {test_budget}m but caps the job at "
                    f"{job_budget}m. Setup can use up to {ceiling}m plus {JOB_OVERHEAD_MINUTES}m of "
                    f"runner overhead, so the job deadline would preempt pytest; raise "
                    f"job-timeout-minutes to at least {required}."
                )


def workflow_errors(rel: Path, text: str, ceiling: int, base_text: str) -> Iterator[str]:
    for expression in arithmetic_expressions(text):
        yield (
            f"{rel}: `${{{{ {expression} }}}}` uses arithmetic, which GitHub expressions do not "
            "support. The workflow will fail at startup with no jobs and no check run."
        )

    workflow: Final = parse_workflow(text)
    if isinstance(workflow, str):
        yield f"{rel}: {workflow}"
        return

    yield from timeout_contract_errors(rel, workflow, ceiling, base_text)


def main() -> None:
    base_text: Final = BASE_WORKFLOW_PATH.read_text()
    ceiling: Final = setup_ceiling_minutes(base_text)
    errors: Final = tuple(
        error
        for path in sorted(WORKFLOWS_DIR.glob("*.y*ml"))
        for error in workflow_errors(path.relative_to(REPO_ROOT), path.read_text(), ceiling, base_text)
    )

    if errors:
        raise WorkflowStartupError(
            "Workflow startup invariants violated:\n  - " + "\n  - ".join(errors)
        )

    print(f"Workflow startup invariants hold (setup ceiling {ceiling}m)")


if __name__ == "__main__":
    try:
        main()
    except WorkflowStartupError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
