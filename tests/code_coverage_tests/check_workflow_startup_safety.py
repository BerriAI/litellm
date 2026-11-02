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
2. Unit matrix rows keep the job timeout at or above
   the test budget plus the setup ceilings plus the runner overhead below.
   Otherwise the job deadline preempts pytest inside its own advertised budget,
   which is the failure the split timeouts exist to prevent, and it shows up as
   a cancelled shard whose tests were passing. A budget this check cannot resolve
   is reported rather than skipped, so a mistyped input or matrix column surfaces
   here instead of leaving the pair silently unchecked.
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
UNIT_WORKFLOW_PATH: Final = WORKFLOWS_DIR / "test-unit.yml"

# Runner time the job clock charges but no step owns: job init, the gaps between
# steps, and post-job cleanup. Without it a job capped at exactly test + setup
# would still preempt pytest inside its own budget.
JOB_OVERHEAD_MINUTES: Final = 5

EXPRESSION: Final = re.compile(r"\$\{\{(?P<body>.*?)\}\}", re.DOTALL)
QUOTED: Final = re.compile(r"'[^']*'")
ARITHMETIC: Final = re.compile(r"[+*]")
JOB_DEADLINE: Final = "${{ matrix.job-timeout-minutes }}"
TEST_DEADLINE: Final = "${{ matrix.timeout-minutes }}"


class WorkflowStartupError(Exception):
    pass


class WorkflowJob(BaseModel):
    strategy: Mapping[str, object] = Field(default_factory=dict)
    steps: tuple[Mapping[str, object], ...] = ()
    timeout_minutes: object = Field(default=None, alias="timeout-minutes")

    model_config = {"populate_by_name": True, "frozen": True}


class WorkflowFile(BaseModel):
    jobs: Mapping[str, WorkflowJob] = Field(default_factory=dict)

    model_config = {"frozen": True}


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


def setup_ceiling_minutes(unit_text: str) -> int:
    """Sum the per-step timeouts on everything the unit job runs before pytest."""
    workflow: Final = yaml.safe_load(unit_text)
    steps: Final = workflow["jobs"]["unit"]["steps"]
    return sum(
        s["timeout-minutes"]
        for s in steps
        if s.get("name") != "Run tests" and isinstance(s.get("timeout-minutes"), int)
    )


def timeout_contract_errors(rel: Path, workflow: WorkflowFile, ceiling: int) -> Iterator[str]:
    job: Final = workflow.jobs.get("unit")
    if job is None:
        return
    if job.timeout_minutes != JOB_DEADLINE:
        yield f"{rel}: job `unit` sets timeout-minutes to `{job.timeout_minutes}`, not `{JOB_DEADLINE}`"
    test_deadlines: Final = tuple(s.get("timeout-minutes") for s in job.steps if s.get("name") == "Run tests")
    if test_deadlines != (TEST_DEADLINE,):
        yield f"{rel}: job `unit` step `Run tests` must set timeout-minutes to `{TEST_DEADLINE}`, found {test_deadlines}"
    matrix_value: Final = job.strategy.get("matrix", {})
    if not isinstance(matrix_value, Mapping):
        yield f"{rel}: job `unit` has no readable matrix"
        return
    entries_value: Final = matrix_value.get("include", ())
    if not isinstance(entries_value, Sequence) or isinstance(entries_value, str) or not entries_value:
        yield f"{rel}: job `unit` has no readable matrix include rows"
        return
    for entry in entries_value:
        if not isinstance(entry, Mapping):
            yield f"{rel}: job `unit` has an unreadable matrix row"
            continue
        shard: Final = entry.get("shard", "<unnamed>")
        test_budget: Final = entry.get("timeout-minutes")
        job_budget: Final = entry.get("job-timeout-minutes")
        where: Final = f"{rel}: job `unit`, shard `{shard}`"
        if not isinstance(test_budget, int) or isinstance(test_budget, bool):
            yield f"{where} has no integer timeout-minutes"
            continue
        if not isinstance(job_budget, int) or isinstance(job_budget, bool):
            yield f"{where} has no integer job-timeout-minutes"
            continue
        required: Final = test_budget + ceiling + JOB_OVERHEAD_MINUTES
        if job_budget < required:
            yield (
                f"{where} gives pytest {test_budget}m but caps the job at "
                f"{job_budget}m. Setup can use up to {ceiling}m plus {JOB_OVERHEAD_MINUTES}m of "
                f"runner overhead, so the job deadline would preempt pytest; raise "
                f"job-timeout-minutes to at least {required}."
            )


def workflow_errors(rel: Path, text: str, ceiling: int) -> Iterator[str]:
    for expression in arithmetic_expressions(text):
        yield (
            f"{rel}: `${{{{ {expression} }}}}` uses arithmetic, which GitHub expressions do not "
            "support. The workflow will fail at startup with no jobs and no check run."
        )

    workflow: Final = parse_workflow(text)
    if isinstance(workflow, str):
        yield f"{rel}: {workflow}"
        return

    yield from timeout_contract_errors(rel, workflow, ceiling)


def main() -> None:
    unit_text: Final = UNIT_WORKFLOW_PATH.read_text()
    ceiling: Final = setup_ceiling_minutes(unit_text)
    errors: Final = tuple(
        error
        for path in sorted(WORKFLOWS_DIR.glob("*.y*ml"))
        for error in workflow_errors(path.relative_to(REPO_ROOT), path.read_text(), ceiling)
    )

    if errors:
        raise WorkflowStartupError("Workflow startup invariants violated:\n  - " + "\n  - ".join(errors))

    print(f"Workflow startup invariants hold (setup ceiling {ceiling}m)")


if __name__ == "__main__":
    try:
        main()
    except WorkflowStartupError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
