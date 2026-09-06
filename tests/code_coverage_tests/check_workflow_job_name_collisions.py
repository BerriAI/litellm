"""Catch two workflow jobs that publish check runs under the same name.

A ruleset's required status check names a check run and GitHub matches it by that
name alone. When two jobs publish the same name the required context stops
mapping to the job that proves it: the commit carries two check runs under one
name and nothing says which one the ruleset required. Both being green hides the
clash completely, so the context quietly stops meaning what the ruleset intended.

`.github/workflows/auto-close-duplicates.yml` shipped a job id `test` while
`.github/workflows/test-mcp.yml` already published the required `test` context,
and commit ed5761daef4ae17152446d182c860630c38b7268 carried both check runs.
This invariant has to be enforced here because CI cannot enforce it on itself.

A job publishes its `name:` when it sets one and its job id otherwise. Two shapes
expand that further. A `${{ matrix.key }}` reference publishes one check run per
value the matrix supplies, so two shard lists that overlap collide even though
their templates read differently. A job calling a local reusable workflow
publishes one check run per job of the callee, named `<caller> / <callee>`, which
is why a caller's name never collides with a plain job that happens to match it.
A reference nothing resolves stays in the string, so two jobs carrying the same
unresolved template still compare equal and their collision is still caught.
"""

import itertools
import operator
import re
import sys
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Final

import yaml
from pydantic import BaseModel, Field, ValidationError

REPO_ROOT: Final = Path(__file__).resolve().parent.parent.parent
WORKFLOWS_DIR: Final = REPO_ROOT / ".github" / "workflows"
MATRIX_REF: Final = re.compile(r"\$\{\{\s*matrix\.(?P<key>[\w-]+)\s*\}\}")
LOCAL_CALL_PREFIX: Final = "./"


class CheckRunNameCollision(Exception):
    pass


class Job(BaseModel):
    name: str | None = None
    uses: str | None = None
    strategy: Mapping[str, object] = Field(default_factory=dict)


class Workflow(BaseModel):
    jobs: Mapping[str, Job] = Field(default_factory=dict)


def parse(text: str) -> tuple[Workflow, object] | None:
    """The workflow plus its raw `on:` value, or None when the file is not a workflow."""
    parsed: Final = yaml.safe_load(text)
    if not isinstance(parsed, dict):
        return None
    try:
        return Workflow.model_validate(parsed), parsed.get(True, parsed.get("on"))
    except ValidationError:
        return None


def events(raw_on: object) -> frozenset[str]:
    if isinstance(raw_on, Mapping):
        return frozenset(str(key) for key in raw_on)
    if isinstance(raw_on, str):
        return frozenset({raw_on})
    if isinstance(raw_on, Sequence):
        return frozenset(str(event) for event in raw_on)
    return frozenset()


def publishes_check_runs(raw_on: object) -> bool:
    """A `workflow_call`-only workflow posts its check runs through callers, never itself."""
    return events(raw_on) != frozenset({"workflow_call"})


def matrix_values(job: Job, key: str) -> tuple[str, ...]:
    matrix: Final = job.strategy.get("matrix")
    if not isinstance(matrix, Mapping):
        return ()
    listed: Final = matrix.get(key)
    rows: Final = matrix.get("include")
    from_list: Final = (
        tuple(str(value) for value in listed if isinstance(value, (str, int, float)))
        if isinstance(listed, Sequence) and not isinstance(listed, str)
        else ()
    )
    from_rows: Final = (
        tuple(str(row[key]) for row in rows if isinstance(row, Mapping) and isinstance(row.get(key), (str, int, float)))
        if isinstance(rows, Sequence) and not isinstance(rows, str)
        else ()
    )
    return tuple(dict.fromkeys(from_list + from_rows))


def substituted(template: str, values: Mapping[str, str]) -> str:
    return MATRIX_REF.sub(lambda ref: values.get(ref.group("key"), ref.group(0)), template)


def expand(template: str, job: Job) -> tuple[str, ...]:
    keys: Final = tuple(dict.fromkeys(ref.group("key") for ref in MATRIX_REF.finditer(template)))
    candidates: Final = tuple((key, matrix_values(job, key)) for key in keys)
    resolvable: Final = tuple((key, values) for key, values in candidates if values)
    if not resolvable:
        return (template,)
    return tuple(
        substituted(template, dict(zip((key for key, _ in resolvable), combination)))
        for combination in itertools.product(*(values for _, values in resolvable))
    )


def callee_path(job: Job) -> str | None:
    if job.uses is None or not job.uses.startswith(LOCAL_CALL_PREFIX):
        return None
    return job.uses[len(LOCAL_CALL_PREFIX) :].split("@")[0]


def job_names(job_id: str, job: Job, workflows: Mapping[str, Workflow]) -> tuple[str, ...]:
    prefixes: Final = expand(job.name or job_id, job)
    callee: Final = workflows.get(callee_path(job) or "")
    if callee is None:
        return prefixes
    suffixes: Final = tuple(
        name
        for callee_id, callee_job in callee.jobs.items()
        for name in expand(callee_job.name or callee_id, callee_job)
    )
    return tuple(f"{prefix} / {suffix}" for prefix in prefixes for suffix in suffixes)


def published(sources: Mapping[str, str]) -> Iterator[tuple[str, str]]:
    parsed: Final = {rel: entry for rel, text in sources.items() if (entry := parse(text)) is not None}
    workflows: Final = {rel: workflow for rel, (workflow, _) in parsed.items()}
    for rel, (workflow, raw_on) in parsed.items():
        if not publishes_check_runs(raw_on):
            continue
        for job_id, job in workflow.jobs.items():
            for name in job_names(job_id, job, workflows):
                yield name, f"{rel} job `{job_id}`"


def owners_by_name(sources: Mapping[str, str]) -> Iterator[tuple[str, tuple[str, ...]]]:
    for name, pairs in itertools.groupby(sorted(published(sources)), key=operator.itemgetter(0)):
        yield name, tuple(owner for _, owner in pairs)


def collisions(sources: Mapping[str, str]) -> tuple[str, ...]:
    return tuple(
        f"`{name}` is published by {len(owners)} jobs: {', '.join(owners)}. A required status check matching "
        f"that name cannot say which job proves it; give one of them a distinct `name:` or job id."
        for name, owners in owners_by_name(sources)
        if len(owners) > 1
    )


def workflow_sources() -> Mapping[str, str]:
    """Repo-relative posix paths to text, the keys `uses: ./...` resolves against."""
    return {path.relative_to(REPO_ROOT).as_posix(): path.read_text() for path in sorted(WORKFLOWS_DIR.glob("*.y*ml"))}


def main() -> None:
    sources: Final = workflow_sources()
    found: Final = collisions(sources)
    if found:
        raise CheckRunNameCollision("Check-run names are not unique:\n  - " + "\n  - ".join(found))

    print(f"Check-run names are unique across {len(sources)} workflows")


if __name__ == "__main__":
    try:
        main()
    except CheckRunNameCollision as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
