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
expand that further. A name carrying `${{ ... }}` publishes one check run per
matrix combination it reads, so two shard lists that overlap collide even though
their templates read differently. Each expression is evaluated per combination
over the pieces a job name can hold: string literals, `matrix.<key>`, `format()`,
`==` and `!=`, and the `<cond> && <a> || <b>` idiom, which is how the shards
reach their real `<shard> / Run tests` names rather than staying opaque. A job
calling a local reusable workflow publishes one check run per job of the callee,
named `<caller> / <callee>`, which is why a caller's name never collides with a
plain job that happens to match it. An expression nothing resolves stays in the
string, so two jobs carrying the same unresolved template still compare equal and
their collision is still caught.
"""

import itertools
import operator
import re
import sys
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import Final

import yaml
from pydantic import BaseModel, Field, ValidationError

REPO_ROOT: Final = Path(__file__).resolve().parent.parent.parent
WORKFLOWS_DIR: Final = REPO_ROOT / ".github" / "workflows"
EXPRESSION: Final = re.compile(r"\$\{\{(?P<body>.*?)\}\}", re.DOTALL)
MATRIX_KEY: Final = re.compile(r"matrix\.(?P<key>[\w-]+)")
MATRIX_REF: Final = re.compile(r"^matrix\.(?P<key>[\w-]+)$")
LITERAL: Final = re.compile(r"^'(?P<text>[^']*)'$")
FORMAT_CALL: Final = re.compile(r"^format\((?P<args>.*)\)$", re.DOTALL)
COMPARISON: Final = re.compile(r"^(?P<left>.+?)\s*(?P<operator>==|!=)\s*(?P<right>.+)$", re.DOTALL)
NO_MATRIX: Final[Mapping[str, str]] = MappingProxyType({})
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


def scanned(state: tuple[int, bool], char: str) -> tuple[int, bool]:
    depth, quoted = state
    if char == "'":
        return depth, not quoted
    if quoted:
        return depth, quoted
    return depth + int(char == "(") - int(char == ")"), quoted


def split_outside(text: str, token: str) -> tuple[str, ...]:
    """`text` cut on every `token` that sits outside quotes and parentheses."""
    states: Final = tuple(itertools.accumulate(text, scanned, initial=(0, False)))
    cuts: Final = tuple(
        index
        for index in range(len(text) - len(token) + 1)
        if text.startswith(token, index) and states[index] == (0, False)
    )
    starts: Final = (0, *(cut + len(token) for cut in cuts))
    return tuple(text[start:end] for start, end in zip(starts, (*cuts, len(text))))


def value_of(text: str, values: Mapping[str, str]) -> str | None:
    expression: Final = text.strip()
    literal: Final = LITERAL.match(expression)
    if literal is not None:
        return literal.group("text")
    reference: Final = MATRIX_REF.match(expression)
    if reference is not None:
        return values.get(reference.group("key"))
    call: Final = FORMAT_CALL.match(expression)
    if call is None:
        return None
    arguments: Final = tuple(value_of(part, values) for part in split_outside(call.group("args"), ","))
    resolved: Final = tuple(argument for argument in arguments if argument is not None)
    if not resolved or len(resolved) != len(arguments):
        return None
    return resolved[0].format(*resolved[1:])


def holds(condition: str, values: Mapping[str, str]) -> bool | None:
    comparison: Final = COMPARISON.match(condition.strip())
    if comparison is None:
        return None
    left: Final = value_of(comparison.group("left"), values)
    right: Final = value_of(comparison.group("right"), values)
    if left is None or right is None:
        return None
    return (left == right) == (comparison.group("operator") == "==")


def evaluate(body: str, values: Mapping[str, str]) -> str | None:
    """The single string this expression yields, or None when its shape is not understood."""
    branches: Final = tuple(split_outside(alternative, "&&") for alternative in split_outside(body, "||"))
    outcomes: Final = tuple(tuple(holds(part, values) for part in branch[:-1]) for branch in branches)
    if any(outcome is None for branch in outcomes for outcome in branch):
        return None
    taken: Final = next((branch[-1] for branch, outcome in zip(branches, outcomes) if all(outcome)), None)
    return None if taken is None else value_of(taken, values)


def resolved_span(span: re.Match[str], values: Mapping[str, str]) -> str:
    substitution: Final = evaluate(span.group("body"), values)
    return span.group(0) if substitution is None else substitution


def rendered(template: str, values: Mapping[str, str]) -> str:
    return EXPRESSION.sub(lambda span: resolved_span(span, values), template)


def expand(template: str, job: Job) -> tuple[str, ...]:
    keys: Final = tuple(dict.fromkeys(ref.group("key") for ref in MATRIX_KEY.finditer(template)))
    resolvable: Final = tuple((key, values) for key in keys if (values := matrix_values(job, key)))
    if not resolvable:
        return (rendered(template, NO_MATRIX),)
    return tuple(
        dict.fromkeys(
            rendered(template, dict(zip((key for key, _ in resolvable), combination)))
            for combination in itertools.product(*(values for _, values in resolvable))
        )
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
