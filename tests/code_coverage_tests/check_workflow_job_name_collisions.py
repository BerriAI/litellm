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

A job publishes its `name:` when it sets one, and otherwise its job id plus the
values of the combination it runs, the way GitHub writes `build (3.12)`. A name
carrying `${{ ... }}` publishes one check run per combination the matrix
produces: `exclude` rows drop combinations before `include` rows fold into the
survivors, and each `include` row's values stay together rather than crossing
with the other rows', so two shard lists that overlap collide even though their
templates read differently. Each expression is evaluated per combination over the
pieces a job name can hold: string literals, `matrix.<key>`, `format()`, `==` and
`!=`, and the `<cond> && <a> || <b>` idiom, which is how the shards reach their
real `<shard> / Run tests` names rather than staying opaque.

Whatever the sweep cannot work out is left out of the comparison and reported
instead of guessed, because a guess that lands wrong fails a workflow GitHub
would have published perfectly well. A name left holding an expression over
`inputs`, `needs` or `matrix` reads differently from every job that runs it, so
it is one of those; an expression over `github` and the other contexts fixed for
the whole run reads the same everywhere, so two jobs carrying it still clash and
the check still says so. A matrix that is itself an expression or that lists
values which are not scalars, and a call this sweep cannot follow, go in the same
bucket. The cost is that a real clash hiding behind one of them goes unseen,
which leaves a merge no worse off than before this check existed, where the
opposite direction would block work that was fine.

A job calling a local reusable workflow publishes one check run per job of the
callee, named `<caller> / <callee>` and chained through however many levels of
local calls it takes, which is why a caller's name never collides with a plain
job that happens to match it. A file under `.github/workflows/` that does not read as
a workflow at all is reported rather than skipped, since skipping it silently
would hide every job it holds.
"""

import itertools
import operator
import re
import sys
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Final

import yaml
from pydantic import BaseModel, Field, ValidationError

REPO_ROOT: Final = Path(__file__).resolve().parent.parent.parent
WORKFLOWS_DIR: Final = REPO_ROOT / ".github" / "workflows"
EXPRESSION: Final = re.compile(r"\$\{\{(?P<body>.*?)\}\}", re.DOTALL)
MATRIX_REF: Final = re.compile(r"^matrix\.(?P<key>[\w-]+)$")
LITERAL: Final = re.compile(r"^'(?P<text>[^']*)'$")
FORMAT_CALL: Final = re.compile(r"^format\((?P<args>.*)\)$", re.DOTALL)
COMPARISON: Final = re.compile(r"^(?P<left>.+?)\s*(?P<operator>==|!=)\s*(?P<right>.+)$", re.DOTALL)
CONTEXT_REF: Final = re.compile(r"\b(?P<root>[a-z][\w-]*)\s*\.")
RUN_FIXED_CONTEXTS: Final = frozenset({"github", "vars", "env", "runner", "secrets"})
NO_MATRIX: Final[Mapping[str, str]] = MappingProxyType({})
NO_CALLERS: Final[frozenset[str]] = frozenset()
SCALAR: Final = (str, int, float)
MATRIX_DIRECTIVES: Final = frozenset({"include", "exclude"})
LOCAL_CALL_PREFIX: Final = "./"


@dataclass(frozen=True, slots=True)
class Unreadable:
    reason: str


@dataclass(frozen=True, slots=True)
class Opaque:
    reason: str


@dataclass(frozen=True, slots=True)
class Names:
    """The check-run names a job publishes, beside the reasons the rest of them stay unknown."""

    known: tuple[str, ...] = ()
    unknown: tuple[str, ...] = ()


class Job(BaseModel):
    name: object = None
    uses: str | None = None
    strategy: Mapping[str, object] = Field(default_factory=dict)


class Workflow(BaseModel):
    jobs: Mapping[str, Job] = Field(default_factory=dict)


def scalar_text(value: object) -> str:
    """A YAML scalar the way GitHub renders it, so `true` never reaches a name as `True`."""
    return str(value).lower() if isinstance(value, bool) else str(value)


def parse(source: str) -> tuple[Workflow, object] | Unreadable:
    """The workflow plus its raw `on:` value, or why the file does not read as one."""
    parsed: Final = yaml.safe_load(source)
    if not isinstance(parsed, dict):
        return Unreadable("its top level is not a mapping of workflow keys")
    try:
        return Workflow.model_validate(parsed), parsed.get(True, parsed.get("on"))
    except ValidationError as error:
        return Unreadable(f"{error.error_count()} of its job definitions have a shape GitHub would reject")


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


def scalar_list(value: object) -> tuple[str, ...] | Opaque:
    """One matrix key's values, or why the combinations it produces cannot be worked out."""
    if not isinstance(value, Sequence) or isinstance(value, str):
        return Opaque("a matrix key holds something other than a list of values")
    if any(not isinstance(item, SCALAR) for item in value):
        return Opaque("a matrix key lists values that are not plain scalars")
    return tuple(scalar_text(item) for item in value)


def listed_values(matrix: Mapping[str, object]) -> tuple[tuple[str, tuple[str, ...]], ...] | Opaque:
    listed: Final = tuple(
        (str(key), scalar_list(values)) for key, values in matrix.items() if str(key) not in MATRIX_DIRECTIVES
    )
    opaque: Final = next((values for _, values in listed if isinstance(values, Opaque)), None)
    if opaque is not None:
        return opaque
    return tuple((key, values) for key, values in listed if not isinstance(values, Opaque))


def directive_rows(matrix: Mapping[str, object], directive: str) -> tuple[Mapping[str, str], ...]:
    rows: Final = matrix.get(directive)
    if not isinstance(rows, Sequence) or isinstance(rows, str):
        return ()
    return tuple(
        MappingProxyType({str(key): scalar_text(value) for key, value in row.items() if isinstance(value, SCALAR)})
        for row in rows
        if isinstance(row, Mapping)
    )


def drops(row: Mapping[str, str], combination: Mapping[str, str]) -> bool:
    """GitHub removes a combination that carries every value one `exclude` row names."""
    return all(combination.get(key) == value for key, value in row.items())


def extends(row: Mapping[str, str], combination: Mapping[str, str]) -> bool:
    """GitHub folds an `include` row into a combination only where it overwrites no listed value."""
    return all(combination[key] == value for key, value in row.items() if key in combination)


def extended(combination: Mapping[str, str], rows: Sequence[Mapping[str, str]]) -> Mapping[str, str]:
    additions: Final = {key: value for row in rows if extends(row, combination) for key, value in row.items()}
    return MappingProxyType({**combination, **additions})


def crossed_values(listed: Sequence[tuple[str, tuple[str, ...]]]) -> tuple[Mapping[str, str], ...]:
    if not listed:
        return ()
    return tuple(
        MappingProxyType(dict(zip((key for key, _ in listed), values)))
        for values in itertools.product(*(values for _, values in listed))
    )


def matrix_combinations(job: Job) -> tuple[Mapping[str, str], ...] | Opaque:
    """One mapping per job the matrix produces, `exclude` applied before `include` as GitHub does."""
    matrix: Final = job.strategy.get("matrix")
    if matrix is None:
        return ()
    if not isinstance(matrix, Mapping):
        return Opaque("the matrix itself comes from an expression")
    listed: Final = listed_values(matrix)
    if isinstance(listed, Opaque):
        return listed
    rows: Final = directive_rows(matrix, "include")
    dropped: Final = directive_rows(matrix, "exclude")
    kept: Final = tuple(
        combination for combination in crossed_values(listed) if not any(drops(row, combination) for row in dropped)
    )
    standalone: Final = tuple(row for row in rows if not any(extends(row, combination) for combination in kept))
    return (*(extended(combination, rows) for combination in kept), *standalone)


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


def formatted(template: str, arguments: Sequence[str]) -> str | None:
    """A `format()` whose placeholders the arguments cannot fill resolves to nothing, never a crash."""
    try:
        return template.format(*arguments)
    except (IndexError, KeyError, ValueError):
        return None


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
    return formatted(resolved[0], resolved[1:])


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


def stable(name: str) -> bool:
    """An expression over the run's own contexts reads the same from every job, so two of them still clash."""
    return all(
        reference.group("root") in RUN_FIXED_CONTEXTS
        for span in EXPRESSION.finditer(name)
        for reference in CONTEXT_REF.finditer(span.group("body"))
    )


def comparable(name: str) -> bool:
    """A leftover expression over `inputs`, `needs` or `matrix` names a different check run per job."""
    return EXPRESSION.search(name) is None or stable(name)


def settled(names: Sequence[str]) -> Names:
    return Names(
        tuple(name for name in names if comparable(name)),
        tuple(f"its name stays `{name}`" for name in names if not comparable(name)),
    )


def expand(template: str, job: Job) -> Names:
    combinations: Final = matrix_combinations(job)
    if isinstance(combinations, Opaque):
        return Names((), (combinations.reason,))
    over: Final = combinations or (NO_MATRIX,)
    return settled(tuple(dict.fromkeys(rendered(template, values) for values in over)))


def suffixed(job_id: str, combination: Mapping[str, str]) -> str:
    """The name GitHub gives a job with no `name:`, its id plus the combination it runs."""
    return f"{job_id} ({', '.join(combination.values())})" if combination else job_id


def published_names(job_id: str, job: Job) -> Names:
    if job.name is not None:
        return expand(scalar_text(job.name), job)
    combinations: Final = matrix_combinations(job)
    if isinstance(combinations, Opaque):
        return Names((), (combinations.reason,))
    suffixes: Final = tuple(dict.fromkeys(suffixed(job_id, values) for values in combinations))
    return Names(suffixes or (job_id,))


def callee_path(job: Job) -> str | None:
    if job.uses is None or not job.uses.startswith(LOCAL_CALL_PREFIX):
        return None
    return job.uses[len(LOCAL_CALL_PREFIX) :].split("@")[0]


def joined(groups: Sequence[Names]) -> Names:
    return Names(
        tuple(name for group in groups for name in group.known),
        tuple(reason for group in groups for reason in group.unknown),
    )


def call_blocker(job: Job, workflows: Mapping[str, Workflow], callers: frozenset[str]) -> str | None:
    path: Final = callee_path(job)
    if path is None:
        return "it calls a reusable workflow outside this repository"
    if path in callers:
        return f"its call to {path} loops back on itself"
    return None if path in workflows else f"it calls {path}, which this checkout does not hold"


def job_names(job_id: str, job: Job, workflows: Mapping[str, Workflow], callers: frozenset[str] = NO_CALLERS) -> Names:
    prefixes: Final = published_names(job_id, job)
    if job.uses is None:
        return prefixes
    blocker: Final = call_blocker(job, workflows, callers)
    if blocker is not None:
        return Names((), (*prefixes.unknown, blocker))
    path: Final = callee_path(job) or ""
    suffixes: Final = joined(
        tuple(
            job_names(callee_id, callee_job, workflows, callers | {path})
            for callee_id, callee_job in workflows[path].jobs.items()
        )
    )
    return Names(
        tuple(f"{prefix} / {suffix}" for prefix in prefixes.known for suffix in suffixes.known),
        (*prefixes.unknown, *suffixes.unknown),
    )


def readable(sources: Mapping[str, str]) -> Mapping[str, tuple[Workflow, object]]:
    parsed: Final = {rel: parse(source) for rel, source in sources.items()}
    return MappingProxyType({rel: entry for rel, entry in parsed.items() if not isinstance(entry, Unreadable)})


def unreadable(sources: Mapping[str, str]) -> tuple[str, ...]:
    parsed: Final = {rel: parse(source) for rel, source in sources.items()}
    return tuple(
        f"{rel} sits in the workflows directory but {entry.reason}, so none of its jobs were checked."
        for rel, entry in sorted(parsed.items())
        if isinstance(entry, Unreadable)
    )


def scanned_jobs(sources: Mapping[str, str]) -> Iterator[tuple[str, str, Names]]:
    parsed: Final = readable(sources)
    workflows: Final = {rel: workflow for rel, (workflow, _) in parsed.items()}
    for rel, (workflow, raw_on) in parsed.items():
        if not publishes_check_runs(raw_on):
            continue
        for job_id, job in workflow.jobs.items():
            yield rel, job_id, job_names(job_id, job, workflows)


def published(sources: Mapping[str, str]) -> Iterator[tuple[str, str]]:
    for rel, job_id, names in scanned_jobs(sources):
        for name in names.known:
            yield name, f"{rel} job `{job_id}`"


def blind_spots(sources: Mapping[str, str]) -> tuple[str, ...]:
    """Jobs whose published names GitHub decides at run time, which no offline sweep can compare."""
    return tuple(
        f"{rel} job `{job_id}` publishes a name this check cannot work out because {reason}."
        for rel, job_id, names in scanned_jobs(sources)
        for reason in sorted(names.unknown)
    )


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


def report(header: str, problems: Sequence[str]) -> None:
    if problems:
        print(f"ERROR: {header}:\n  - " + "\n  - ".join(problems), file=sys.stderr)


def exit_code(sources: Mapping[str, str]) -> int:
    unread: Final = unreadable(sources)
    found: Final = collisions(sources)
    blind: Final = blind_spots(sources)
    if blind:
        print("NOTE: names left out of the comparison:\n  - " + "\n  - ".join(blind))
    report("Some workflows could not be read", unread)
    report("Check-run names are not unique", found)
    if unread or found:
        return 1

    print(f"Check-run names are unique across {len(sources)} workflows")
    return 0


def main() -> int:
    return exit_code(workflow_sources())


if __name__ == "__main__":
    sys.exit(main())
