from __future__ import annotations

import ast
import operator
import pathlib
import re
import sys
import warnings
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Final

import yaml

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
WORKFLOW_DIR = REPO_ROOT / ".github" / "workflows"
CIRCLECI_CONFIG = REPO_ROOT / ".circleci" / "config.yml"
ALLOWLIST_FILE = REPO_ROOT / ".github" / "ci-coverage-allowlist.yml"
TESTS_ROOT = REPO_ROOT / "tests"

ALLOWLIST_KEYS = frozenset({"description", "test_paths", "dockerfiles"})
PATH_FILTER_KEYS = frozenset({"paths", "paths-ignore"})
TEST_PATH_KEYS = frozenset({"test-path", "test-paths"})
DOCKERFILE_INPUT_KEYS = frozenset({"file", "dockerfile"})
TEST_RUNNER_RE = re.compile(r"\bpytest\b|\bcircleci tests\b|\bhelm unittest\b|\bplaywright test\b|\bpython[0-9.]*\s")
IMAGE_BUILD_RE = re.compile(r"\bdocker\s+(?:buildx\s+)?build\b")
TEST_TOKEN_RE = re.compile(r"tests/[A-Za-z0-9_./*?-]+")
DOCKERFILE_TOKEN_RE = re.compile(r"[A-Za-z0-9_./-]*Dockerfile[A-Za-z0-9_.-]*")
COMMENT_RE = re.compile(r"^\s*#.*$", re.MULTILINE)
GLOB_CHARS = frozenset("*?")

# Trees whose jobs are sharded with no catch-all bucket, so every child that holds
# tests has to be named by some shard or it runs nowhere. A child listed here is
# itself decomposed one level deeper and is checked through its own entry.
SHARDED_ROOTS: tuple[str, ...] = (
    "tests/proxy_unit_tests",
    "tests/test_litellm",
    "tests/test_litellm/proxy",
)


@dataclass(frozen=True, slots=True)
class AllowEntry:
    paths: tuple[str, ...]
    reason: str


@dataclass(frozen=True, slots=True)
class Allowlist:
    test_paths: tuple[AllowEntry, ...]
    dockerfiles: tuple[AllowEntry, ...]

    def covers_test(self, relative_path: str) -> bool:
        return any(_token_covers(path, relative_path) for entry in self.test_paths for path in entry.paths)

    def covers_dockerfile(self, relative_path: str) -> bool:
        return any(relative_path == path for entry in self.dockerfiles for path in entry.paths)


@dataclass(frozen=True, slots=True)
class Section:
    name: str
    entries: tuple[AllowEntry, ...]
    candidates: tuple[str, ...]
    matches: Callable[[str, str], bool]


@dataclass(frozen=True, slots=True)
class Scalar:
    key: str
    value: str


@dataclass(frozen=True, slots=True)
class Finding:
    subject: str
    detail: str


def _scalars(node: object, key: str) -> tuple[Scalar, ...]:
    if isinstance(node, str):
        return (Scalar(key=key, value=node),)
    if isinstance(node, Mapping):
        return tuple(
            scalar
            for child_key, value in node.items()
            if child_key not in PATH_FILTER_KEYS
            for scalar in _scalars(value, str(child_key))
        )
    if isinstance(node, Sequence):
        return tuple(scalar for item in node for scalar in _scalars(item, key))
    return ()


def _config_files() -> tuple[pathlib.Path, ...]:
    workflows = tuple(sorted(path for path in WORKFLOW_DIR.iterdir() if path.suffix in (".yml", ".yaml")))
    circleci = (CIRCLECI_CONFIG,) if CIRCLECI_CONFIG.is_file() else ()
    return workflows + circleci


def _all_scalars() -> tuple[Scalar, ...]:
    return tuple(
        scalar
        for path in _config_files()
        for scalar in _scalars(yaml.safe_load(path.read_text(encoding="utf-8")), path.name)
    )


def _uncommented(value: str) -> str:
    return COMMENT_RE.sub("", value)


def _invoked_test_tokens(scalars: Iterable[Scalar]) -> frozenset[str]:
    return frozenset(
        match.group(0).rstrip("/")
        for scalar in scalars
        if scalar.key in TEST_PATH_KEYS or TEST_RUNNER_RE.search(scalar.value)
        for match in TEST_TOKEN_RE.finditer(_uncommented(scalar.value))
    )


def _built_dockerfile_tokens(scalars: Iterable[Scalar]) -> frozenset[str]:
    return frozenset(
        match.group(0)
        for scalar in scalars
        if scalar.key in DOCKERFILE_INPUT_KEYS or IMAGE_BUILD_RE.search(scalar.value)
        for match in DOCKERFILE_TOKEN_RE.finditer(_uncommented(scalar.value))
    )


def _glob_to_regex(token: str, *, subtree: bool) -> re.Pattern[str]:
    parts = re.split(r"(\*\*/|\*\*|\*|\?|\[[^\]]*\])", token)
    translated = "".join(
        {"**/": r"(?:.*/)?", "**": r".*", "*": r"[^/]*", "?": r"[^/]"}.get(part)
        or (part if part.startswith("[") and part.endswith("]") else re.escape(part))
        for part in parts
    )
    return re.compile(rf"{translated}(?:/.*)?$" if subtree else rf"{translated}$")


def _token_covers(token: str, relative_path: str) -> bool:
    if GLOB_CHARS & set(token):
        return _glob_to_regex(token, subtree=True).match(relative_path) is not None
    return relative_path == token or relative_path.startswith(f"{token}/")


def _token_names(token: str, relative_path: str) -> bool:
    """Whether the token names this path itself, rather than merely containing it.

    A sharded tree has no catch-all bucket, so the ancestor token the census is happy
    with (`tests/x` standing in for everything below it) is exactly what would let a
    newly added child ride along without a shard.
    """
    if GLOB_CHARS & set(token):
        return _glob_to_regex(token, subtree=False).match(relative_path) is not None
    return token == relative_path


def _test_files() -> tuple[str, ...]:
    return tuple(
        sorted(
            path.relative_to(REPO_ROOT).as_posix()
            for path in TESTS_ROOT.rglob("test_*.py")
            if path.is_file() and "node_modules" not in path.parts
        )
    )


def _dockerfiles() -> tuple[str, ...]:
    return tuple(
        sorted(
            path.relative_to(REPO_ROOT).as_posix()
            for path in REPO_ROOT.rglob("Dockerfile*")
            if path.is_file()
            and ".git" not in path.parts
            and "node_modules" not in path.parts
            and not path.name.endswith(".dockerignore")
        )
    )


def _uncovered_tests(allowlist: Allowlist, tokens: frozenset[str]) -> tuple[Finding, ...]:
    uncovered = tuple(
        relative_path
        for relative_path in _test_files()
        if not any(_token_covers(token, relative_path) for token in tokens) and not allowlist.covers_test(relative_path)
    )
    directories = tuple(dict.fromkeys(path.rsplit("/", 1)[0] for path in uncovered))
    return tuple(
        Finding(
            subject=directory,
            detail=_describe(tuple(p for p in uncovered if p.rsplit("/", 1)[0] == directory)),
        )
        for directory in directories
    )


def _describe(paths: tuple[str, ...]) -> str:
    names = ", ".join(path.rsplit("/", 1)[1] for path in paths[:3])
    suffix = f", +{len(paths) - 3} more" if len(paths) > 3 else ""
    return f"{len(paths)} test file(s) invoked by no job: {names}{suffix}"


GLOB_CALL_RE = re.compile(r'circleci tests glob "([^"]+)"')
KEYWORD_RE = re.compile(r"-k\s+\\?[\"']([^\"'\\]+)")


@dataclass(frozen=True, slots=True)
class Slice:
    """One job's selection: the files it globs, narrowed by its `-k` expression."""

    job: str
    globs: tuple[str, ...]
    named: frozenset[str]
    required: tuple[str, ...]
    excluded: tuple[str, ...]
    understood: bool

    def claims(self, relative_path: str, inner_names: frozenset[str]) -> bool:
        """Whether this job runs any test in the file.

        The question is deliberately per-file, not per-test. An excluded term is only
        honoured when it appears in the path, because that is the case where it takes
        the whole module with it; a term matching one function inside drops that test
        and leaves the file claimed. Losing a whole file is the failure worth a gate,
        and answering per-test would mean a baseline of test ids that churns on every
        rename.
        """
        if relative_path in self.named:
            return True
        if not any(_token_covers(glob, relative_path) for glob in self.globs):
            return False
        if not self.understood:
            return True  # a `-k` this parser cannot model is assumed to claim everything
        if any(term.lower() in relative_path.lower() for term in self.excluded):
            return False
        return not self.required or any(
            term.lower() in name.lower() for term in self.required for name in inner_names
        )


def _strings(node: object) -> Iterable[str]:
    if isinstance(node, str):
        yield node
    elif isinstance(node, dict):
        for value in node.values():
            yield from _strings(value)
    elif isinstance(node, list):
        for value in node:
            yield from _strings(value)


def _keyword_terms(
    expressions: Sequence[str], *, attributable: bool = True
) -> tuple[tuple[str, ...], tuple[str, ...], bool]:
    """A `-k` expression as (required, excluded, understood).

    Only flat `and` chains of bare terms are modelled. Anything with `or`, parentheses
    or negation of a group is left unmodelled, and its job is then treated as claiming
    every file it globs, so an unparsed selector can never raise a false alarm.

    `attributable` is False when a job runs several pytest commands, since a selector
    read out of the job's text cannot then be tied to the glob it belongs to, and
    pairing one command's exclusion with another's glob would invent a gap.
    """
    terms: Final = tuple(part.strip() for expression in expressions for part in expression.split(" and "))
    if not attributable and terms:
        return (), (), False
    if any(("or " in term) or ("(" in term) or (term.startswith("not ") and " " in term[4:]) for term in terms):
        return (), (), False
    return (
        tuple(term for term in terms if term and not term.startswith("not ")),
        tuple(term[4:].strip() for term in terms if term.startswith("not ")),
        True,
    )


def _slices() -> tuple[Slice, ...]:
    if not CIRCLECI_CONFIG.exists():
        return ()
    jobs: Final = yaml.safe_load(CIRCLECI_CONFIG.read_text()).get("jobs", {})
    return tuple(
        Slice(job=job, globs=globs, named=named, required=required, excluded=excluded, understood=understood)
        for job, body in jobs.items()
        for text in ("\n".join(_strings(body)),)
        if "pytest" in text
        for globs in (tuple(GLOB_CALL_RE.findall(text)),)
        for named in (frozenset(TEST_TOKEN_RE.findall(text)) & frozenset(_test_files()),)
        for required, excluded, understood in (
            _keyword_terms(tuple(KEYWORD_RE.findall(text)), attributable=len(globs) < 2),
        )
        if globs or named
    )


def _matchable_names(relative_path: str) -> frozenset[str]:
    """Every name a `-k` term can match for this file: its path, plus the names inside it.

    pytest matches a keyword against an item's own name and each of its parents', so a
    positive term hits a file when it appears in the path or in a class or function name.
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # test files carry stray escapes; their names still parse
            tree: Final = ast.parse((REPO_ROOT / relative_path).read_text())
    except (OSError, SyntaxError):
        return frozenset({relative_path})
    return frozenset({relative_path}) | frozenset(
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    )


def _workflow_named_tokens() -> frozenset[str]:
    """Test tokens a GitHub Actions job names directly.

    A CircleCI `-k` that deselects a file no longer means the file runs nowhere once a
    workflow names it, so the slice check has to credit those the same way the census does.
    """
    return _invoked_test_tokens(
        scalar
        for path in _config_files()
        if path != CIRCLECI_CONFIG
        for scalar in _scalars(yaml.safe_load(path.read_text(encoding="utf-8")), path.name)
    )


def _deselected_everywhere(allowlist: Allowlist) -> tuple[Finding, ...]:
    slices: Final = _slices()
    named_by_workflow: Final = _workflow_named_tokens()
    globbed: Final = tuple(
        path
        for path in _test_files()
        if any(_token_covers(glob, path) for slice_ in slices for glob in slice_.globs)
    )
    return tuple(
        Finding(
            subject=path,
            detail="globbed by a job, then deselected by every one of their -k expressions",
        )
        for path in globbed
        if not allowlist.covers_test(path)
        and not any(_token_covers(token, path) for token in named_by_workflow)
        and not any(slice_.claims(path, _matchable_names(path)) for slice_ in slices)
    )


def _holds_tests(directory: pathlib.Path) -> bool:
    return any(directory.rglob("test_*.py"))


def _shard_children(root: str, repo_root: pathlib.Path = REPO_ROOT) -> tuple[str, ...]:
    """Children of a sharded root that carry tests, so each one needs its own shard.

    A directory earns an entry by containing a test file rather than by being named
    `test_*`, which is what keeps fixture directories (`test_configs`, `expected_*`)
    out without a hand-maintained list of exceptions.
    """
    return tuple(
        sorted(
            child.relative_to(repo_root).as_posix()
            for child in (repo_root / root).iterdir()
            if not child.name.startswith(".")
            and (
                _holds_tests(child)
                if child.is_dir()
                else child.name.startswith("test_") and child.suffix == ".py"
            )
        )
    )


def _unassigned_shard_children(
    tokens: frozenset[str],
    roots: tuple[str, ...] = SHARDED_ROOTS,
    repo_root: pathlib.Path = REPO_ROOT,
) -> tuple[Finding, ...]:
    return tuple(
        Finding(subject=child, detail=f"holds tests but no shard of {root} names it")
        for root in roots
        if (repo_root / root).is_dir()
        for child in _shard_children(root, repo_root)
        if child not in roots and not any(_token_names(token, child) for token in tokens)
    )


def _uncovered_dockerfiles(allowlist: Allowlist, tokens: frozenset[str]) -> tuple[Finding, ...]:
    return tuple(
        Finding(subject=relative_path, detail="built by no job")
        for relative_path in _dockerfiles()
        if relative_path not in tokens and not allowlist.covers_dockerfile(relative_path)
    )


def _stale_allowlist_paths(
    allowlist: Allowlist,
    *,
    test_files: tuple[str, ...],
    dockerfiles: tuple[str, ...],
) -> tuple[Finding, ...]:
    sections: Final[tuple[Section, ...]] = (
        Section("test_paths", allowlist.test_paths, test_files, _token_covers),
        Section("dockerfiles", allowlist.dockerfiles, dockerfiles, operator.eq),
    )
    return tuple(
        Finding(subject=path, detail=f"listed under '{section.name}' but matches no file the census looks at")
        for section in sections
        for entry in section.entries
        for path in entry.paths
        if not any(section.matches(path, candidate) for candidate in section.candidates)
    )


def _parse_entry(item: object, section: str) -> AllowEntry:
    if not isinstance(item, dict):
        raise SystemExit(f"{ALLOWLIST_FILE.name}: '{section}' entries must be mappings")
    paths = item.get("paths")
    reason = item.get("reason")
    if (
        not isinstance(paths, list)
        or not paths
        or not all(isinstance(path, str) for path in paths)
        or not isinstance(reason, str)
        or not reason.strip()
    ):
        raise SystemExit(
            f"{ALLOWLIST_FILE.name}: every '{section}' entry needs a non-empty 'paths' "
            "list of strings and a non-empty 'reason'"
        )
    return AllowEntry(paths=tuple(paths), reason=reason)


def _parse_entries(raw: object, section: str) -> tuple[AllowEntry, ...]:
    if not isinstance(raw, list):
        raise SystemExit(f"{ALLOWLIST_FILE.name}: '{section}' must be a list")
    return tuple(_parse_entry(item, section) for item in raw)


def _load_allowlist() -> Allowlist:
    if not ALLOWLIST_FILE.is_file():
        return Allowlist(test_paths=(), dockerfiles=())
    raw = yaml.safe_load(ALLOWLIST_FILE.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise SystemExit(f"{ALLOWLIST_FILE.name}: top level must be a mapping")
    unknown = sorted(str(key) for key in raw if key not in ALLOWLIST_KEYS)
    if unknown:
        raise SystemExit(
            f"{ALLOWLIST_FILE.name}: unknown top-level key(s) {unknown}; expected only {sorted(ALLOWLIST_KEYS)}"
        )
    return Allowlist(
        test_paths=_parse_entries(raw.get("test_paths", []), "test_paths"),
        dockerfiles=_parse_entries(raw.get("dockerfiles", []), "dockerfiles"),
    )


def _write(message: str) -> None:
    sys.stdout.write(f"{message}\n")


def _report(title: str, findings: tuple[Finding, ...], remedy: str) -> None:
    _write(f"ERROR: {title}")
    for finding in findings:
        _write(f"  - {finding.subject}: {finding.detail}")
    _write("")
    _write(remedy)
    _write("")


def _check_slices() -> int:
    findings: Final = _deselected_everywhere(_load_allowlist())
    if findings:
        _report(
            "test files a -k expression removes from every job that globs them",
            findings,
            "Give each one a job whose -k keeps it, or list it in "
            ".github/ci-coverage-allowlist.yml with the reason it may stay unrun.",
        )
        return 1

    _write(f"OK: no test file is globbed by a job and then deselected by every -k across {len(_slices())} slices.")
    return 0


def _check_shards() -> int:
    findings = _unassigned_shard_children(_invoked_test_tokens(_all_scalars()))
    if findings:
        _report(
            "test directories and files that no shard claims",
            findings,
            "Add each to the shard it belongs to. A directory that is itself split across "
            "several shards belongs in SHARDED_ROOTS instead, so its own children get checked.",
        )
        return 1

    counted = sum(len(_shard_children(root)) for root in SHARDED_ROOTS if (REPO_ROOT / root).is_dir())
    _write(f"OK: all {counted} test children across {len(SHARDED_ROOTS)} sharded trees are assigned to a shard.")
    return 0


def main() -> int:
    if "--shards" in sys.argv[1:]:
        return _check_shards()
    if "--slices" in sys.argv[1:]:
        return _check_slices()

    allowlist = _load_allowlist()
    scalars = _all_scalars()

    test_findings = _uncovered_tests(allowlist, _invoked_test_tokens(scalars))
    dockerfile_findings = _uncovered_dockerfiles(allowlist, _built_dockerfile_tokens(scalars))
    stale_findings = _stale_allowlist_paths(allowlist, test_files=_test_files(), dockerfiles=_dockerfiles())

    if stale_findings:
        _report(
            "allowlist entries that exempt nothing",
            stale_findings,
            "Delete each from .github/ci-coverage-allowlist.yml; the file it named is gone or was renamed.",
        )
    if test_findings:
        _report(
            "test files that no CI job invokes",
            test_findings,
            "Add each to a job's test path, or list it in .github/ci-coverage-allowlist.yml with a reason.",
        )
    if dockerfile_findings:
        _report(
            "Dockerfiles that no CI job builds",
            dockerfile_findings,
            "Build each in a workflow, or list it in .github/ci-coverage-allowlist.yml with a reason.",
        )
    if stale_findings or test_findings or dockerfile_findings:
        return 1

    _write(
        f"OK: {len(_test_files())} test files and {len(_dockerfiles())} Dockerfiles are each "
        "invoked by at least one job or carry an explicit allowlist entry."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
