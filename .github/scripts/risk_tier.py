from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Final, Literal, TypeAlias

import yaml
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError
from typing_extensions import ReadOnly, TypedDict

Tier: TypeAlias = Literal["low", "medium", "high"]
ReadFile: TypeAlias = Callable[[str, str], str | None]
RewriteCheck: TypeAlias = Callable[[str | None, str | None], str | None]

TEST_DEF_RE: Final = re.compile(r"^\s*(?:async\s+)?def\s+test_|^\s*(?:it|test)\(")
SKIP_RE: Final = re.compile(
    r"pytest\.mark\.(?:skip(?!if)|xfail)|unittest\.skip\b"
    r"|\b(?:it|test|describe)\.(?:skip\(\s*[\"'`]|only\()|\bx(?:it|test|describe)\("
)
SKIP_CALL_RE: Final = re.compile(r"\bpytest\.(?:skip|importorskip)\(|\b(?:it|test|describe)\.fixme\(")
ASSERT_RE: Final = re.compile(r"^\s*assert\b|\bexpect\(")
GLOB_TOKEN_RE: Final = re.compile(r"(\*\*/|\*\*|\*|\?)")
DIFF_BLOCK_SEPARATOR: Final = "\ndiff --git "
RENAME_FROM_RE: Final = re.compile(r"^rename from (.+)$")
RENAME_TO_RE: Final = re.compile(r"^rename to (.+)$")
QUOTED_PATH_ESCAPE_RE: Final = re.compile(r'\\(?:([abfnrtv"\\])|([0-7]{3}))')
QUOTED_PATH_ESCAPES: Final = MappingProxyType(
    {"a": "\a", "b": "\b", "f": "\f", "n": "\n", "r": "\r", "t": "\t", "v": "\v", '"': '"', "\\": "\\"}
)
MAX_LISTED_PATHS: Final = 5
JSON_OBJECT: Final = TypeAdapter(dict[str, object])


class SizeLimit(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    lines_under: int = Field(gt=0)
    files_up_to: int = Field(gt=0)


class SizeRules(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    low: SizeLimit
    medium: SizeLimit
    ignore: tuple[str, ...] = ()


class ModuleRules(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    medium_from: int = Field(gt=1)
    high_from: int = Field(gt=1)


class PathRules(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    high: tuple[str, ...]
    medium: tuple[str, ...] = ()
    low: tuple[str, ...]


class TestRules(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    files: tuple[str, ...]


class AuthorRules(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    low: tuple[str, ...]


class GuardRules(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    additive_rows: tuple[str, ...] = ()
    lowered_limits: tuple[str, ...] = ()


class RiskConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    paths: PathRules
    modules: ModuleRules
    size: SizeRules
    tests: TestRules
    authors: AuthorRules
    guards: GuardRules = GuardRules()


@dataclass(frozen=True, slots=True)
class PathMatcher:
    patterns: tuple[re.Pattern[str], ...]

    @staticmethod
    def from_globs(globs: Sequence[str]) -> PathMatcher:
        return PathMatcher(tuple(glob_to_regex(glob) for glob in globs))

    def matches(self, path: str) -> bool:
        return any(pattern.fullmatch(path) for pattern in self.patterns)


@dataclass(frozen=True, slots=True)
class FileChange:
    path: str
    added_lines: tuple[str, ...]
    deleted_lines: tuple[str, ...]
    previous_path: str | None = None
    guarded_rewrite: str | None = None

    @property
    def paths(self) -> tuple[str, ...]:
        return (self.path,) if self.previous_path is None else (self.previous_path, self.path)

    @property
    def line_count(self) -> int:
        return len(self.added_lines) + len(self.deleted_lines)


@dataclass(frozen=True, slots=True)
class Rules:
    config: RiskConfig
    high_paths: PathMatcher
    medium_paths: PathMatcher
    low_paths: PathMatcher
    test_files: PathMatcher
    size_ignored: PathMatcher
    additive_rows: PathMatcher
    lowered_limits: PathMatcher

    @staticmethod
    def from_config(config: RiskConfig) -> Rules:
        return Rules(
            config=config,
            high_paths=PathMatcher.from_globs(config.paths.high),
            medium_paths=PathMatcher.from_globs(config.paths.medium),
            low_paths=PathMatcher.from_globs(config.paths.low),
            test_files=PathMatcher.from_globs(config.tests.files),
            size_ignored=PathMatcher.from_globs(config.size.ignore),
            additive_rows=PathMatcher.from_globs(config.guards.additive_rows),
            lowered_limits=PathMatcher.from_globs(config.guards.lowered_limits),
        )

    def path_tier(self, path: str) -> Tier:
        if self.high_paths.matches(path):
            return "high"
        if self.medium_paths.matches(path):
            return "medium"
        if self.test_files.matches(path) or self.low_paths.matches(path):
            return "low"
        return "medium"

    def change_tier(self, change: FileChange) -> Tier:
        rewrite_tier: Final[Tier] = "low" if change.guarded_rewrite is None else "medium"
        return highest((rewrite_tier, *(self.path_tier(path) for path in change.paths)))

    def rewrite_check(self, path: str) -> RewriteCheck | None:
        if self.additive_rows.matches(path):
            return rewritten_rows
        if self.lowered_limits.matches(path):
            return raised_limits
        return None

    def is_production(self, change: FileChange) -> bool:
        return self.change_tier(change) != "low"


@dataclass(frozen=True, slots=True)
class Factor:
    name: str
    tier: Tier
    reason: str


class FactorJson(TypedDict):
    name: ReadOnly[str]
    tier: ReadOnly[Tier]
    reason: ReadOnly[str]


class VerdictJson(TypedDict):
    tier: ReadOnly[Tier]
    factors: ReadOnly[tuple[FactorJson, ...]]
    summary: ReadOnly[str]


@dataclass(frozen=True, slots=True)
class Verdict:
    tier: Tier
    factors: tuple[Factor, ...]

    def summary_markdown(self) -> str:
        rows: Final = "\n".join(f"| {factor.name} | {factor.tier} | {factor.reason} |" for factor in self.factors)
        return f"risk: {self.tier} (shadow mode, nothing is blocked)\n\n| factor | tier | why |\n| --- | --- | --- |\n{rows}\n"

    def to_json(self) -> str:
        payload: Final[VerdictJson] = {
            "tier": self.tier,
            "factors": tuple(FactorJson(name=f.name, tier=f.tier, reason=f.reason) for f in self.factors),
            "summary": self.summary_markdown(),
        }
        return json.dumps(payload, indent=2)


def glob_to_regex(pattern: str) -> re.Pattern[str]:
    return re.compile("".join(_translate_glob_token(token) for token in GLOB_TOKEN_RE.split(pattern)))


def _translate_glob_token(token: str) -> str:
    match token:
        case "**/":
            return "(?:.*/)?"
        case "**":
            return ".*"
        case "*":
            return "[^/]*"
        case "?":
            return "[^/]"
        case _:
            return re.escape(token)


def load_config(path: Path) -> RiskConfig:
    return RiskConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


def parse_diff(diff_text: str) -> tuple[FileChange, ...]:
    blocks: Final = ("\n" + diff_text).split(DIFF_BLOCK_SEPARATOR)[1:]
    return tuple(_parse_block(block) for block in blocks)


def _unquote_git_path(quoted: str) -> str:
    def decode(match: re.Match[str]) -> str:
        return QUOTED_PATH_ESCAPES[match[1]] if match[1] else chr(int(match[2], 8))

    return QUOTED_PATH_ESCAPE_RE.sub(decode, quoted[1:-1])


def _unquote(path: str) -> str:
    return _unquote_git_path(path) if path.startswith('"') else path


def _header_path(header: str) -> str:
    one_side: Final = header[: (len(header) - 1) // 2]
    return _unquote(one_side).removeprefix("a/")


def _rename_side(extended_header: Sequence[str], pattern: re.Pattern[str]) -> str | None:
    return next((_unquote(match[1]) for line in extended_header if (match := pattern.match(line))), None)


def _parse_block(block: str) -> FileChange:
    header, _, body = block.partition("\n")
    lines: Final = body.split("\n")
    first_hunk: Final = next((index for index, line in enumerate(lines) if line.startswith("@@")), len(lines))
    extended_header: Final = lines[:first_hunk]
    hunk_lines: Final = lines[first_hunk:]
    renamed_to: Final = _rename_side(extended_header, RENAME_TO_RE)
    return FileChange(
        path=renamed_to if renamed_to is not None else _header_path(header),
        added_lines=tuple(line[1:] for line in hunk_lines if line.startswith("+")),
        deleted_lines=tuple(line[1:] for line in hunk_lines if line.startswith("-")),
        previous_path=_rename_side(extended_header, RENAME_FROM_RE) if renamed_to is not None else None,
    )


def module_key(path: str) -> str:
    parts: Final = path.split("/")
    if parts[0] != "litellm":
        return parts[0]
    depth: Final = 3 if len(parts) > 3 else 2
    return "/".join(parts[:depth])


def highest(tiers: Sequence[Tier]) -> Tier:
    if "high" in tiers:
        return "high"
    if "medium" in tiers:
        return "medium"
    return "low"


def _listed(paths: Sequence[str]) -> str:
    shown: Final = ", ".join(f"`{path}`" for path in paths[:MAX_LISTED_PATHS])
    rest: Final = len(paths) - MAX_LISTED_PATHS
    return f"{shown} and {rest} more" if rest > 0 else shown


def paths_factor(changes: Sequence[FileChange], rules: Rules) -> Factor:
    tier: Final = highest(tuple(rules.change_tier(change) for change in changes))
    matching: Final = tuple(change for change in changes if rules.change_tier(change) == tier)
    match tier:
        case "high":
            always_human: Final = tuple(
                path for change in matching for path in change.paths if rules.path_tier(path) == "high"
            )
            return Factor("paths", "high", f"always-human: {_listed(always_human)}")
        case "medium":
            rewritten: Final = tuple(
                f"{change.guarded_rewrite} in `{change.path}`" for change in matching if change.guarded_rewrite
            )
            outside: Final = len(matching) - len(rewritten)
            outside_note: Final = (
                (f"{outside} file(s) outside the docs, tests, and model map tiers",) if outside else ()
            )
            return Factor("paths", "medium", "; ".join((*rewritten, *outside_note)))
        case "low":
            return Factor("paths", "low", "docs, tests, cookbook, or model map only")


def modules_factor(changes: Sequence[FileChange], rules: Rules) -> Factor:
    modules: Final = tuple(
        sorted(
            frozenset(module_key(path) for change in changes for path in change.paths if rules.path_tier(path) != "low")
        )
    )
    count: Final = len(modules)
    tier: Final[Tier] = (
        "high"
        if count >= rules.config.modules.high_from
        else "medium"
        if count >= rules.config.modules.medium_from
        else "low"
    )
    return Factor("modules", tier, f"{count} module(s): {_listed(modules)}" if modules else "no production module")


def size_factor(changes: Sequence[FileChange], rules: Rules) -> Factor:
    counted: Final = tuple(
        change for change in changes if rules.is_production(change) and not rules.size_ignored.matches(change.path)
    )
    lines: Final = sum(change.line_count for change in counted)
    files: Final = len(counted)
    limits: Final = rules.config.size
    tier: Final[Tier] = (
        "low"
        if lines < limits.low.lines_under and files <= limits.low.files_up_to
        else "medium"
        if lines < limits.medium.lines_under and files <= limits.medium.files_up_to
        else "high"
    )
    return Factor("size", tier, f"{lines} line(s) across {files} file(s) outside the docs, tests, and model map tiers")


def _net(changes: Sequence[FileChange], pattern: re.Pattern[str]) -> int:
    added: Final = sum(1 for change in changes for line in change.added_lines if pattern.search(line))
    deleted: Final = sum(1 for change in changes for line in change.deleted_lines if pattern.search(line))
    return added - deleted


def tests_factor(changes: Sequence[FileChange], rules: Rules) -> Factor:
    test_changes: Final = tuple(change for change in changes if rules.test_files.matches(change.path))
    net_tests: Final = _net(test_changes, TEST_DEF_RE)
    net_skips: Final = _net(test_changes, SKIP_RE)
    silenced: Final = sum(
        max(_net((change,), SKIP_CALL_RE), 0) for change in test_changes if _net((change,), TEST_DEF_RE) <= 0
    )
    net_asserts: Final = _net(test_changes, ASSERT_RE)
    production_changed: Final = any(rules.is_production(change) for change in changes)
    if net_tests < 0:
        return Factor("tests", "high", f"{-net_tests} test(s) removed")
    if net_skips > 0:
        return Factor("tests", "high", f"{net_skips} skip marker(s) added")
    if silenced > 0:
        return Factor("tests", "high", f"{silenced} skip call(s) added to existing tests")
    if net_asserts < 0:
        return Factor("tests", "high", f"{-net_asserts} assertion(s) removed")
    if not production_changed:
        return Factor("tests", "low", "no production code changed")
    if net_tests > 0:
        return Factor("tests", "low", f"{net_tests} test(s) added")
    if test_changes:
        return Factor("tests", "medium", "tests edited, none added")
    return Factor("tests", "medium", "production code changed with no test touched")


def author_factor(author: str, from_fork: bool, rules: Rules) -> Factor:
    if from_fork:
        return Factor("author", "high", f"`{author}` from a fork")
    if author in rules.config.authors.low:
        return Factor("author", "low", f"`{author}` opened it on an internal branch")
    return Factor("author", "medium", f"`{author}` opened it by hand on an internal branch")


def classify(changes: Sequence[FileChange], author: str, from_fork: bool, rules: Rules) -> Verdict:
    factors: Final = (
        paths_factor(changes, rules),
        modules_factor(changes, rules),
        size_factor(changes, rules),
        tests_factor(changes, rules),
        author_factor(author, from_fork, rules),
    )
    return Verdict(highest(tuple(factor.tier for factor in factors)), factors)


def _json_object(text: str | None) -> Mapping[str, object] | None:
    if text is None:
        return None
    try:
        return MappingProxyType(JSON_OBJECT.validate_json(text))
    except ValidationError:
        return None


def _as_object(value: object) -> Mapping[str, object] | None:
    try:
        return MappingProxyType(JSON_OBJECT.validate_python(value))
    except ValidationError:
        return None


def rewritten_rows(base_text: str | None, head_text: str | None) -> str | None:
    if base_text is None:
        return None
    base: Final = _json_object(base_text)
    head: Final = _json_object(head_text)
    if base is None or head is None:
        return "not a JSON object on both sides"
    rewritten: Final = sum(1 for key, value in base.items() if key not in head or head[key] != value)
    return f"{rewritten} existing row(s) changed or removed" if rewritten else None


def _limits(value: object, prefix: str = "") -> tuple[tuple[str, float], ...]:
    node: Final = _as_object(value)
    if node is None:
        return ()
    limit: Final = node.get("limit")
    own: Final = ((prefix, float(limit)),) if isinstance(limit, int | float) and not isinstance(limit, bool) else ()
    nested: Final = tuple(
        pair for key, child in node.items() if key != "limit" for pair in _limits(child, f"{prefix}/{key}")
    )
    return (*own, *nested)


def raised_limits(base_text: str | None, head_text: str | None) -> str | None:
    if base_text is None:
        return None
    base: Final = _json_object(base_text)
    head: Final = _json_object(head_text)
    if base is None or head is None:
        return "not a JSON object on both sides"
    base_limits: Final = MappingProxyType(dict(_limits(base)))
    raised: Final = sum(1 for key, limit in _limits(head) if key not in base_limits or limit > base_limits[key])
    return f"{raised} limit(s) raised" if raised else None


def with_guards(
    changes: Sequence[FileChange], rules: Rules, base: str, head: str, read_file: ReadFile
) -> tuple[FileChange, ...]:
    return tuple(_guarded(change, rules, base, head, read_file) for change in changes)


def _guarded(change: FileChange, rules: Rules, base: str, head: str, read_file: ReadFile) -> FileChange:
    check: Final = rules.rewrite_check(change.path)
    if check is None:
        return change
    reason: Final = check(read_file(base, change.previous_path or change.path), read_file(head, change.path))
    return change if reason is None else replace(change, guarded_rewrite=reason)


def git_show(repo: Path, rev: str, path: str) -> str | None:
    completed: Final = subprocess.run(
        ("git", "show", f"{rev}:{path}"),
        cwd=repo,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    return completed.stdout if completed.returncode == 0 else None


def git_diff(repo: Path, base: str, head: str) -> str:
    completed: Final = subprocess.run(
        ("git", "-c", "core.quotePath=false", "diff", "--find-renames", "--no-ext-diff", "-U0", base, head),
        cwd=repo,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        check=True,
    )
    return completed.stdout


def build_parser() -> argparse.ArgumentParser:
    parser: Final = argparse.ArgumentParser(description="Compute a pull request's floor risk tier from its diff")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--base", required=True)
    parser.add_argument("--head", required=True)
    parser.add_argument("--author", required=True)
    parser.add_argument("--from-fork", action="store_true")
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--json-out", type=Path)
    return parser


class CliArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    config: Path
    base: str
    head: str
    author: str
    from_fork: bool
    repo: Path
    json_out: Path | None


def main(argv: Sequence[str] | None = None) -> int:
    args: Final = CliArgs.model_validate(vars(build_parser().parse_args(argv)))
    rules: Final = Rules.from_config(load_config(args.config))
    changes: Final = with_guards(
        parse_diff(git_diff(args.repo, args.base, args.head)),
        rules,
        args.base,
        args.head,
        lambda rev, path: git_show(args.repo, rev, path),
    )
    verdict: Final = classify(changes, args.author, args.from_fork, rules)
    if args.json_out is not None:
        args.json_out.write_text(verdict.to_json(), encoding="utf-8")
    sys.stdout.write(verdict.summary_markdown())
    return 0


if __name__ == "__main__":
    sys.exit(main())
