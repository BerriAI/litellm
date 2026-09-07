from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Final, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

Tier = Literal["low", "medium", "high"]

TEST_DEF_RE: Final = re.compile(r"^\s*(?:async\s+)?def\s+test_|^\s*(?:it|test)\(")
SKIP_RE: Final = re.compile(
    r"pytest\.mark\.(?:skip(?!if)|xfail)|unittest\.skip\b|\b(?:it|test|describe)\.skip\(\s*[\"'`]|\bx(?:it|test|describe)\("
)
ASSERT_RE: Final = re.compile(r"^\s*assert\b|\bexpect\(")
GLOB_TOKEN_RE: Final = re.compile(r"(\*\*/|\*\*|\*|\?)")
DIFF_BLOCK_SEPARATOR: Final = "\ndiff --git "
QUOTED_PATH_ESCAPE_RE: Final = re.compile(r'\\(?:([abfnrtv"\\])|([0-7]{3}))')
QUOTED_PATH_ESCAPES: Final = MappingProxyType(
    {"a": "\a", "b": "\b", "f": "\f", "n": "\n", "r": "\r", "t": "\t", "v": "\v", '"': '"', "\\": "\\"}
)
MAX_LISTED_PATHS: Final = 5


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
    low: tuple[str, ...]


class TestRules(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    files: tuple[str, ...]


class RiskConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    paths: PathRules
    modules: ModuleRules
    size: SizeRules
    tests: TestRules


@dataclass(frozen=True, slots=True)
class PathMatcher:
    patterns: tuple[re.Pattern[str], ...]

    @staticmethod
    def from_globs(globs: Sequence[str]) -> PathMatcher:
        return PathMatcher(tuple(glob_to_regex(glob) for glob in globs))

    def matches(self, path: str) -> bool:
        return any(pattern.fullmatch(path) for pattern in self.patterns)


@dataclass(frozen=True, slots=True)
class Rules:
    config: RiskConfig
    high_paths: PathMatcher
    low_paths: PathMatcher
    test_files: PathMatcher
    size_ignored: PathMatcher

    @staticmethod
    def from_config(config: RiskConfig) -> Rules:
        return Rules(
            config=config,
            high_paths=PathMatcher.from_globs(config.paths.high),
            low_paths=PathMatcher.from_globs(config.paths.low),
            test_files=PathMatcher.from_globs(config.tests.files),
            size_ignored=PathMatcher.from_globs(config.size.ignore),
        )

    def path_tier(self, path: str) -> Tier:
        if self.high_paths.matches(path):
            return "high"
        if self.test_files.matches(path) or self.low_paths.matches(path):
            return "low"
        return "medium"

    def is_production(self, path: str) -> bool:
        return self.path_tier(path) != "low"


@dataclass(frozen=True, slots=True)
class FileChange:
    path: str
    added_lines: tuple[str, ...]
    deleted_lines: tuple[str, ...]

    @property
    def line_count(self) -> int:
        return len(self.added_lines) + len(self.deleted_lines)


@dataclass(frozen=True, slots=True)
class Factor:
    name: str
    tier: Tier
    reason: str


@dataclass(frozen=True, slots=True)
class Verdict:
    tier: Tier
    factors: tuple[Factor, ...]

    def summary_markdown(self) -> str:
        rows = "\n".join(f"| {factor.name} | {factor.tier} | {factor.reason} |" for factor in self.factors)
        return f"risk: {self.tier} (shadow mode, nothing is blocked)\n\n| factor | tier | why |\n| --- | --- | --- |\n{rows}\n"

    def to_json(self) -> str:
        payload = {
            "tier": self.tier,
            "factors": [{"name": f.name, "tier": f.tier, "reason": f.reason} for f in self.factors],
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
    blocks = ("\n" + diff_text).split(DIFF_BLOCK_SEPARATOR)[1:]
    return tuple(_parse_block(block) for block in blocks)


def _unquote_git_path(quoted: str) -> str:
    def decode(match: re.Match[str]) -> str:
        return QUOTED_PATH_ESCAPES[match[1]] if match[1] else chr(int(match[2], 8))

    return QUOTED_PATH_ESCAPE_RE.sub(decode, quoted[1:-1])


def _header_path(header: str) -> str:
    one_side = header[: (len(header) - 1) // 2]
    unquoted = _unquote_git_path(one_side) if one_side.startswith('"') else one_side
    return unquoted.removeprefix("a/")


def _parse_block(block: str) -> FileChange:
    header, _, body = block.partition("\n")
    path = _header_path(header)
    lines = body.split("\n")
    first_hunk = next((index for index, line in enumerate(lines) if line.startswith("@@")), len(lines))
    hunk_lines = lines[first_hunk:]
    return FileChange(
        path=path,
        added_lines=tuple(line[1:] for line in hunk_lines if line.startswith("+")),
        deleted_lines=tuple(line[1:] for line in hunk_lines if line.startswith("-")),
    )


def module_key(path: str) -> str:
    parts = path.split("/")
    if parts[0] != "litellm":
        return parts[0]
    depth = 3 if len(parts) > 3 else 2
    return "/".join(parts[:depth])


def highest(tiers: Sequence[Tier]) -> Tier:
    if "high" in tiers:
        return "high"
    if "medium" in tiers:
        return "medium"
    return "low"


def _listed(paths: Sequence[str]) -> str:
    shown = ", ".join(f"`{path}`" for path in paths[:MAX_LISTED_PATHS])
    rest = len(paths) - MAX_LISTED_PATHS
    return f"{shown} and {rest} more" if rest > 0 else shown


def paths_factor(changes: Sequence[FileChange], rules: Rules) -> Factor:
    tier = highest([rules.path_tier(change.path) for change in changes])
    matching = [change.path for change in changes if rules.path_tier(change.path) == tier]
    match tier:
        case "high":
            return Factor("paths", "high", f"always-human: {_listed(matching)}")
        case "medium":
            return Factor("paths", "medium", f"{len(matching)} file(s) outside the docs, tests, and model map tiers")
        case "low":
            return Factor("paths", "low", "docs, tests, cookbook, or model map only")


def modules_factor(changes: Sequence[FileChange], rules: Rules) -> Factor:
    modules = sorted({module_key(change.path) for change in changes if rules.is_production(change.path)})
    count = len(modules)
    tier: Tier = (
        "high"
        if count >= rules.config.modules.high_from
        else "medium"
        if count >= rules.config.modules.medium_from
        else "low"
    )
    return Factor("modules", tier, f"{count} module(s): {_listed(modules)}" if modules else "no production module")


def size_factor(changes: Sequence[FileChange], rules: Rules) -> Factor:
    counted = [change for change in changes if not rules.size_ignored.matches(change.path)]
    lines = sum(change.line_count for change in counted)
    files = len(counted)
    limits = rules.config.size
    tier: Tier = (
        "low"
        if lines < limits.low.lines_under and files <= limits.low.files_up_to
        else "medium"
        if lines < limits.medium.lines_under and files <= limits.medium.files_up_to
        else "high"
    )
    return Factor("size", tier, f"{lines} line(s) across {files} file(s)")


def _net(changes: Sequence[FileChange], pattern: re.Pattern[str]) -> int:
    added = sum(1 for change in changes for line in change.added_lines if pattern.search(line))
    deleted = sum(1 for change in changes for line in change.deleted_lines if pattern.search(line))
    return added - deleted


def tests_factor(changes: Sequence[FileChange], rules: Rules) -> Factor:
    test_changes = [change for change in changes if rules.test_files.matches(change.path)]
    net_tests = _net(test_changes, TEST_DEF_RE)
    net_skips = _net(test_changes, SKIP_RE)
    net_asserts = _net(test_changes, ASSERT_RE)
    production_changed = any(rules.is_production(change.path) for change in changes)
    if net_tests < 0:
        return Factor("tests", "high", f"{-net_tests} test(s) removed")
    if net_skips > 0:
        return Factor("tests", "high", f"{net_skips} skip marker(s) added")
    if net_asserts < 0:
        return Factor("tests", "high", f"{-net_asserts} assertion(s) removed")
    if not production_changed:
        return Factor("tests", "low", "no production code changed")
    if net_tests > 0:
        return Factor("tests", "low", f"{net_tests} test(s) added")
    if test_changes:
        return Factor("tests", "medium", "tests edited, none added")
    return Factor("tests", "medium", "production code changed with no test touched")


def author_factor(author: str, from_fork: bool) -> Factor:
    if from_fork:
        return Factor("author", "high", f"`{author}` from a fork")
    return Factor("author", "low", f"`{author}` on an internal branch")


def classify(changes: Sequence[FileChange], author: str, from_fork: bool, rules: Rules) -> Verdict:
    factors = (
        paths_factor(changes, rules),
        modules_factor(changes, rules),
        size_factor(changes, rules),
        tests_factor(changes, rules),
        author_factor(author, from_fork),
    )
    return Verdict(highest([factor.tier for factor in factors]), factors)


def git_diff(repo: Path, base: str, head: str) -> str:
    completed = subprocess.run(
        ["git", "-c", "core.quotePath=false", "diff", "--no-renames", "--no-ext-diff", "-U0", base, head],
        cwd=repo,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        check=True,
    )
    return completed.stdout


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute a pull request's floor risk tier from its diff")
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
    args = CliArgs.model_validate(vars(build_parser().parse_args(argv)))
    rules = Rules.from_config(load_config(args.config))
    changes = parse_diff(git_diff(args.repo, args.base, args.head))
    verdict = classify(changes, args.author, args.from_fork, rules)
    if args.json_out is not None:
        args.json_out.write_text(verdict.to_json(), encoding="utf-8")
    sys.stdout.write(verdict.summary_markdown())
    return 0


if __name__ == "__main__":
    sys.exit(main())
