from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections.abc import Callable, Iterator
from types import MappingProxyType
from typing import Final, TypeAlias

from pydantic import BaseModel, ConfigDict, PositiveInt, TypeAdapter, ValidationError

Command: TypeAlias = Callable[..., str]
VersionKey: TypeAlias = tuple[int, int, int, bool, tuple[tuple[int, int, str], ...]]
ReleaseCommits: TypeAlias = tuple[str, tuple[str, ...]]

SEMVER: Final = re.compile(
    r"v?(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)"
    r"(?:-((?:0|[1-9][0-9]*|[0-9]*[A-Za-z-][0-9A-Za-z-]*)"
    r"(?:\.(?:0|[1-9][0-9]*|[0-9]*[A-Za-z-][0-9A-Za-z-]*))*))?"
    r"(?:\+[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?"
)
BATCH_SIZE: Final = 30


class Options(BaseModel):
    model_config = ConfigDict(frozen=True)

    authors: tuple[str, ...]
    count: PositiveInt = 5
    stable: bool = False


class PullRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    number: int
    title: str
    url: str
    author: str
    merge_commit: str


PRS: Final = TypeAdapter(tuple[PullRequest, ...])


def command(*args: str) -> str:
    result: Final = subprocess.run(args, check=True, capture_output=True, text=True, timeout=300)
    return result.stdout.strip()


def version_key(tag: str) -> VersionKey | None:
    match: Final = SEMVER.fullmatch(tag)
    if match is None:
        return None
    major, minor, patch, prerelease = match.groups()
    identifiers: Final = tuple(
        (0, int(part), "") if part.isdigit() else (1, 0, part) for part in (prerelease or "").split(".")
    )
    return int(major), int(minor), int(patch), prerelease is None, identifiers


def version_tags(names: tuple[str, ...], stable: bool = False) -> tuple[str, ...]:
    versions: Final = tuple((name, key) for name in names if (key := version_key(name)) is not None)
    selected: Final = tuple(item for item in versions if not stable or item[1][3])
    return tuple(name for name, _ in sorted(selected, key=lambda item: (item[1], item[0])))


def release_commits(tags: tuple[str, ...], count: int, run: Command) -> tuple[ReleaseCommits, ...]:
    start: Final = max(0, len(tags) - count)
    return tuple(
        (
            tag,
            tuple(
                run(
                    "git",
                    "rev-list",
                    f"refs/tags/{tag}",
                    "--not",
                    *(f"refs/tags/{older}" for older in tags[:index]),
                    "--",
                ).splitlines()
            ),
        )
        for index, tag in enumerate(tags)
        if index >= start
    )


def batch_prs(repo: str, commits: tuple[str, ...], run: Command) -> tuple[PullRequest, ...]:
    owner, name = repo.split("/")
    fields: Final = " ".join(
        f"c{index}: object(expression: {json.dumps(sha)}) {{ ... on Commit {{"
        " associatedPullRequests(first: 100) { pageInfo { hasNextPage }"
        " nodes { number title url state author { login } mergeCommit { oid } repository { nameWithOwner } }"
        " } } }"
        for index, sha in enumerate(commits)
    )
    query: Final = f"query {{ repository(owner: {json.dumps(owner)}, name: {json.dumps(name)}) {{ {fields} }} }}"
    jq: Final = (
        "[.data.repository[] | .associatedPullRequests |"
        ' if .pageInfo.hasNextPage then error("Too many PRs associated with a commit") else .nodes[] end |'
        f' select(.state == "MERGED" and .repository.nameWithOwner == {json.dumps(repo)}) |'
        ' {number, title, url, author: (.author.login // ""), merge_commit: (.mergeCommit.oid // "")}]'
    )
    return PRS.validate_json(run("gh", "api", "graphql", "-f", f"query={query}", "--jq", jq))


def commit_prs(repo: str, commits: tuple[str, ...], run: Command) -> Iterator[PullRequest]:
    for offset in range(0, len(commits), BATCH_SIZE):
        yield from batch_prs(repo, commits[offset : offset + BATCH_SIZE], run)


def report(repo: str, releases: tuple[ReleaseCommits, ...], authors: tuple[str, ...], run: Command) -> str:
    selected_authors: Final = frozenset(author.removeprefix("@").casefold() for author in authors)

    def section(tag: str, commits: tuple[str, ...]) -> str:
        included: Final = frozenset(commits)
        matching: Final = MappingProxyType(
            {
                pr.number: pr
                for pr in commit_prs(repo, commits, run)
                if pr.author.casefold() in selected_authors and pr.merge_commit in included
            }
        )
        lines: Final = tuple(
            f"  #{pr.number} {pr.title} (@{pr.author})\n    {pr.url}"
            for pr in sorted(matching.values(), key=lambda pr: pr.number, reverse=True)
        )
        return f"{tag}\n" + ("\n".join(lines) if lines else "  No matching PRs")

    return "\n\n".join(section(tag, commits) for tag, commits in reversed(releases))


def main(argv: tuple[str, ...] | None = None, run: Command = command) -> int:
    parser: Final = argparse.ArgumentParser(
        description="Show authors' PRs first included in the latest semantic version tags of this checkout's origin",
        epilog="Requires git, an authenticated gh CLI, and a full clone. Prereleases are included unless --stable is set",
    )
    parser.add_argument("authors", nargs="+", help="GitHub usernames, separated by spaces")
    parser.add_argument("--count", type=int, default=5, help="Number of version tags to show (default 5)")
    parser.add_argument("--stable", action="store_true", help="Only show versions without a prerelease suffix")
    try:
        options: Final = Options.model_validate(vars(parser.parse_args(argv)))
        if run("git", "rev-parse", "--is-shallow-repository") == "true":
            sys.stderr.write("A full clone is required. Run git fetch --unshallow origin first\n")
            return 1
        origin: Final = run("git", "remote", "get-url", "origin")
        repo: Final = run("gh", "repo", "view", origin, "--json", "nameWithOwner", "--jq", ".nameWithOwner")
        remote_tags: Final = run("git", "ls-remote", "--tags", "--refs", "origin")
        tags: Final = version_tags(
            tuple(line.split("\t", 1)[1].removeprefix("refs/tags/") for line in remote_tags.splitlines()),
            options.stable,
        )
        if not tags:
            sys.stderr.write("No semantic version tags found on origin\n")
            return 1
        run("git", "fetch", "--no-tags", "origin", "refs/tags/*:refs/tags/*")
        releases: Final = release_commits(tags, options.count, run)
        sys.stdout.write(report(repo, releases, options.authors, run) + "\n")
        return 0
    except subprocess.CalledProcessError as exc:
        stderr: Final = TypeAdapter(str).validate_python(getattr(exc, "stderr", ""))
        sys.stderr.write((stderr.strip() or str(exc)) + "\n")
        return 1
    except (OSError, subprocess.TimeoutExpired, ValidationError) as exc:
        sys.stderr.write(str(exc) + "\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
