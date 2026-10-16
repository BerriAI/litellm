import json
import re
import subprocess
from collections.abc import Callable
from typing import Final

import pytest

from release_prs import main, report, version_tags


def test_semantic_versions_ignore_random_tags_and_sort_numeric_prereleases() -> None:
    names: Final = (
        "v2.0.0-dev.10",
        "v2.0.0-dev.2",
        "v2.0.0-rc.1",
        "v2.0.0",
        "v1.10.0",
        "1.9.0",
        "v1.10.0+build.2",
        "nightly",
        "v2.0",
        "v02.0.0",
        "v2.0.0-dev.01",
        "v2.0.0-",
        "v2.0.0+",
    )
    assert version_tags(names) == (
        "1.9.0",
        "v1.10.0",
        "v1.10.0+build.2",
        "v2.0.0-dev.2",
        "v2.0.0-dev.10",
        "v2.0.0-rc.1",
        "v2.0.0",
    )
    assert version_tags(names, stable=True) == ("1.9.0", "v1.10.0", "v1.10.0+build.2", "v2.0.0")


def test_report_filters_authors_and_actual_merge_commits_and_deduplicates() -> None:
    def github(*args: str) -> str:
        return json.dumps(
            (
                dict(number=12, title="Second fix", url="https://example.test/pull/12", author="bob", merge_commit="b"),
                dict(
                    number=11, title="First fix", url="https://example.test/pull/11", author="Alice", merge_commit="a"
                ),
                dict(
                    number=11, title="First fix", url="https://example.test/pull/11", author="Alice", merge_commit="a"
                ),
                dict(
                    number=13,
                    title="Other author",
                    url="https://example.test/pull/13",
                    author="carol",
                    merge_commit="c",
                ),
                dict(
                    number=14,
                    title="Merged later",
                    url="https://example.test/pull/14",
                    author="Alice",
                    merge_commit="d",
                ),
            )
        )

    releases: Final = (("v1.0.0", ("a", "b", "c")), ("v1.0.1", ()))
    assert report("owner/repo", releases, ("@alice", "BOB"), github) == (
        "v1.0.1\n  No matching PRs\n\n"
        "v1.0.0\n  #12 Second fix (@bob)\n    https://example.test/pull/12\n"
        "  #11 First fix (@Alice)\n    https://example.test/pull/11"
    )


def release_commands() -> Callable[..., str]:
    def run(*args: str) -> str:
        if args[:2] == ("gh", "repo"):
            return "owner/repo"
        if args[:2] == ("gh", "api"):
            return json.dumps(
                tuple(
                    dict(
                        number=number,
                        title=f"Fix {number}",
                        url=f"https://example.test/pull/{number}",
                        author="alice",
                        merge_commit=sha,
                    )
                    for number, sha in ((1, "old"), (2, "shared"), (3, "branch"), (4, "new"))
                )
            )
        if args[:2] == ("git", "rev-parse"):
            return "false"
        if args[:2] == ("git", "remote"):
            return "https://github.com/owner/repo.git"
        if args[:2] == ("git", "ls-remote"):
            return "\n".join(f"sha\trefs/tags/{tag}" for tag in ("v1.0.0", "v1.0.1", "v1.1.0", "junk", "v1.2.0-rc.1"))
        if args[:2] == ("git", "fetch"):
            return ""
        histories: Final = {
            "refs/tags/v1.0.0": frozenset(("old",)),
            "refs/tags/v1.0.1": frozenset(("old", "shared")),
            "refs/tags/v1.1.0": frozenset(("old", "branch")),
            "refs/tags/v1.2.0-rc.1": frozenset(("old", "shared", "new")),
        }
        excluded: Final = frozenset().union(*(histories[tag] for tag in args[4:-1]))
        return "\n".join(sorted(histories[args[2]] - excluded))

    return run


@pytest.mark.parametrize("stable", [False, True])
def test_cli_limits_releases_and_uses_all_older_tags_across_divergent_branches(
    capsys: pytest.CaptureFixture[str], stable: bool
) -> None:
    args: Final = ("alice", "--count", "2") + (("--stable",) if stable else ())
    assert main(args, run=release_commands()) == 0
    captured: Final = capsys.readouterr()
    assert captured.err == ""
    assert captured.out == (
        "v1.1.0\n  #3 Fix 3 (@alice)\n    https://example.test/pull/3\n\n"
        "v1.0.1\n  #2 Fix 2 (@alice)\n    https://example.test/pull/2\n"
        if stable
        else "v1.2.0-rc.1\n  #4 Fix 4 (@alice)\n    https://example.test/pull/4\n\n"
        "v1.1.0\n  #3 Fix 3 (@alice)\n    https://example.test/pull/3\n"
    )


def test_cli_rejects_nonpositive_count(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(("alice", "--count", "0"), run=release_commands()) == 1
    captured: Final = capsys.readouterr()
    assert captured.out == ""
    assert "greater than 0" in captured.err


def test_report_includes_prs_after_the_first_api_batch() -> None:
    def github(*args: str) -> str:
        query: Final = args[4]
        shas: Final = re.findall(r'object\(expression: "([^"]+)"\)', query)
        return json.dumps(
            tuple(
                dict(
                    number=int(sha),
                    title=f"Fix {sha}",
                    url=f"https://example.test/pull/{sha}",
                    author="alice",
                    merge_commit=sha,
                )
                for sha in shas
            )
        )

    commits: Final = tuple(str(number) for number in range(1, 36))
    expected: Final = "v1.0.0\n" + "\n".join(
        f"  #{number} Fix {number} (@alice)\n    https://example.test/pull/{number}" for number in range(35, 0, -1)
    )
    assert report("owner/repo", (("v1.0.0", commits),), ("alice",), github) == expected


def test_cli_reports_github_failure_without_printing_partial_results(capsys: pytest.CaptureFixture[str]) -> None:
    def unavailable(*args: str) -> str:
        if args[:2] == ("gh", "api"):
            raise subprocess.CalledProcessError(1, args, stderr="GitHub authentication required")
        return release_commands()(*args)

    assert main(("alice",), run=unavailable) == 1
    captured: Final = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "GitHub authentication required\n"
