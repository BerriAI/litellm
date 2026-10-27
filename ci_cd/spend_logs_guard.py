"""Required check: a pull request that touches LiteLLM_SpendLogs in schema.prisma or a migration
needs an approving review from the spend-log owner before it can merge.

CODEOWNERS can only match file paths, and every schema change edits the same schema.prisma copies,
so a path rule would put every migration on the owner. This gate reads the pull request's diff from
the GitHub API instead (`pulls/<n>/files`): when no added line in the watched files mentions the table
it passes at once, otherwise it passes only once the owner's latest review on the pull request is an
approval. The workflow re-runs on every review. The pull request's code is never fetched or run.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Final

from pydantic import BaseModel, ConfigDict, TypeAdapter

WATCHED: Final = (
    "schema.prisma",
    "litellm/proxy/schema.prisma",
    "litellm-proxy-extras/litellm_proxy_extras/schema.prisma",
)
MIGRATIONS_PREFIX: Final = "litellm-proxy-extras/litellm_proxy_extras/migrations/"
TABLE: Final = re.compile(r"(?<![A-Za-z0-9])LiteLLM_SpendLogs", re.IGNORECASE)
# https://docs.github.com/en/rest/pulls/pulls#list-pull-requests-files (2026-10-26): at most 3000 files are listed
FILES_API_CAP: Final = 3000


class Reviewer(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")
    login: str


class Review(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")
    state: str
    user: Reviewer


class ChangedFile(BaseModel):
    """One entry of `GET /repos/{owner}/{repo}/pulls/{n}/files`. `patch` is omitted for binary or oversized diffs."""

    model_config = ConfigDict(frozen=True, extra="ignore")
    filename: str
    changes: int
    patch: str | None = None


_REVIEWS: Final = TypeAdapter(list[Review])
_FILES: Final = TypeAdapter(list[ChangedFile])


def watched(path: str) -> bool:
    return path in WATCHED or (path.startswith(MIGRATIONS_PREFIX) and path.endswith("/migration.sql"))


def _added_lines(patch: str) -> tuple[str, ...]:
    return tuple(line[1:] for line in patch.splitlines() if line.startswith("+"))


def spend_log_hits(files: Sequence[ChangedFile]) -> tuple[str, ...]:
    """`path: added line` for every added line in a watched file that names the table."""
    return tuple(f"{path}: {line.strip()}" for path, line in _watched_added_lines(files) if TABLE.search(line))


def _watched_added_lines(files: Sequence[ChangedFile]) -> tuple[tuple[str, str], ...]:
    return tuple(
        (file.filename, line) for file in files if watched(file.filename) for line in _added_lines(file.patch or "")
    )  # comprehension-ok: path then its lines


def unreadable(files: Sequence[ChangedFile]) -> tuple[str, ...]:
    """Watched files whose diff the API withheld, plus a marker when the file list itself was cut off."""
    withheld: Final = tuple(f.filename for f in files if watched(f.filename) and f.changes and f.patch is None)
    return withheld + (("(file list truncated by the API)",) if len(files) >= FILES_API_CAP else ())


def approved_by(reviews: Sequence[Review], login: str) -> bool:
    """True when `login`'s most recent review on the pull request is an approval."""
    states: Final = tuple(review.state for review in reviews if review.user.login == login)
    return bool(states) and states[-1] == "APPROVED"


class Arguments(BaseModel):
    model_config = ConfigDict(frozen=True)
    files: Path
    reviews: Path
    approver: str


def main(argv: Sequence[str]) -> int:
    parser: Final = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--files", required=True, help="JSON file with the pull request's `pulls/<n>/files` entries")
    parser.add_argument("--reviews", required=True, help="JSON file with the pull request's reviews")
    parser.add_argument("--approver", required=True, help="GitHub login whose approval unlocks the merge")
    args: Final = Arguments.model_validate(vars(parser.parse_args(argv)))
    files: Final = _FILES.validate_json(args.files.read_bytes())
    hidden: Final = unreadable(files)
    if hidden:
        names: Final = "".join(f"  {name}\n" for name in hidden)
        sys.stdout.write(
            "The GitHub API did not return the full diff for a watched file, so LiteLLM_SpendLogs cannot be checked. "
            f"Split the pull request:\n{names}"
        )
        return 1
    hits: Final = spend_log_hits(files)
    if not hits:
        sys.stdout.write("No added line in schema.prisma or a migration names LiteLLM_SpendLogs; nothing to review.\n")
        return 0
    if approved_by(_REVIEWS.validate_json(args.reviews.read_bytes()), args.approver):
        sys.stdout.write(f"LiteLLM_SpendLogs change approved by {args.approver}: {', '.join(hits)}\n")
        return 0
    listed: Final = "".join(f"  {hit}\n" for hit in hits)
    sys.stdout.write(
        f"This pull request changes LiteLLM_SpendLogs and needs an approving review from @{args.approver}:\n"
        f"{listed}The check re-runs when a review is submitted.\n"
    )
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
