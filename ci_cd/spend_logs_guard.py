"""Required check: a pull request that touches LiteLLM_SpendLogs in schema.prisma or a migration
needs an approving review from the spend-log owner before it can merge.

CODEOWNERS can only match file paths, and every schema change edits the same schema.prisma copies,
so a path rule would put every migration on the owner. This gate reads the diff instead: when no
added line in the watched files mentions the table it passes at once, otherwise it passes only once
the owner's latest review on the pull request is an approval. The workflow re-runs on every review.
"""

from __future__ import annotations

import argparse
import re
import subprocess
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


class Reviewer(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")
    login: str


class Review(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")
    state: str
    user: Reviewer


_REVIEWS: Final = TypeAdapter(list[Review])


def watched(path: str) -> bool:
    return path in WATCHED or (path.startswith(MIGRATIONS_PREFIX) and path.endswith("/migration.sql"))


def _added_lines(section: str) -> tuple[str, ...]:
    body: Final = section.split("\n@@", 1)
    return tuple(line[1:] for line in body[1].splitlines() if line.startswith("+")) if len(body) == 2 else ()


def spend_log_hits(diff: str) -> tuple[str, ...]:
    """`path: added line` for every added line in a watched file that names the table."""
    sections: Final = re.split(r"^diff --git [^\n]*\n", diff, flags=re.MULTILINE)
    return tuple(f"{path}: {line.strip()}" for path, line in _watched_added_lines(sections) if TABLE.search(line))


def _watched_added_lines(sections: Sequence[str]) -> tuple[tuple[str, str], ...]:
    pairs: Final = tuple((_path_of(section), section) for section in sections)
    return tuple(
        (path, line) for path, section in pairs if watched(path) for line in _added_lines(section)
    )  # comprehension-ok: path then its lines


def _path_of(section: str) -> str:
    match: Final = re.search(r"^\+\+\+ b/(.+)$", section, flags=re.MULTILINE)
    return match.group(1) if match else ""


def approved_by(reviews: Sequence[Review], login: str) -> bool:
    """True when `login`'s most recent review on the pull request is an approval."""
    states: Final = tuple(review.state for review in reviews if review.user.login == login)
    return bool(states) and states[-1] == "APPROVED"


def added_diff(base: str, head: str) -> str:
    result: Final = subprocess.run(
        ("git", "diff", "--unified=0", "--no-color", f"{base}..{head}", "--", *WATCHED, MIGRATIONS_PREFIX),
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


class Arguments(BaseModel):
    model_config = ConfigDict(frozen=True)
    base: str
    head: str
    approver: str
    reviews: Path


def main(argv: Sequence[str]) -> int:
    parser: Final = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, help="merge base of the pull request")
    parser.add_argument("--head", required=True, help="head sha of the pull request")
    parser.add_argument("--approver", required=True, help="GitHub login whose approval unlocks the merge")
    parser.add_argument("--reviews", required=True, help="JSON file with the pull request's reviews")
    args: Final = Arguments.model_validate(vars(parser.parse_args(argv)))
    hits: Final = spend_log_hits(added_diff(args.base, args.head))
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
