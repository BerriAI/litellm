"""Required check: a pull request that touches LiteLLM_SpendLogs in schema.prisma or a migration
needs an approving review from the spend-log owner before it can merge.

CODEOWNERS can only match file paths, and every schema change edits the same schema.prisma copies,
so a path rule would put every migration on the owner. This gate reads the diff instead: a schema.prisma
hunk counts when it adds or removes a line inside the `model LiteLLM_SpendLogs { ... }` block, a migration
hunk counts when an added line names the table. Without such a hunk it passes at once, otherwise it passes
only once the owner's latest review on the pull request is an approval that still covers every spend-log line
in the head (approvals survive pushes, so a later commit may not add spend-log lines the owner never saw).
Reviews do not trigger the workflow, so the check is re-run by hand after approving.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Final

from pydantic import BaseModel, ConfigDict, TypeAdapter

WATCHED: Final = (
    "schema.prisma",
    "litellm/proxy/schema.prisma",
    "litellm-proxy-extras/litellm_proxy_extras/schema.prisma",
)
MIGRATIONS_PREFIX: Final = "litellm-proxy-extras/litellm_proxy_extras/migrations/"
TABLE: Final = re.compile(r"(?<![A-Za-z0-9])LiteLLM_SpendLogs(?![A-Za-z0-9])", re.IGNORECASE)
DECIDING_STATES: Final = frozenset({"APPROVED", "CHANGES_REQUESTED", "DISMISSED"})
MODEL_START: Final = re.compile(r"^model\s+LiteLLM_SpendLogs\s*\{")
HUNK: Final = re.compile(r"^@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@[^\n]*\n", re.MULTILINE)


class Reviewer(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")
    login: str


class Review(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")
    state: str
    user: Reviewer
    commit_id: str | None = None


_REVIEWS: Final = TypeAdapter(list[Review])


@dataclass(frozen=True, slots=True)
class Block:
    """Lines of `model LiteLLM_SpendLogs { ... }` in one schema.prisma copy at the merge base and at the head."""

    base: range
    head: range


@dataclass(frozen=True, slots=True)
class Hunk:
    removed: tuple[tuple[int, str], ...]
    added: tuple[tuple[int, str], ...]


def watched(path: str) -> bool:
    return path in WATCHED or (path.startswith(MIGRATIONS_PREFIX) and path.endswith("/migration.sql"))


def model_block(schema: str) -> range:
    """1-based line range of `model LiteLLM_SpendLogs { ... }` including both braces, empty when the schema has none."""
    lines: Final = schema.splitlines()
    starts: Final = tuple(number for number, line in enumerate(lines, 1) if MODEL_START.match(line))
    if not starts:
        return range(0)
    ends: Final = tuple(number for number, line in enumerate(lines, 1) if number > starts[0] and line.startswith("}"))
    return range(starts[0], (ends[0] if ends else len(lines)) + 1)


def _hunks(section: str) -> tuple[Hunk, ...]:
    headers: Final = tuple(HUNK.finditer(section))
    ends: Final = tuple(header.start() for header in headers[1:]) + (len(section),)
    return tuple(_hunk(header, section[header.end() : end]) for header, end in zip(headers, ends))


def _hunk(header: re.Match[str], body: str) -> Hunk:
    lines: Final = body.splitlines()
    return Hunk(
        removed=tuple(enumerate((line[1:] for line in lines if line.startswith("-")), int(header.group(1)))),
        added=tuple(enumerate((line[1:] for line in lines if line.startswith("+")), int(header.group(2)))),
    )


def spend_log_hits(diff: str, blocks: Mapping[str, Block]) -> tuple[str, ...]:
    """`path: +line` or `path: -line` for every changed line that touches the table.

    A schema.prisma line counts when its number falls inside that copy's `blocks` entry (head for added lines,
    merge base for removed ones), a migration line when it is added and names the table."""
    sections: Final = re.split(r"^diff --git [^\n]*\n", diff, flags=re.MULTILINE)
    paths: Final = tuple((_path_of(section), section) for section in sections)
    return tuple(
        hit for path, section in paths if watched(path) for hit in _section_hits(path, _hunks(section), blocks)
    )  # comprehension-ok: path then its hits


def _section_hits(path: str, hunks: Sequence[Hunk], blocks: Mapping[str, Block]) -> tuple[str, ...]:
    if path not in blocks:
        return tuple(f"{path}: +{line.strip()}" for hunk in hunks for _, line in hunk.added if TABLE.search(line))
    block: Final = blocks[path]
    removed: Final = tuple(
        f"{path}: -{line.strip()}" for hunk in hunks for number, line in hunk.removed if number in block.base
    )  # comprehension-ok: hunk then its lines
    added: Final = tuple(
        f"{path}: +{line.strip()}" for hunk in hunks for number, line in hunk.added if number in block.head
    )  # comprehension-ok: hunk then its lines
    return removed + added


def _path_of(section: str) -> str:
    match: Final = re.search(r"^\+\+\+ b/(.+)$", section, flags=re.MULTILINE)
    return match.group(1) if match else ""


def approved_by(reviews: Sequence[Review], login: str) -> bool:
    """True when `login`'s most recent review that changes GitHub's approval state is an approval.

    COMMENTED and PENDING reviews leave an earlier approval in force on GitHub, so they are skipped here too.
    """
    own: Final = tuple(review.state for review in reviews if review.user.login == login)
    states: Final = tuple(state for state in own if state in DECIDING_STATES)
    return bool(states) and states[-1] == "APPROVED"


def latest_approval(reviews: Sequence[Review], login: str) -> Review | None:
    """`login`'s most recent APPROVED review on the pull request, the one GitHub keeps in force after pushes."""
    own: Final = tuple(review for review in reviews if review.user.login == login and review.state == "APPROVED")
    return own[-1] if own else None


def approval_covers(
    approval: Review, head: str, hits: Sequence[str], hits_at: Callable[[str], Sequence[str] | None]
) -> bool:
    """True when the approval is on the head, or the head adds no spend-log line the approved commit did not."""
    if approval.commit_id is None:
        return False
    if approval.commit_id == head:
        return True
    reviewed: Final = hits_at(approval.commit_id)
    return reviewed is not None and set(hits) <= set(reviewed)


def added_diff(base: str, head: str) -> str:
    result: Final = subprocess.run(
        ("git", "diff", "--unified=0", "--no-color", f"{base}..{head}", "--", *WATCHED, MIGRATIONS_PREFIX),
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def file_at(revision: str, path: str) -> str:
    """Contents of `path` at `revision`, empty when the file does not exist there."""
    result: Final = subprocess.run(("git", "show", f"{revision}:{path}"), check=False, capture_output=True, text=True)
    return result.stdout if result.returncode == 0 else ""


def schema_blocks(base: str, head: str) -> Mapping[str, Block]:
    return MappingProxyType(
        {path: Block(base=model_block(file_at(base, path)), head=model_block(file_at(head, path))) for path in WATCHED}
    )


def fetched(commit: str) -> bool:
    result: Final = subprocess.run(("git", "fetch", "--no-tags", "--depth=1", "origin", commit), check=False)
    return result.returncode == 0


def reviewed_hits(base: str, commit: str) -> tuple[str, ...] | None:
    return spend_log_hits(added_diff(base, commit), schema_blocks(base, commit)) if fetched(commit) else None


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
    hits: Final = spend_log_hits(added_diff(args.base, args.head), schema_blocks(args.base, args.head))
    if not hits:
        sys.stdout.write(
            "No change inside model LiteLLM_SpendLogs in schema.prisma and no added migration line names it; "
            "nothing to review.\n"
        )
        return 0
    reviews: Final = _REVIEWS.validate_json(args.reviews.read_bytes())
    approval: Final = latest_approval(reviews, args.approver) if approved_by(reviews, args.approver) else None
    if approval is not None and approval_covers(
        approval, args.head, hits, lambda commit: reviewed_hits(args.base, commit)
    ):
        sys.stdout.write(f"LiteLLM_SpendLogs change approved by {args.approver}: {', '.join(hits)}\n")
        return 0
    listed: Final = "".join(f"  {hit}\n" for hit in hits)
    reason: Final = (
        f"This pull request changes LiteLLM_SpendLogs and needs an approving review from @{args.approver}:\n"
        if approval is None
        else f"LiteLLM_SpendLogs lines changed after @{args.approver} approved {approval.commit_id}, "
        f"a new approving review is needed:\n"
    )
    sys.stdout.write(f"{reason}{listed}After approving, re-run this job.\n")
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
