"""Fail when a pull request body that uses the general PR template leaves out its required sections."""

import os
import re
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Final

TEMPLATE_PATH: Final = Path(".github/PULL_REQUEST_TEMPLATE/general.md")
REQUIRED_HEADINGS: Final[tuple[str, ...]] = (
    "## Handwritten 1-3 sentence description",
    "## TLDR",
    "## Type",
)
PLACEHOLDER: Final = "<blah>"
AI_SUMMARY_TAG: Final = "<make-me-human-readable>"
AI_SUMMARY_CLOSING_TAG: Final = "</make-me-human-readable>"
HEADING_PATTERN: Final = re.compile(r"^## .+$", re.MULTILINE)
SECTION_PATTERN: Final = re.compile(r"^## .+$\n(?P<content>(?:(?!^## ).*\n?)*)", re.MULTILINE)


def template_headings(template_text: str) -> tuple[str, ...]:
    return tuple(HEADING_PATTERN.findall(template_text))


def body_headings(body: str) -> tuple[str, ...]:
    return tuple(heading.strip() for heading in HEADING_PATTERN.findall(body))


def uses_template(found: Iterable[str], known: Iterable[str]) -> bool:
    return any(heading in known for heading in found)


def section_is_blank(body: str, heading: str) -> bool:
    return any(
        match.group().startswith(heading + "\n") and not match.group("content").strip()
        for match in SECTION_PATTERN.finditer(body)
    )


def problems(body: str, template_text: str) -> tuple[str, ...]:
    known: Final = template_headings(template_text)
    found: Final = body_headings(body)
    if not uses_template(found, known):
        return ()
    missing: Final = tuple(f"missing section {heading!r}" for heading in REQUIRED_HEADINGS if heading not in found)
    first: Final = found[0] if found else None
    misplaced: Final = (
        (f"{REQUIRED_HEADINGS[0]!r} must be the first section, found {first!r} first",)
        if REQUIRED_HEADINGS[0] in found and first != REQUIRED_HEADINGS[0]
        else ()
    )
    blank: Final = (
        (f"the section under {REQUIRED_HEADINGS[0]!r} is empty: write the 1-3 sentence summary there",)
        if REQUIRED_HEADINGS[0] in found and section_is_blank(body, REQUIRED_HEADINGS[0])
        else ()
    )
    placeholder: Final = (f"template placeholder {PLACEHOLDER!r} left in the body",) if PLACEHOLDER in body else ()
    ai_summary: Final = (
        (
            f"the summary under {REQUIRED_HEADINGS[0]!r} is still AI-written: the PR author rewrites it in their own "
            f"words and deletes the {AI_SUMMARY_TAG} tag lines",
        )
        if AI_SUMMARY_TAG in body or AI_SUMMARY_CLOSING_TAG in body
        else ()
    )
    return (*missing, *misplaced, *blank, *placeholder, *ai_summary)


def main() -> int:
    body: Final = os.environ.get("PR_BODY", "")
    found: Final = problems(body, TEMPLATE_PATH.read_text())
    for problem in found:
        print(f"::error::PR body: {problem}")  # noqa: T201  # CI annotation
    if found:
        print(  # noqa: T201  # CI log
            f"Keep every heading of {TEMPLATE_PATH} you have content for, verbatim, in the template's order"
        )
        return 1
    print("PR body follows the template")  # noqa: T201  # CI log
    return 0


if __name__ == "__main__":
    sys.exit(main())
