#!/usr/bin/env python3
"""Gate PR bodies on the machine-checkable rules that live inside HTML comments in
.github/pull_request_template.md, since those comments are invisible in rendered PRs
and are stripped from the template copies that agent harnesses inject into context."""

from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass
from functools import reduce
from pathlib import Path
from typing import Final

BULLET_WORD_TARGET: Final = 10
BULLET_WORD_CAP: Final = 14
BULLET_SECTIONS: Final = ("tldr", "caveats (if any)")
QA_RUNBOOK_SECTION: Final = "qa runbook"
E2E_PREFIX: Final = "tests/e2e/"
PLACEHOLDER_TOKENS: Final = ("<blah>",)
NO_CAVEATS_PATTERN: Final = re.compile(r"^(none|n/?a)[.!]?$", re.IGNORECASE)
HTML_COMMENT_PATTERN: Final = re.compile(r"<!--.*?-->", re.DOTALL)
HEADING_PATTERN: Final = re.compile(r"^#{2,6}\s+(?P<title>.+?)\s*$")
BULLET_PATTERN: Final = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+(?P<text>.*)$")
LABEL_PATTERN: Final = re.compile(r"^[^-*+].*:\s*$")


@dataclass(frozen=True, slots=True)
class Violation:
    section: str
    detail: str

    def __str__(self) -> str:
        return f"{self.section}: {self.detail}"


@dataclass(frozen=True, slots=True)
class Section:
    title: str
    lines: tuple[str, ...]


def strip_html_comments(body: str) -> str:
    return HTML_COMMENT_PATTERN.sub("", body.replace("\r\n", "\n"))


def split_sections(body: str) -> tuple[Section, ...]:
    lines: Final = tuple(body.split("\n"))
    headings: Final = tuple(
        (index, match.group("title")) for index, line in enumerate(lines) if (match := HEADING_PATTERN.match(line))
    )
    ends: Final = tuple(index for index, _ in headings[1:]) + (len(lines),)
    return tuple(Section(title=title, lines=lines[start + 1 : end]) for (start, title), end in zip(headings, ends))


def group_bullets(lines: tuple[str, ...]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    def classify(acc: tuple[tuple[str, ...], tuple[str, ...]], line: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
        bullets, prose = acc
        stripped: Final = line.strip()
        if not stripped:
            return acc
        bullet_match: Final = BULLET_PATTERN.match(line)
        if bullet_match:
            return (*bullets, bullet_match.group("text").strip()), prose
        if bullets and line[:1].isspace():
            return (*bullets[:-1], f"{bullets[-1]} {stripped}"), prose
        return bullets, (*prose, stripped)

    return reduce(classify, lines, ((), ()))


def check_bullet_section(section: Section) -> tuple[Violation, ...]:
    bullets, prose = group_bullets(section.lines)
    prose_violations: Final = tuple(
        Violation(section.title, f'prose line "{line[:60]}" must be a short bullet instead')
        for line in prose
        if not LABEL_PATTERN.match(line) and not NO_CAVEATS_PATTERN.match(line)
    )
    length_violations: Final = tuple(
        Violation(
            section.title,
            f'bullet "{bullet[:60]}" has {len(bullet.split())} words;'
            f" keep bullets to roughly {BULLET_WORD_TARGET} words ({BULLET_WORD_CAP} max)",
        )
        for bullet in bullets
        if len(bullet.split()) > BULLET_WORD_CAP
    )
    return prose_violations + length_violations


def is_placeholder_line(line: str) -> bool:
    bullet_match: Final = BULLET_PATTERN.match(line)
    content: Final = bullet_match.group("text").strip() if bullet_match else line.strip()
    return content == "..." or any(token in line for token in PLACEHOLDER_TOKENS)


def check_placeholders(sections: tuple[Section, ...]) -> tuple[Violation, ...]:
    return tuple(
        Violation(section.title, f'template placeholder left in: "{line.strip()[:60]}"')
        for section in sections
        for line in section.lines
        if is_placeholder_line(line)
    )


def check_qa_runbook(sections: tuple[Section, ...], changed_files: tuple[str, ...]) -> tuple[Violation, ...]:
    has_runbook: Final = any(section.title.lower() == QA_RUNBOOK_SECTION for section in sections)
    touches_e2e: Final = any(path.startswith(E2E_PREFIX) for path in changed_files)
    if has_runbook and not touches_e2e:
        return (
            Violation(
                "QA runbook",
                "delete this section; the template only wants it when the PR edits tests/e2e",
            ),
        )
    return ()


def check_body(body: str, changed_files: tuple[str, ...]) -> tuple[Violation, ...]:
    sections: Final = split_sections(strip_html_comments(body))
    bullet_violations: Final = tuple(
        violation
        for section in sections
        if section.title.lower() in BULLET_SECTIONS
        for violation in check_bullet_section(section)
    )
    return bullet_violations + check_placeholders(sections) + check_qa_runbook(sections, changed_files)


def main() -> int:
    parser: Final = argparse.ArgumentParser()
    parser.add_argument("--changed-files", type=Path, required=True)
    args: Final = parser.parse_args()
    body: Final = os.environ.get("PR_BODY", "")
    changed_files: Final = tuple(
        line.strip() for line in args.changed_files.read_text(encoding="utf-8").splitlines() if line.strip()
    )
    violations: Final = check_body(body, changed_files)
    for violation in violations:
        print(f"::error title=PR body template::{violation}")
    if violations:
        print(
            "\nThe rules above come from the HTML comments inside .github/pull_request_template.md;"
            " open that file to see every rule next to its section."
        )
        return 1
    print("PR body follows the template comment rules.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
