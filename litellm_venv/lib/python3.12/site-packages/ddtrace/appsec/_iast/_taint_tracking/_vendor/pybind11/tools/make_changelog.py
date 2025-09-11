#!/usr/bin/env python3
from __future__ import annotations

import re

import ghapi.all
from rich import print
from rich.syntax import Syntax

ENTRY = re.compile(
    r"""
    Suggested \s changelog \s entry:
    .*
    ```rst
    \s*
    (.*?)
    \s*
    ```
""",
    re.DOTALL | re.VERBOSE,
)

print()


api = ghapi.all.GhApi(owner="pybind", repo="pybind11")

issues_pages = ghapi.page.paged(
    api.issues.list_for_repo, labels="needs changelog", state="closed"
)
issues = (issue for page in issues_pages for issue in page)
missing = []
cats_descr = {
    "feat": "New Features",
    "fix": "Bug fixes",
    "fix(types)": "",
    "fix(cmake)": "",
    "docs": "Documentation",
    "tests": "Tests",
    "ci": "CI",
    "chore": "Other",
    "unknown": "Uncategorised",
}
cats: dict[str, list[str]] = {c: [] for c in cats_descr}

for issue in issues:
    changelog = ENTRY.findall(issue.body or "")
    if not changelog or not changelog[0]:
        missing.append(issue)
    else:
        (msg,) = changelog
        if msg.startswith("- "):
            msg = msg[2:]
        if not msg.startswith("* "):
            msg = "* " + msg
        if not msg.endswith("."):
            msg += "."

        msg += f"\n  `#{issue.number} <{issue.html_url}>`_"
        for cat in cats:
            if issue.title.lower().startswith(f"{cat}:"):
                cats[cat].append(msg)
                break
        else:
            cats["unknown"].append(msg)

for cat, msgs in cats.items():
    if msgs:
        desc = cats_descr[cat]
        print(f"[bold]{desc}:" if desc else f".. {cat}")
        print()
        for msg in msgs:
            print(Syntax(msg, "rst", theme="ansi_light", word_wrap=True))
            print()
        print()

if missing:
    print()
    print("[blue]" + "-" * 30)
    print()

    for issue in missing:
        print(f"[red bold]Missing:[/red bold][red] {issue.title}")
        print(f"[red]  {issue.html_url}\n")

    print("[bold]Template:\n")
    msg = "## Suggested changelog entry:\n\n```rst\n\n```"
    print(Syntax(msg, "md", theme="ansi_light"))

print()
