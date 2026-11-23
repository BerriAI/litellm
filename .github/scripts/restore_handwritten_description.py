#!/usr/bin/env python3
"""Put a PR author's handwritten description back when a body edit reset it to the template placeholder.

Reads the `pull_request` `edited` event at $GITHUB_EVENT_PATH. When the body before the edit had real text
under the handwritten heading and the body after it has only the placeholder there, patches the PR's live
body so it carries the earlier text again, keeping any edit made since the event. A section the author
emptied on purpose is left alone.

Usage
-----
    GH_TOKEN=... python3 restore_handwritten_description.py
"""

from __future__ import annotations

import json
import os
import pathlib
import re
import subprocess
import sys
from typing import Final

HEADING: Final = "## Handwritten 1-3 sentence description"
PLACEHOLDER: Final = "[insert handwritten description here]"
SECTION: Final = re.compile(
    rf"^{re.escape(HEADING)}[ \t]*\r?\n(?P<text>.*?)(?=^<details>|^##? |\Z)", re.DOTALL | re.MULTILINE
)
HTML_COMMENT: Final = re.compile(r"<!--.*?-->", re.DOTALL)


def handwritten_text(body: str) -> str | None:
    found: Final = SECTION.search(body)
    return None if found is None else HTML_COMMENT.sub("", found["text"]).strip()


def restored_body(before: str, after: str) -> str | None:
    written: Final = handwritten_text(before)
    if not written or PLACEHOLDER in written or handwritten_text(after) != PLACEHOLDER:
        return None
    section: Final = SECTION.search(after)
    assert section is not None
    start: Final = section.start("text") + section["text"].index(PLACEHOLDER)
    return after[:start] + written + after[start + len(PLACEHOLDER) :]


def main() -> int:
    event: Final = json.loads(pathlib.Path(os.environ["GITHUB_EVENT_PATH"]).read_text(encoding="utf-8"))
    before: Final = event.get("changes", {}).get("body", {}).get("from")
    pull_request: Final = event["pull_request"]
    if restored_body(before or "", pull_request.get("body") or "") is None:
        sys.stdout.write("Handwritten description is intact, nothing to restore\n")
        return 0
    live: Final = subprocess.run(
        ("gh", "api", pull_request["url"], "--jq", ".body"), capture_output=True, text=True, check=True
    ).stdout.removesuffix("\n")
    body: Final = restored_body(before or "", live)
    if body is None:
        sys.stdout.write("Handwritten description is intact, nothing to restore\n")
        return 0
    subprocess.run(
        ("gh", "api", "--method", "PATCH", pull_request["url"], "--input", "-", "--silent"),
        input=json.dumps({"body": body}),
        text=True,
        check=True,
    )
    sys.stdout.write(f"Restored the handwritten description on #{pull_request['number']}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
