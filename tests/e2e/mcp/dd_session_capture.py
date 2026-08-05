"""Capture a Datadog browser session for the MCP OAuth e2e tests.

Run this once to log into Datadog and save the browser session:

    uv run python tests/e2e/mcp/dd_session_capture.py

Then set the env var and run the OAuth tests:

    export E2E_DD_STORAGE_STATE=tests/e2e/mcp/.dd_session.json
    uv run pytest tests/e2e/mcp/test_mcp_datadog_oauth_e2e.py -v
"""

from __future__ import annotations

import os
from pathlib import Path

DEFAULT_STATE_PATH = Path(__file__).parent / ".dd_session.json"


def capture(state_path: Path) -> None:
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        context = browser.new_context()
        page = context.new_page()
        page.goto("https://app.datadoghq.com/account/login")
        print("Log into Datadog in the browser, then press Enter here.")
        input()
        context.storage_state(path=str(state_path))
        browser.close()
    print(f"Session saved to {state_path}")
    print(f'  export E2E_DD_STORAGE_STATE="{state_path}"')


if __name__ == "__main__":
    state = Path(os.environ.get("E2E_DD_STORAGE_STATE", str(DEFAULT_STATE_PATH)))
    capture(state)
