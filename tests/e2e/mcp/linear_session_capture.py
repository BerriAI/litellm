"""One-time helper to capture a logged-in Linear browser session for the
real-Linear MCP e2e test.

The real-Linear test drives the genuine gateway-managed authorization_code
dance against ``mcp.linear.app``. The only step that cannot be scripted is
Linear's login (magic link / SSO), so a human authenticates once here and the
resulting session (cookies + local storage) is persisted to disk. The e2e test
then loads that session in a headless Playwright context and clicks Approve on
Linear's consent screen every run, with no human and no login automation.

Run it with the e2e venv, log into Linear in the window that opens, then return
to the terminal and press Enter:

    LITELLM=~/litellm-mcpe2e
    "$LITELLM"/.venv/bin/python "$LITELLM"/tests/e2e/mcp/linear_session_capture.py

The session is written to ``E2E_LINEAR_STORAGE_STATE`` (default
``~/.litellm-e2e/linear_storage_state.json``), outside the repo. It is a
secret: never commit it. Re-run this whenever Linear expires the session.
"""

from __future__ import annotations

import os
from pathlib import Path

from playwright.sync_api import sync_playwright

DEFAULT_STATE_PATH = Path.home() / ".litellm-e2e" / "linear_storage_state.json"


def capture(state_path: Path) -> None:
    """Open a headed browser at Linear, wait for the human to log in, then save
    the authenticated session to ``state_path``."""
    state_path.parent.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=False)
        context = browser.new_context()
        page = context.new_page()
        page.goto("https://linear.app/login", wait_until="domcontentloaded")
        print("\n" + "=" * 72)
        print("Log into Linear in the browser window that just opened.")
        print("If Linear emails you a magic link, paste the link into THIS window's")
        print("address bar (opening it in your default browser won't capture the")
        print("session). Google SSO works too as long as you complete it here.")
        print("When your Linear workspace has loaded, come back and press Enter.")
        print("=" * 72)
        input("Press Enter once you are logged in... ")
        page.goto("https://mcp.linear.app/", wait_until="domcontentloaded")
        context.storage_state(path=str(state_path))
        browser.close()
    print(f"\nSaved Linear session to {state_path}")
    print("Point the e2e test at it with:")
    print(f'  export E2E_LINEAR_STORAGE_STATE="{state_path}"')


if __name__ == "__main__":
    capture(Path(os.environ.get("E2E_LINEAR_STORAGE_STATE", str(DEFAULT_STATE_PATH))))
