"""Capture a Datadog browser session for the MCP OAuth e2e tests.

Run this once to log into Datadog and save the browser session outside the
repo (default: $TMPDIR/litellm-e2e-dd-session.json):

    uv run python tests/e2e/mcp/dd_session_capture.py

Then set the env var and run the OAuth tests:

    export E2E_DD_STORAGE_STATE="$TMPDIR/litellm-e2e-dd-session.json"
    uv run pytest tests/e2e/mcp/test_mcp_datadog_oauth_e2e.py -v
"""

from __future__ import annotations

import os
import stat
import tempfile
from pathlib import Path

DEFAULT_STATE_PATH = Path(tempfile.gettempdir()) / "litellm-e2e-dd-session.json"


def _repo_root() -> Path | None:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / ".git").exists():
            return parent
    return None


def _datadog_login_url() -> str:
    site = (
        os.environ.get("DD_SITE", "datadoghq.com") or "datadoghq.com"
    ).strip().removeprefix("https://").removeprefix("http://").rstrip("/")
    if site.startswith("app."):
        site = site[len("app.") :]
    site = site or "datadoghq.com"
    host = "app.datadoghq.com" if site == "datadoghq.com" else f"app.{site}"
    return f"https://{host}/account/login"


def capture(state_path: Path) -> None:
    state_path = state_path.expanduser().resolve()
    repo = _repo_root()
    if repo is not None and (state_path == repo or repo in state_path.parents):
        raise SystemExit(
            f"Refusing to write session state under the repo ({state_path}). "
            f"Set E2E_DD_STORAGE_STATE to a path outside the tree "
            f"(default: {DEFAULT_STATE_PATH})."
        )
    state_path.parent.mkdir(parents=True, exist_ok=True)

    from playwright.sync_api import sync_playwright

    login_url = _datadog_login_url()
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        context = browser.new_context()
        page = context.new_page()
        page.goto(login_url)
        print(f"Log into Datadog at {login_url} in the browser, then press Enter here.")
        input()
        context.storage_state(path=str(state_path))
        browser.close()
    os.chmod(state_path, stat.S_IRUSR | stat.S_IWUSR)
    print(f"Session saved to {state_path} (mode 0600)")
    print(f'  export E2E_DD_STORAGE_STATE="{state_path}"')


if __name__ == "__main__":
    state = Path(os.environ.get("E2E_DD_STORAGE_STATE", str(DEFAULT_STATE_PATH)))
    capture(state)
