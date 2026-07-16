"""Management suite fixtures: the client plus a logged-in dashboard page.

Lifecycle/skip/marker live in the parent conftest. The browser fixtures drive
the dashboard the proxy serves at /ui, so browser tests exercise exactly what an
end user sees. playwright is an optional dependency loaded behind importorskip
inside the fixture, so the API tests in this suite collect and run without it:

    uv pip install playwright && uv run playwright install chromium
"""

from typing import TYPE_CHECKING, Iterator

import pytest

from e2e_config import PROXY_BASE_URL, UI_PASSWORD, UI_USERNAME
from management_client import ManagementClient, build_client

if TYPE_CHECKING:
    from playwright.sync_api import Browser, Page


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "covers: registry cell a test covers, e.g. mgmt.key.generate.persists",
    )


@pytest.fixture(scope="session")
def client() -> ManagementClient:
    return build_client()


@pytest.fixture(scope="session")
def browser() -> "Iterator[Browser]":
    pytest.importorskip("playwright.sync_api", reason="playwright not installed")
    from playwright.sync_api import sync_playwright

    with sync_playwright() as playwright:
        launched = playwright.chromium.launch()
        yield launched
        launched.close()


@pytest.fixture
def ui_page(browser: "Browser") -> "Iterator[Page]":
    context = browser.new_context()
    try:
        page = context.new_page()
        page.goto(f"{PROXY_BASE_URL}/ui/")
        page.fill("#username", UI_USERNAME)
        page.fill("#password", UI_PASSWORD)
        page.click('button[type="submit"]')
        page.wait_for_function("() => document.cookie.includes('token=')")
        yield page
    finally:
        context.close()
