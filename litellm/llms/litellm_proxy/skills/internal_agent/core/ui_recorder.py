"""
Playwright-based browser UI recorder for the LiteLLM internal repro agent.

Records a .webm video of what an affected user sees in the LiteLLM dashboard,
so repro reports include UI evidence alongside the API trace.

Usage:
    with UIRecorder(base_url="http://localhost:4000", output_path="/sandbox/ui_repro.webm") as r:
        r.login(token="sk-1234")
        r.navigate("/ui/virtual-keys")
        r.click("Create Key")
        r.select_dropdown("Team", "T1")
        r.annotate("Project dropdown is empty for internal_user — bug confirmed")
"""

import json
import os
import time
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from playwright.sync_api import Page


class UIRecorder:
    """
    Records browser video of LiteLLM dashboard interactions.

    Wraps Playwright sync API with a context manager so the video
    file is flushed automatically on __exit__.
    """

    def __init__(self, base_url: str, output_path: str = "/sandbox/ui_repro.webm"):
        self.base_url = base_url.rstrip("/")
        self.output_path = output_path
        self._playwright = None
        self._browser = None
        self._context = None
        self.page: Optional["Page"] = None

    def __enter__(self):
        from playwright.sync_api import sync_playwright

        video_dir = os.path.dirname(self.output_path)
        os.makedirs(video_dir, exist_ok=True)

        self._playwright = sync_playwright().start()
        self._browser = self._playwright.chromium.launch(
            headless=True,
            args=["--no-sandbox", "--disable-setuid-sandbox"],
        )
        self._context = self._browser.new_context(
            viewport={"width": 1280, "height": 800},
            record_video_dir=video_dir,
            record_video_size={"width": 1280, "height": 800},
        )
        self.page = self._context.new_page()
        return self

    def __exit__(self, *_):
        # Closing the context flushes the video file to disk
        if self._context:
            self._context.close()
        if self._browser:
            self._browser.close()
        if self._playwright:
            self._playwright.stop()

        # Playwright names the file by tab ID; rename to the requested output_path
        video_dir = os.path.dirname(self.output_path)
        webm_files = [f for f in os.listdir(video_dir) if f.endswith(".webm")]
        if webm_files:
            src = os.path.join(video_dir, webm_files[0])
            if src != self.output_path:
                os.rename(src, self.output_path)

    # ------------------------------------------------------------------
    # High-level helpers the skill code calls
    # ------------------------------------------------------------------

    def _page(self) -> "Page":
        assert self.page is not None, "UIRecorder must be used as a context manager"
        return self.page

    def login(
        self,
        token: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
    ):
        """
        Log in to the LiteLLM UI.

        Prefer token-based login (sets sessionStorage directly) to avoid flaky
        form interactions in headless mode.
        """
        page = self._page()
        page.goto(f"{self.base_url}/ui/login")
        page.wait_for_load_state("networkidle")

        if token:
            page.evaluate(f"sessionStorage.setItem('token', '{token}')")
            page.goto(f"{self.base_url}/ui")
            page.wait_for_load_state("networkidle")
        elif username and password:
            page.fill("[data-testid='username']", username)
            page.fill("[data-testid='password']", password)
            page.click("[data-testid='login-btn']")
            page.wait_for_load_state("networkidle")

    def navigate(self, path: str):
        """Navigate to a UI path and wait for the page to settle."""
        page = self._page()
        page.goto(f"{self.base_url}{path}")
        page.wait_for_load_state("networkidle")
        time.sleep(0.5)  # brief pause so the viewport is stable in the recording

    def click(self, label: str):
        """Click the first visible element whose text matches label."""
        page = self._page()
        page.get_by_text(label, exact=False).first.click()
        page.wait_for_load_state("networkidle")
        time.sleep(0.3)

    def select_dropdown(self, label: str, value: str):
        """
        Open an antd Select whose label contains `label` and pick `value`.

        Antd dropdowns render options outside the Select component, so we click
        the selector to open it then click the matching option.
        """
        page = self._page()
        page.locator(f"text={label}").first.click()
        time.sleep(0.3)
        page.locator(f".ant-select-item-option-content >> text={value}").first.click()
        time.sleep(0.3)

    def annotate(self, message: str):
        """
        Inject a visible overlay banner into the page for the recording.

        Purely visual — makes it clear in the video what the agent is observing.
        """
        page = self._page()
        # json.dumps produces a safe JS string literal (handles quotes, backslashes, etc.)
        js_string = json.dumps(message)
        page.evaluate(
            f"""() => {{
            const banner = document.createElement('div');
            banner.innerText = {js_string};
            banner.style.position = 'fixed';
            banner.style.bottom = '16px';
            banner.style.left = '50%';
            banner.style.transform = 'translateX(-50%)';
            banner.style.background = 'rgba(0,0,0,0.82)';
            banner.style.color = '#fff';
            banner.style.fontSize = '14px';
            banner.style.padding = '8px 20px';
            banner.style.borderRadius = '6px';
            banner.style.zIndex = '99999';
            banner.style.maxWidth = '90vw';
            banner.style.textAlign = 'center';
            document.body.appendChild(banner);
            setTimeout(() => banner.remove(), 3500);
        }}"""
        )
        time.sleep(3.6)

    def screenshot(self, name: str = "screenshot.png") -> str:
        """Save a full-page screenshot to /sandbox/."""
        path = f"/sandbox/{name}"
        self._page().screenshot(path=path, full_page=True)
        return path
