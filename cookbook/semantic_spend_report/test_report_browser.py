import json
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
DATA = json.loads((ARTIFACTS / "presentation-data.json").read_text())
LABELS = {c["request_id"]: c["category"] for c in DATA["classifications"]}


def main() -> None:
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            executable_path="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
            headless=True,
        )
        page = browser.new_page(viewport={"width": 1600, "height": 1120}, device_scale_factor=1)
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto((ARTIFACTS / "report.html").as_uri())
        page.wait_for_selector("#ranking-body tr[data-entity]")
        assert page.locator("#request-count").inner_text() == "24"
        assert page.locator("#ranking-body tr[data-entity]").count() == 3
        total = sum(float(r["spend_usd"]) for r in DATA["requests"])
        assert abs(float(page.locator("#total-spend").get_attribute("data-value")) - total) < 1e-12
        amounts = page.locator("#ranking-body tr[data-entity]").evaluate_all(
            "rows => rows.map(r => Number(r.dataset.spend))"
        )
        assert amounts == sorted(amounts, reverse=True)
        assert abs(sum(amounts) - total) < 1e-12
        bucket_spend = page.locator("[data-bucket-spend]").evaluate_all(
            "nodes => nodes.reduce((sum, n) => sum + Number(n.dataset.bucketSpend), 0)"
        )
        bucket_count = page.locator("[data-bucket-count]").evaluate_all(
            "nodes => nodes.reduce((sum, n) => sum + Number(n.dataset.bucketCount), 0)"
        )
        assert abs(bucket_spend - total) < 1e-12
        assert bucket_count == len(DATA["requests"])
        page.screenshot(path=str(ARTIFACTS / "rankings-overview.png"), full_page=True)
        page.locator(".entity-name").first.click()
        assert page.locator("#detail-drawer").is_visible()
        assert 2 <= page.locator("blockquote").count() <= 3
        assert page.locator(".findings li").count() <= 3
        page.screenshot(path=str(ARTIFACTS / "rankings-session-detail.png"))
        page.locator(".quote-source").first.focus()
        page.keyboard.press("Enter")
        assert page.locator("#detail-drawer").is_visible()
        assert page.locator(".request-details[open]").count() == 1
        page.keyboard.press("Escape")
        assert not page.locator("#detail-drawer").is_visible()
        for category in DATA["categories"]:
            cid = category["id"]
            page.locator(f'#category-tabs button[data-category="{cid}"]').click()
            expected = [r for r in DATA["requests"] if LABELS[r["request_id"]] == cid]
            assert int(page.locator("#request-count").inner_text()) == len(expected)
            cost = sum(float(r["spend_usd"]) for r in expected)
            assert abs(float(page.locator("#total-spend").get_attribute("data-value")) - cost) < 1e-12
            assert page.locator("#timeline svg").get_attribute("data-request-count") == str(len(expected))
            page.locator(".entity-name").first.click()
            visible_quotes = page.locator(".quote-source").evaluate_all("nodes => nodes.map(n => n.dataset.evidence)")
            assert all(LABELS[rid] == cid for rid in visible_quotes)
            page.keyboard.press("Escape")
        page.locator("#reset-filters").click()
        page.locator('[data-view="categories"]').click()
        assert page.locator("#ranking-body tr[data-entity]").count() == 6
        page.get_by_role("button", name="Docs", exact=True).click()
        assert page.locator("#drawer-title").inner_text() == "Docs"
        page.screenshot(path=str(ARTIFACTS / "rankings-category-detail.png"))
        page.keyboard.press("Escape")
        page.locator("#reset-filters").click()
        page.locator("#period-filter").select_option("first")
        first_count = int(page.locator("#request-count").inner_text())
        assert 0 < first_count < 24
        page.locator(".entity-name").first.click()
        assert page.locator(".scope-warning").is_visible()
        page.keyboard.press("Escape")
        page.locator("#period-filter").select_option("last")
        last_count = int(page.locator("#request-count").inner_text())
        assert first_count + last_count == 24
        page.locator("#start-filter").fill("2030-01-01T00:00")
        page.locator("#start-filter").dispatch_event("change")
        assert page.locator("#filter-error").is_visible()
        page.locator("#end-filter").fill("2030-01-01T00:01")
        page.locator("#end-filter").dispatch_event("change")
        assert page.locator("#request-count").inner_text() == "0"
        assert "No requests match" in page.locator("#ranking-body").inner_text()
        page.locator("#reset-filters").click()
        page.locator("#user-filter").select_option(DATA["manifest"]["user_id"])
        assert page.locator("#request-count").inner_text() == "24"
        page.locator("#sort-filter").select_option("requests")
        counts = page.locator("#ranking-body tr[data-entity]").evaluate_all(
            "rows => rows.map(r => Number(r.dataset.count))"
        )
        assert counts == sorted(counts, reverse=True)
        page.set_viewport_size({"width": 390, "height": 844})
        page.screenshot(path=str(ARTIFACTS / "rankings-mobile.png"), full_page=True)
        assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
        page.locator(".entity-name").first.click()
        assert page.locator("#detail-drawer").bounding_box()["width"] <= 390
        assert not errors, errors
        browser.close()
        (ARTIFACTS / "browser-verification.json").write_text(
            json.dumps(
                {
                    "passed": True,
                    "browser": "Installed Chrome via Playwright",
                    "page_errors": errors,
                    "checks": [
                        "initial totals",
                        "plotted bucket totals",
                        "keyboard quote navigation",
                        "descending spend ranking",
                        "session drawer",
                        "quote evidence expansion",
                        "Escape dismissal",
                        "all six category filters",
                        "filtered chart counts",
                        "quote scope",
                        "category drawer",
                        "partial-window summary handling",
                        "window partition",
                        "invalid range",
                        "empty range",
                        "user filter",
                        "request sorting",
                        "mobile layout",
                        "mobile drawer",
                    ],
                    "first_half_requests": first_count,
                    "second_half_requests": last_count,
                },
                indent=2,
            )
        )
        print("Browser checks passed")


if __name__ == "__main__":
    main()
