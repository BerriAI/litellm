import { chromium } from "playwright";

const BASE = "http://127.0.0.1:21501";
const EXEC =
  "/Users/yucheng/Library/Caches/ms-playwright/chromium_headless_shell-1234/chrome-headless-shell-mac-arm64/chrome-headless-shell";
const who = process.argv[2] || "viewer";
const creds = who === "viewer" ? ["uitrk-viewer@example.com", "uitrk-viewer-pw"] : ["admin", "sk-uitrack-21501"];

const browser = await chromium.launch({ executablePath: EXEC, headless: true });
const ctx = await browser.newContext({ viewport: { width: 1600, height: 1200 } });
const page = await ctx.newPage();
const calls = [];
page.on("response", (r) => {
  if (r.url().includes("/health/services")) calls.push({ url: r.url(), status: r.status() });
});

await page.goto(`${BASE}/ui/login`, { waitUntil: "domcontentloaded" });
await page.getByPlaceholder("Enter your username").fill(creds[0]);
await page.getByPlaceholder("Enter your password").fill(creds[1]);
await page.getByRole("button", { name: "Login", exact: true }).click();
await page.waitForLoadState("networkidle");
await page.goto(`${BASE}/ui/logging-and-alerts`, { waitUntil: "domcontentloaded" });
await page.waitForLoadState("networkidle");
await page.waitForTimeout(3000);

await page.locator('[data-testid="callback-actions-datadog-success_and_failure"]').click();
await page.waitForTimeout(500);
await page.locator('[data-testid="callback-action-test"]').click();
await page.waitForTimeout(4000);
console.log(who, "health/services calls:", JSON.stringify(calls));
const toast = await page.evaluate(() =>
  [...document.querySelectorAll('[class*="toast"], .ant-message, [role="status"], [role="alert"]')]
    .map((e) => e.innerText.trim())
    .filter(Boolean)
    .slice(0, 5),
);
console.log(who, "toast:", JSON.stringify(toast));
await browser.close();
