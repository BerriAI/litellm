import { chromium } from "playwright";

const BASE = "http://127.0.0.1:21501";
const EXEC =
  "/Users/yucheng/Library/Caches/ms-playwright/chromium_headless_shell-1234/chrome-headless-shell-mac-arm64/chrome-headless-shell/chrome-headless-shell";

const browser = await chromium.launch({
  executablePath:
    "/Users/yucheng/Library/Caches/ms-playwright/chromium_headless_shell-1234/chrome-headless-shell-mac-arm64/chrome-headless-shell",
  headless: true,
});
const ctx = await browser.newContext({ viewport: { width: 1600, height: 1200 } });
const page = await ctx.newPage();
const errs = [];
page.on("console", (m) => errs.push(`[${m.type()}] ${m.text().slice(0, 500)}`));
page.on("pageerror", (e) => errs.push("PAGEERROR: " + (e && e.stack ? e.stack.slice(0, 1200) : String(e))));

await page.goto(`${BASE}/ui/login`, { waitUntil: "domcontentloaded" });
await page.getByPlaceholder("Enter your username").fill("admin");
await page.getByPlaceholder("Enter your password").fill("sk-uitrack-21501");
await page.getByRole("button", { name: "Login", exact: true }).click();
await page.waitForLoadState("networkidle");
console.log("after login url:", page.url());

await page.goto(`${BASE}/ui/logging-and-alerts`, { waitUntil: "domcontentloaded" });
await page.waitForLoadState("networkidle");
await page.waitForTimeout(4000);
console.log("url:", page.url());
console.log("body:", (await page.evaluate(() => document.body.innerText)).slice(0, 1500));
console.log("tables:", await page.evaluate(() => document.querySelectorAll("table").length));
console.log("---- console/pageerrors ----");
for (const e of errs) console.log(e);
await browser.close();
