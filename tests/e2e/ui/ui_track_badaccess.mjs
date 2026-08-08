import { chromium } from "playwright";
import fs from "node:fs";

const BASE = "http://127.0.0.1:21501";
const EXEC =
  "/Users/yucheng/Library/Caches/ms-playwright/chromium_headless_shell-1234/chrome-headless-shell-mac-arm64/chrome-headless-shell";
const results = [];
const notes = [];
function check(id, desc, ok, actual, expected) {
  results.push({ id, desc, status: ok ? "PASS" : "FAIL", actual: ok ? undefined : actual, expected: ok ? undefined : expected });
}

const who = process.argv[2] || "admin";
const creds = who === "viewer" ? ["uitrk-viewer@example.com", "uitrk-viewer-pw"] : ["admin", "sk-uitrack-21501"];

const browser = await chromium.launch({ executablePath: EXEC, headless: true });
const ctx = await browser.newContext({ viewport: { width: 1600, height: 1200 } });
const page = await ctx.newPage();
const errs = [];
page.on("pageerror", (e) => errs.push(String(e).slice(0, 400)));

await page.goto(`${BASE}/ui/login`, { waitUntil: "domcontentloaded" });
await page.getByPlaceholder("Enter your username").fill(creds[0]);
await page.getByPlaceholder("Enter your password").fill(creds[1]);
await page.getByRole("button", { name: "Login", exact: true }).click();
await page.waitForLoadState("networkidle");

await page.goto(`${BASE}/ui/logging-and-alerts`, { waitUntil: "domcontentloaded" });
await page.waitForLoadState("networkidle");
await page.waitForTimeout(4000);
const body = await page.evaluate(() => document.body.innerText);
const tables = await page.evaluate(() => document.querySelectorAll("table").length);
notes.push({ k: "body", v: body.slice(0, 400) });
notes.push({ k: "pageerrors", v: errs.slice(0, 4) });
notes.push({ k: "tables", v: tables });
await page.screenshot({ path: `/tmp/uitrack_badaccess_${who}.png`, fullPage: true }).catch(() => {});

const rendered = tables > 0 && body.includes("Active Logging Callbacks");
check(
  `BAD1-${who}`,
  `Destinations page renders for ${who} with one stored destination whose credential_info.access is a malformed (non-list) teams value`,
  rendered,
  `tables=${tables}; body="${body.slice(0, 120).replace(/\n/g, " / ")}"; pageerror="${(errs[0] || "").slice(0, 160)}"`,
  "table renders, malformed row shows a dash for Scope",
);

// blast radius: is any other dashboard page affected?
await page.goto(`${BASE}/ui/teams`, { waitUntil: "domcontentloaded" });
await page.waitForLoadState("networkidle");
await page.waitForTimeout(2500);
const teamsBody = await page.evaluate(() => document.body.innerText);
notes.push({ k: "teamsBody", v: teamsBody.slice(0, 200) });
check(
  `BAD2-${who}`,
  `The failure is scoped to the destinations page (/ui/teams still renders) for ${who}`,
  !teamsBody.includes("This page couldn"),
  teamsBody.slice(0, 150),
  "/ui/teams renders normally",
);

await browser.close();
fs.writeFileSync(`/tmp/uitrack_badaccess_${who}.json`, JSON.stringify({ results, notes }, null, 2));
console.log(JSON.stringify(results, null, 2));
console.log("NOTES", JSON.stringify(notes, null, 2).slice(0, 1500));
