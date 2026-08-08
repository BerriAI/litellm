import { chromium } from "playwright";
import fs from "node:fs";

const BASE = "http://127.0.0.1:21501";
const EXEC =
  "/Users/yucheng/Library/Caches/ms-playwright/chromium_headless_shell-1234/chrome-headless-shell-mac-arm64/chrome-headless-shell";
const ADMIN_USER = "admin";
const ADMIN_PASS = "sk-uitrack-21501";
const VIEWER_USER = "uitrk-viewer@example.com";
const VIEWER_PASS = "uitrk-viewer-pw";
const OUT = process.argv[2] || "/tmp/ui_track_result.json";

const results = [];
const notes = [];
let consoleErrors = [];

function check(id, desc, ok, actual, expected) {
  results.push({ id, desc, status: ok ? "PASS" : "FAIL", actual: ok ? undefined : actual, expected: ok ? undefined : expected });
}

function note(k, v) {
  notes.push({ k, v });
}

async function login(page, user, pass) {
  await page.goto(`${BASE}/ui/login`, { waitUntil: "domcontentloaded" });
  await page.getByPlaceholder("Enter your username").fill(user);
  await page.getByPlaceholder("Enter your password").fill(pass);
  await page.getByRole("button", { name: "Login", exact: true }).click();
  await page.waitForLoadState("networkidle");
}

async function dismissFeedback(page) {
  const b = page.getByText("Don't ask me again");
  if (await b.isVisible({ timeout: 1500 }).catch(() => false)) {
    await b.click().catch(() => {});
  }
}

async function gotoDestinations(page) {
  await page.goto(`${BASE}/ui/logging-and-alerts`, { waitUntil: "domcontentloaded" });
  await page.waitForLoadState("networkidle");
  await dismissFeedback(page);
  // wait for the destination rows to land (credentials fetch is async)
  await page
    .locator("table tbody tr", { hasText: "uitrk-global" })
    .first()
    .waitFor({ timeout: 20000 })
    .catch(() => {});
  await page.waitForTimeout(1200);
}

async function scrape(page) {
  return page.evaluate(() => {
    const tables = [...document.querySelectorAll("table")];
    const table = tables[0];
    if (!table) return { headers: [], rows: [], tableCount: tables.length, bodyText: document.body.innerText.slice(0, 800) };
    const headers = [...table.querySelectorAll("thead th")].map((th) => th.innerText.trim());
    const rows = [...table.querySelectorAll("tbody tr")].map((tr) => {
      const tds = [...tr.querySelectorAll("td")];
      const nameCell = tds[0];
      const spans = nameCell ? [...nameCell.querySelectorAll("span")] : [];
      const actionsCell = tds[tds.length - 1];
      const trigger = actionsCell ? actionsCell.querySelector('[data-testid^="callback-actions-"]') : null;
      const scopeCell = tds[2];
      return {
        cells: tds.map((td) => td.innerText.trim()),
        name: spans[0] ? spans[0].innerText.trim() : "",
        sub: spans[1] ? spans[1].innerText.trim() : "",
        mode: tds[1] ? tds[1].innerText.trim() : "",
        scope: scopeCell ? scopeCell.innerText.trim() : "",
        scopeBadges: scopeCell
          ? [...scopeCell.querySelectorAll("span")].map((e) => e.innerText.trim()).filter(Boolean)
          : [],
        scopeTitle: scopeCell && scopeCell.querySelector("[title]") ? scopeCell.querySelector("[title]").getAttribute("title") : null,
        hasTrigger: !!trigger,
        triggerTestId: trigger ? trigger.getAttribute("data-testid") : null,
      };
    });
    return { headers, rows, tableCount: tables.length, bodyText: "" };
  });
}

async function openMenu(page, testId) {
  await page.locator(`[data-testid="${testId}"]`).click();
  await page.waitForTimeout(500);
  const items = await page.evaluate(() => {
    const menu = document.querySelector('[role="menu"]');
    if (!menu) return null;
    return [...menu.querySelectorAll('[role="menuitem"]')].map((el) => ({
      text: el.innerText.trim(),
      testId: el.getAttribute("data-testid"),
    }));
  });
  return items;
}

async function closeMenu(page) {
  await page.keyboard.press("Escape");
  await page.waitForTimeout(300);
}

async function apiGet(path) {
  const r = await fetch(`${BASE}${path}`, { headers: { Authorization: `Bearer ${ADMIN_PASS}` } });
  return r.json();
}

const EXPECTED = {
  "uitrk-global": { sub: "Generic OTLP Collector · http://127.0.0.1:21599/v1/traces", scope: "Global access" },
  "uitrk-team": { sub: "Arize", scope: "team: uitrk-team-1" },
  "uitrk-org": { sub: "Weave", scope: "org: uitrk-org-1" },
  "uitrk-team-and-org": { sub: "Langfuse OTEL · https://cloud.langfuse.com", scope: "team: uitrk-team-2|org: uitrk-org-2" },
  "uitrk-many-teams": { sub: "Generic OTLP Collector", scope: "FOUR_PLUS_MORE" },
  "uitrk-no-access": { sub: "Generic OTLP Collector", scope: "—" },
  "uitrk-empty-access": { sub: "Generic OTLP Collector", scope: "—" },
  "uitrk-no-description": { sub: "-", scope: "Not active" },
  "uitrk-adapter-reject": { sub: "Langfuse OTEL", scope: "Not active" },
  datadog: { sub: "Generic OTLP Collector", scope: "team: uitrk-team-1" },
};
const DEST_COUNT = 10;

(async () => {
  const browser = await chromium.launch({ executablePath: EXEC, headless: true });
  const ctx = await browser.newContext({ viewport: { width: 1600, height: 1200 } });
  const page = await ctx.newPage();
  page.on("console", (m) => {
    if (m.type() === "error") consoleErrors.push(m.text().slice(0, 300));
  });
  page.on("pageerror", (e) => consoleErrors.push("PAGEERROR: " + String(e).slice(0, 300)));

  const creds = await apiGet("/credentials");
  const rtd = Object.fromEntries(
    creds.credentials
      .filter((c) => (c.credential_info || {}).credential_type === "logging")
      .map((c) => [c.credential_name, c.resolves_to_destination]),
  );
  note("resolves_to_destination", rtd);

  // ---------- ADMIN ----------
  await login(page, ADMIN_USER, ADMIN_PASS);
  await gotoDestinations(page);

  const url = page.url();
  check("N1", "Destinations page renders at path route /ui/logging-and-alerts", url.includes("/ui/logging-and-alerts"), url, "/ui/logging-and-alerts");

  const tabVisible = await page.getByText("Logging Callbacks", { exact: true }).first().isVisible().catch(() => false);
  check("N2", "Logging Callbacks tab is exposed", tabVisible, String(tabVisible), "true");

  const headingVisible = await page.getByText("Active Logging Callbacks", { exact: true }).first().isVisible().catch(() => false);
  check("T2", '"Active Logging Callbacks" heading present', headingVisible, String(headingVisible), "true");

  let scraped = await scrape(page);
  note("adminScrape", scraped);
  await page.screenshot({ path: "/tmp/uitrack_admin_table.png", fullPage: true }).catch(() => {});

  const hdr = scraped.headers.map((h) => h.trim());
  check("T1", "Table has Callback Name / Mode / Scope columns", hdr[0] === "Callback Name" && hdr[1] === "Mode" && hdr[2] === "Scope", JSON.stringify(hdr), '["Callback Name","Mode","Scope",...]');
  check("T3", "Actions column header is screen-reader-only (renders empty)", hdr.length === 4 && hdr[3] === "Actions", JSON.stringify(hdr), '4 headers, last "Actions"');

  const byName = {};
  for (const r of scraped.rows) {
    const k = r.name + "|" + r.mode;
    byName[k] = r;
  }
  const destRows = scraped.rows.filter((r) => r.mode === "—");
  const cfgRows = scraped.rows.filter((r) => r.mode !== "—");
  check("T4", "Config callbacks and destinations share one table", scraped.tableCount >= 1 && destRows.length > 0 && cfgRows.length > 0, `tables=${scraped.tableCount} dest=${destRows.length} cfg=${cfgRows.length}`, "one table containing both kinds");
  check("T5", "All 10 destination fixtures render as rows", destRows.length === DEST_COUNT, `${destRows.length} destination rows: ${destRows.map((r) => r.name).join(",")}`, "10");

  // per-fixture checks
  for (const [name, exp] of Object.entries(EXPECTED)) {
    const row = destRows.find((r) => r.name === name);
    const idBase = "R-" + name;
    if (!row) {
      check(idBase + "-name", `Row "${name}" present with its own name`, false, "row not found", name);
      check(idBase + "-sub", `Row "${name}" backend sub-line`, false, "row not found", exp.sub);
      check(idBase + "-mode", `Row "${name}" Mode is a dash`, false, "row not found", "—");
      check(idBase + "-scope", `Row "${name}" Scope`, false, "row not found", exp.scope);
      continue;
    }
    check(idBase + "-name", `Row "${name}" present with its own name`, row.name === name, row.name, name);
    check(idBase + "-sub", `Row "${name}" backend sub-line`, row.sub === exp.sub, row.sub, exp.sub);
    check(idBase + "-mode", `Row "${name}" Mode is a dash (destinations have no mode)`, row.mode === "—", row.mode, "—");

    let scopeOk;
    let scopeExp = exp.scope;
    if (exp.scope === "FOUR_PLUS_MORE") {
      const badges = row.scope.split("\n").map((s) => s.trim()).filter(Boolean);
      const teamBadges = badges.filter((b) => b.startsWith("team:"));
      scopeOk = teamBadges.length === 4 && badges.includes("+2 more");
      scopeExp = "4 team badges + '+2 more'";
    } else if (exp.scope.includes("|")) {
      scopeOk = exp.scope.split("|").every((part) => row.scope.includes(part));
    } else {
      scopeOk = row.scope.replace(/\s+/g, " ").trim() === exp.scope;
    }
    check(idBase + "-scope", `Row "${name}" Scope cell`, scopeOk, JSON.stringify(row.scope), scopeExp);
  }

  // Cross-check: Not active <=> resolves_to_destination === false
  for (const name of Object.keys(EXPECTED)) {
    const row = destRows.find((r) => r.name === name);
    if (!row) continue;
    const uiNotActive = row.scope.includes("Not active");
    const backendFalse = rtd[name] === false;
    check(
      "X-rtd-" + name,
      `"Not active" for ${name} matches GET /credentials resolves_to_destination`,
      uiNotActive === backendFalse,
      `ui_not_active=${uiNotActive} resolves_to_destination=${rtd[name]}`,
      "the two agree",
    );
  }

  // Cross-check against /team/info + /organization/info resolved_logging_exporters
  const teamsList = await apiGet("/team/list");
  const teamIdByAlias = Object.fromEntries((teamsList || []).map((t) => [t.team_alias, t.team_id]));
  note("teamIdByAlias", teamIdByAlias);
  const teamExpect = {
    "uitrk-team-1": ["uitrk-global", "uitrk-team", "uitrk-many-teams", "datadog"],
    "uitrk-team-2": ["uitrk-global", "uitrk-team-and-org", "uitrk-many-teams"],
    "uitrk-team-3": ["uitrk-global", "uitrk-many-teams"],
    "uitrk-team-7": ["uitrk-global"],
  };
  for (const [alias, expNames] of Object.entries(teamExpect)) {
    const info = await apiGet(`/team/info?team_id=${teamIdByAlias[alias]}`);
    const actual = ((info.team_info || info).resolved_logging_exporters || []).slice().sort();
    // what the UI Scope column claims for this team: rows whose scope shows Global access,
    // or a team badge naming this alias (expanded badges only)
    const uiClaim = destRows
      .filter((r) => r.scope.includes("Global access") || r.scope.includes(`team: ${alias}`))
      .map((r) => r.name)
      .sort();
    const missingFromUi = actual.filter((n) => !uiClaim.includes(n));
    // uitrk-many-teams collapses behind "+N more" for teams 5/6; allow the collapse
    const collapsed = missingFromUi.filter((n) => n !== "uitrk-many-teams");
    check(
      "X-team-" + alias,
      `Scope column agrees with /team/info resolved_logging_exporters for ${alias}`,
      collapsed.length === 0,
      `backend=${JSON.stringify(actual)} ui=${JSON.stringify(uiClaim)}`,
      "no backend-granted destination missing from the UI scope (modulo +N more collapse)",
    );
  }
  const orgList = await apiGet("/organization/list");
  const orgIdByAlias = Object.fromEntries((orgList || []).map((o) => [o.organization_alias, o.organization_id]));
  for (const alias of ["uitrk-org-1", "uitrk-org-2"]) {
    const info = await apiGet(`/organization/info?organization_id=${orgIdByAlias[alias]}`);
    const actual = (info.resolved_logging_exporters || []).slice().sort();
    const uiClaim = destRows
      .filter((r) => r.scope.includes("Global access") || r.scope.includes(`org: ${alias}`))
      .map((r) => r.name)
      .sort();
    check(
      "X-org-" + alias,
      `Scope column agrees with /organization/info resolved_logging_exporters for ${alias}`,
      JSON.stringify(actual) === JSON.stringify(uiClaim),
      `backend=${JSON.stringify(actual)} ui=${JSON.stringify(uiClaim)}`,
      "identical sets",
    );
  }

  // Check 3: the destination named datadog vs the real datadog config callback
  const ddDest = destRows.find((r) => r.name === "datadog");
  const ddCfg = cfgRows.find((r) => r.name.toLowerCase() === "datadog");
  check("DD1", 'Destination named "datadog" keeps its own lowercase name', ddDest && ddDest.name === "datadog", ddDest ? ddDest.name : "missing", "datadog");
  check("DD2", "Real datadog config callback row also present", !!ddCfg, ddCfg ? ddCfg.name : "missing", 'a config-callback row named Datadog');
  check("DD3", "The two datadog rows are distinguishable (different displayed name)", !!ddDest && !!ddCfg && ddDest.name !== ddCfg.name, `dest="${ddDest && ddDest.name}" cfg="${ddCfg && ddCfg.name}"`, "different strings");
  check("DD4", "datadog config-callback row shows a Mode badge, destination shows a dash", !!ddCfg && ddCfg.mode !== "—" && !!ddDest && ddDest.mode === "—", `cfg mode="${ddCfg && ddCfg.mode}" dest mode="${ddDest && ddDest.mode}"`, "cfg has a mode, dest has —");
  check("DD5", "datadog config-callback Scope is a dash (not a destination)", !!ddCfg && ddCfg.scope === "—", ddCfg ? ddCfg.scope : "missing", "—");
  check("DD6", "datadog config-callback row has no backend sub-line", !!ddCfg && ddCfg.sub === "", ddCfg ? `"${ddCfg.sub}"` : "missing", '""');
  check("DD7", "The two datadog rows have distinct action triggers", !!ddDest && !!ddCfg && ddDest.triggerTestId !== ddCfg.triggerTestId, `${ddDest && ddDest.triggerTestId} vs ${ddCfg && ddCfg.triggerTestId}`, "distinct data-testids");

  // Actions menus (admin)
  const destTrigger = destRows.find((r) => r.name === "uitrk-global").triggerTestId;
  const destItems = await openMenu(page, destTrigger);
  note("adminDestMenu", destItems);
  const destTexts = (destItems || []).map((i) => i.text);
  check("A1", "Destination menu offers Edit scope", destTexts.includes("Edit scope"), JSON.stringify(destTexts), 'includes "Edit scope"');
  check("A2", "Destination menu offers Delete", destTexts.includes("Delete"), JSON.stringify(destTexts), 'includes "Delete"');
  check("A3", "Destination menu offers no Test", !destTexts.includes("Test"), JSON.stringify(destTexts), 'no "Test"');
  check("A4", "Destination menu offers no plain Edit", !destTexts.includes("Edit"), JSON.stringify(destTexts), 'no bare "Edit"');
  check("A5", "Destination menu has exactly 2 items", (destItems || []).length === 2, JSON.stringify(destTexts), "2 items");
  await closeMenu(page);

  const cfgTrigger = ddCfg.triggerTestId;
  const cfgItems = await openMenu(page, cfgTrigger);
  note("adminCfgMenu", cfgItems);
  const cfgTexts = (cfgItems || []).map((i) => i.text);
  check("A6", "Config-callback menu offers Test", cfgTexts.includes("Test"), JSON.stringify(cfgTexts), 'includes "Test"');
  check("A7", "Config-callback menu offers Edit", cfgTexts.includes("Edit"), JSON.stringify(cfgTexts), 'includes "Edit"');
  check("A8", "Config-callback menu offers Delete", cfgTexts.includes("Delete"), JSON.stringify(cfgTexts), 'includes "Delete"');
  check("A9", "Config-callback menu offers no Edit scope", !cfgTexts.includes("Edit scope"), JSON.stringify(cfgTexts), 'no "Edit scope"');
  check("A10", "Config-callback menu has exactly 3 items", (cfgItems || []).length === 3, JSON.stringify(cfgTexts), "3 items");
  await closeMenu(page);

  // Every destination row has a trigger for a full admin
  const allDestHaveTrigger = destRows.every((r) => r.hasTrigger);
  check("A11", "Every destination row exposes an actions trigger for a full admin", allDestHaveTrigger, JSON.stringify(destRows.filter((r) => !r.hasTrigger).map((r) => r.name)), "all have triggers");

  // Menu on a "Not active" destination still offers Edit scope + Delete
  const naTrigger = destRows.find((r) => r.name === "uitrk-adapter-reject").triggerTestId;
  const naItems = await openMenu(page, naTrigger);
  const naTexts = (naItems || []).map((i) => i.text);
  check("A12", 'A "Not active" destination still offers Edit scope + Delete', naTexts.includes("Edit scope") && naTexts.includes("Delete"), JSON.stringify(naTexts), '["Edit scope","Delete"]');
  await closeMenu(page);

  // ---------- Delete dialog ----------
  const delTrigger = destRows.find((r) => r.name === "uitrk-team-and-org").triggerTestId;
  await openMenu(page, delTrigger);
  await page.locator('[data-testid="destination-action-delete"]').click();
  await page.waitForTimeout(900);
  const dlg = await page.evaluate(() => {
    const dialogs = [...document.querySelectorAll('.ant-modal-wrap, [role="dialog"]')].filter((d) => d.offsetParent !== null || d.getBoundingClientRect().height > 0);
    const d = dialogs[dialogs.length - 1];
    return d ? { text: d.innerText, html: d.innerHTML.length } : null;
  });
  note("deleteDialog", dlg);
  await page.screenshot({ path: "/tmp/uitrack_delete_dialog.png", fullPage: true }).catch(() => {});
  const dtext = (dlg && dlg.text) || "";
  check("D1", "Delete dialog is titled for a destination", dtext.includes("Delete Destination"), JSON.stringify(dtext.slice(0, 200)), '"Delete Destination"');
  check("D2", "Delete dialog names the destination", dtext.includes("uitrk-team-and-org"), JSON.stringify(dtext.slice(0, 400)), '"uitrk-team-and-org"');
  check("D3", "Delete dialog names the backend", dtext.includes("Langfuse OTEL"), JSON.stringify(dtext.slice(0, 400)), '"Langfuse OTEL · https://cloud.langfuse.com"');
  check("D4", "Delete dialog invents no Mode value", !/\bMode\b/.test(dtext) && !/\bsuccess\b/.test(dtext), JSON.stringify(dtext.slice(0, 400)), "no Mode row, no invented 'success'");
  check("D5", "Delete dialog warns stored collector credentials go with it", dtext.includes("stored collector credentials are deleted with it"), JSON.stringify(dtext.slice(0, 400)), "collector-credentials warning");
  check("D6", 'Delete dialog uses the "Destination Information" section title', dtext.includes("Destination Information"), JSON.stringify(dtext.slice(0, 400)), '"Destination Information"');
  check("D7", "Delete dialog says the action cannot be undone", dtext.includes("cannot be undone"), JSON.stringify(dtext.slice(0, 400)), '"cannot be undone"');
  // cancel
  const cancelBtn = page.getByRole("button", { name: /^Cancel$/ }).last();
  await cancelBtn.click().catch(async () => {
    await page.keyboard.press("Escape");
  });
  await page.waitForTimeout(800);
  const afterCancel = await scrape(page);
  check("D8", "Cancelling the delete dialog deletes nothing", afterCancel.rows.some((r) => r.name === "uitrk-team-and-org"), `${afterCancel.rows.length} rows`, "destination still present");

  // Delete dialog for a config callback, for contrast
  await openMenu(page, cfgTrigger);
  await page.locator('[data-testid="callback-action-delete"]').click();
  await page.waitForTimeout(900);
  const dlg2 = await page.evaluate(() => {
    const dialogs = [...document.querySelectorAll('.ant-modal-wrap, [role="dialog"]')].filter((d) => d.getBoundingClientRect().height > 0);
    const d = dialogs[dialogs.length - 1];
    return d ? d.innerText : null;
  });
  note("deleteDialogCfg", dlg2);
  check("D9", "Config-callback delete dialog is titled Delete Callback (not Destination)", !!dlg2 && dlg2.includes("Delete Callback") && !dlg2.includes("Delete Destination"), JSON.stringify((dlg2 || "").slice(0, 200)), '"Delete Callback"');
  check("D10", "Config-callback delete dialog does show a Mode", !!dlg2 && /Mode/.test(dlg2), JSON.stringify((dlg2 || "").slice(0, 300)), "Mode row present");
  const dlg2Mode = (dlg2 || "").split("\n").map((s) => s.trim());
  const modeIdx = dlg2Mode.indexOf("Mode");
  const dlg2ModeValue = modeIdx >= 0 ? dlg2Mode[modeIdx + 1] : null;
  check(
    "D11",
    "Config-callback delete dialog's Mode agrees with the Mode the row renders",
    !!dlg2ModeValue && dlg2ModeValue.toLowerCase().replace(/[^a-z]/g, "") === ddCfg.mode.toLowerCase().replace(/[^a-z]/g, ""),
    `dialog Mode="${dlg2ModeValue}" but the row's Mode badge is "${ddCfg.mode}"`,
    "the two agree",
  );
  check("D12", "Config-callback delete dialog names the callback", !!dlg2 && dlg2.includes("datadog"), JSON.stringify((dlg2 || "").slice(0, 300)), '"datadog"');
  await page.getByRole("button", { name: /^Cancel$/ }).last().click().catch(async () => await page.keyboard.press("Escape"));
  await page.waitForTimeout(600);

  // Delete dialog for a "Not active" destination
  const naDelTrigger = destRows.find((r) => r.name === "uitrk-no-description").triggerTestId;
  await openMenu(page, naDelTrigger);
  await page.locator('[data-testid="destination-action-delete"]').click();
  await page.waitForTimeout(900);
  const dlg3 = await page.evaluate(() => {
    const dialogs = [...document.querySelectorAll('.ant-modal-wrap, [role="dialog"]')].filter((d) => d.getBoundingClientRect().height > 0);
    const d = dialogs[dialogs.length - 1];
    return d ? d.innerText : null;
  });
  note("deleteDialogNotActive", dlg3);
  check("D13", 'Delete dialog for a "Not active" destination is still titled for a destination', !!dlg3 && dlg3.includes("Delete Destination"), JSON.stringify((dlg3 || "").slice(0, 200)), '"Delete Destination"');
  check("D14", 'Delete dialog for a destination with no backend renders a Backend row (placeholder "-")', !!dlg3 && /Backend/.test(dlg3), JSON.stringify((dlg3 || "").slice(0, 300)), "Backend row present");
  await page.getByRole("button", { name: /^Cancel$/ }).last().click().catch(async () => await page.keyboard.press("Escape"));
  await page.waitForTimeout(600);

  // "Not active" badge carries an explanation
  const naRow = destRows.find((r) => r.name === "uitrk-adapter-reject");
  check("S1", '"Not active" badge carries an explanatory tooltip', !!naRow && (naRow.scopeTitle || "").includes("cannot be built"), naRow ? JSON.stringify(naRow.scopeTitle) : "row missing", "tooltip explaining it receives no traces");
  const noDescRow = destRows.find((r) => r.name === "uitrk-no-description");
  check("S2", 'A destination with no backend renders a placeholder sub-line rather than a blank/undefined one', !!noDescRow && noDescRow.sub !== "" && !/undefined|null/i.test(noDescRow.sub), noDescRow ? JSON.stringify(noDescRow.sub) : "row missing", "a non-empty, non-undefined placeholder");

  // The +N more collapse hides real grants (documented behavior)
  const t5 = await apiGet(`/team/info?team_id=${teamIdByAlias["uitrk-team-5"]}`);
  const t5names = ((t5.team_info || t5).resolved_logging_exporters || []);
  const manyRow = destRows.find((r) => r.name === "uitrk-many-teams");
  check(
    "S3",
    'The "+N more" collapse hides team-5/6 grants the backend does report (intended collapse, not a wrong verdict)',
    t5names.includes("uitrk-many-teams") && manyRow.scope.includes("+2 more") && !manyRow.scope.includes("uitrk-team-5"),
    `backend for uitrk-team-5=${JSON.stringify(t5names)} ui scope=${JSON.stringify(manyRow.scope)}`,
    "backend grants it, UI collapses the last two behind +2 more",
  );

  // ---------- Edit scope dialog ----------
  const esTrigger = destRows.find((r) => r.name === "uitrk-many-teams").triggerTestId;
  await openMenu(page, esTrigger);
  await page.locator('[data-testid="destination-action-edit-access"]').click();
  await page.waitForTimeout(1200);
  const es = await page.evaluate(() => {
    const m = [...document.querySelectorAll(".ant-modal")].filter((d) => d.getBoundingClientRect().height > 0).pop();
    if (!m) return null;
    return {
      title: m.querySelector(".ant-modal-title") ? m.querySelector(".ant-modal-title").innerText.trim() : null,
      text: m.innerText,
      labels: [...m.querySelectorAll("label")].map((l) => l.innerText.trim()),
      hasSwitch: !!m.querySelector(".ant-switch"),
      selects: m.querySelectorAll(".ant-select").length,
      selectedTags: [...m.querySelectorAll(".ant-select-selection-item")].map((e) => e.innerText.trim()),
    };
  });
  note("editScopeDialog", es);
  await page.screenshot({ path: "/tmp/uitrack_editscope_dialog.png", fullPage: true }).catch(() => {});
  check("E1", "Edit-scope dialog opens", !!es, String(!!es), "true");
  check("E2", "Edit-scope dialog names the destination", !!es && (es.title || "").includes("uitrk-many-teams"), es ? es.title : "no dialog", "Edit scope — uitrk-many-teams");
  check("E3", "Edit-scope dialog offers Global", !!es && es.labels.includes("Global"), es ? JSON.stringify(es.labels) : "no dialog", 'includes "Global"');
  check("E4", "Edit-scope dialog offers Teams", !!es && es.labels.includes("Teams"), es ? JSON.stringify(es.labels) : "no dialog", 'includes "Teams"');
  check("E5", "Edit-scope dialog offers Organizations", !!es && es.labels.includes("Organizations"), es ? JSON.stringify(es.labels) : "no dialog", 'includes "Organizations"');
  check("E6", "Edit-scope dialog has a Global toggle and two multi-selects", !!es && es.hasSwitch && es.selects >= 2, es ? `switch=${es.hasSwitch} selects=${es.selects}` : "no dialog", "switch + 2 selects");
  check("E7", "Edit-scope dialog pre-seeds the destination's current teams", !!es && es.selectedTags.filter((t) => t.startsWith("uitrk-team-")).length === 6, es ? JSON.stringify(es.selectedTags) : "no dialog", "6 team tags pre-selected");
  await page.getByRole("button", { name: /^Cancel$/ }).last().click().catch(async () => await page.keyboard.press("Escape"));
  await page.waitForTimeout(600);

  // ---------- Navigation ----------
  await page.goto(`${BASE}/ui/teams`, { waitUntil: "domcontentloaded" });
  await page.waitForLoadState("networkidle");
  await page.waitForTimeout(1000);
  const onTeams = page.url().includes("/ui/teams");
  check("N3", "Can navigate away to /ui/teams", onTeams, page.url(), "/ui/teams");
  await gotoDestinations(page);
  const back = await scrape(page);
  check("N4", "Destinations page survives navigating away and back", back.rows.filter((r) => r.mode === "—").length === DEST_COUNT, `${back.rows.filter((r) => r.mode === "—").length} destination rows`, String(DEST_COUNT));
  check("N5", "Scope verdicts are stable across the round trip", JSON.stringify(back.rows.map((r) => [r.name, r.scope])) === JSON.stringify(scraped.rows.map((r) => [r.name, r.scope])), "scope cells differ after round trip", "identical");

  const tabClickable = await page.getByText("Logging Callbacks", { exact: true }).first().isVisible().catch(() => false);
  check("N6", "Logging Callbacks tab still exposed after the round trip", tabClickable, String(tabClickable), "true");

  // The legacy query form is not a route for this page
  await page.goto(`${BASE}/ui?page=logging-callbacks`, { waitUntil: "domcontentloaded" });
  await page.waitForLoadState("networkidle");
  await page.waitForTimeout(1500);
  const qHasTable = await page.getByText("Active Logging Callbacks", { exact: true }).first().isVisible().catch(() => false);
  note("queryFormUrl", page.url());
  check("N7", "The query form ?page=logging-callbacks is NOT a route to this page (path route is the only one)", !qHasTable, `?page=logging-callbacks rendered the destinations table: ${qHasTable}`, "does not render the destinations table");
  await gotoDestinations(page);

  const addBtn = await page.getByRole("button", { name: "Add Callback" }).first().isVisible().catch(() => false);
  check("A13", 'Full admin sees the "Add Callback" button', addBtn, String(addBtn), "true");

  note("adminConsoleErrors", consoleErrors.slice(0, 30));
  check("C1", "No uncaught page errors while rendering the destinations table as admin", !consoleErrors.some((e) => e.startsWith("PAGEERROR")), JSON.stringify(consoleErrors.filter((e) => e.startsWith("PAGEERROR")).slice(0, 5)), "none");

  await ctx.close();

  // ---------- READ-ONLY ADMIN ----------
  consoleErrors = [];
  const ctx2 = await browser.newContext({ viewport: { width: 1600, height: 1200 } });
  const p2 = await ctx2.newPage();
  p2.on("console", (m) => {
    if (m.type() === "error") consoleErrors.push(m.text().slice(0, 300));
  });
  p2.on("pageerror", (e) => consoleErrors.push("PAGEERROR: " + String(e).slice(0, 300)));

  await login(p2, VIEWER_USER, VIEWER_PASS);
  note("viewerUrlAfterLogin", p2.url());
  await gotoDestinations(p2);
  const vScraped = await scrape(p2);
  note("viewerScrape", vScraped);
  await p2.screenshot({ path: "/tmp/uitrack_viewer_table.png", fullPage: true }).catch(() => {});

  const vDest = vScraped.rows.filter((r) => r.mode === "—");
  const vCfg = vScraped.rows.filter((r) => r.mode !== "—");
  check("V1", "Read-only admin can load the destinations page", p2.url().includes("/ui/logging-and-alerts"), p2.url(), "/ui/logging-and-alerts");
  check("V2", "Read-only admin sees all 10 destinations", vDest.length === DEST_COUNT, `${vDest.length}: ${vDest.map((r) => r.name).join(",")}`, String(DEST_COUNT));
  check(
    "V3",
    "Read-only admin sees identical Scope verdicts to the full admin",
    JSON.stringify(vDest.map((r) => [r.name, r.scope])) === JSON.stringify(destRows.map((r) => [r.name, r.scope])),
    JSON.stringify(vDest.map((r) => [r.name, r.scope])),
    JSON.stringify(destRows.map((r) => [r.name, r.scope])),
  );
  const vAdd = await p2.getByRole("button", { name: "Add Callback" }).first().isVisible().catch(() => false);
  check("V4", 'Read-only admin has no "Add Callback" button', !vAdd, String(vAdd), "false");
  const vDestWithTrigger = vDest.filter((r) => r.hasTrigger).map((r) => r.name);
  check("V5", "Read-only admin gets NO actions trigger on any destination row", vDestWithTrigger.length === 0, JSON.stringify(vDestWithTrigger), "[]");
  const vCfgRow = vCfg.find((r) => r.name.toLowerCase() === "datadog");
  check("V6", "Read-only admin keeps the actions trigger on the config-callback row", !!vCfgRow && vCfgRow.hasTrigger, vCfgRow ? String(vCfgRow.hasTrigger) : "config row missing", "true");
  if (vCfgRow && vCfgRow.hasTrigger) {
    const vItems = await openMenu(p2, vCfgRow.triggerTestId);
    note("viewerCfgMenu", vItems);
    const vTexts = (vItems || []).map((i) => i.text);
    check("V7", "Read-only admin's config-callback menu offers Test", vTexts.includes("Test"), JSON.stringify(vTexts), '["Test"]');
    check("V8", "Read-only admin's config-callback menu has no Edit", !vTexts.includes("Edit"), JSON.stringify(vTexts), 'no "Edit"');
    check("V9", "Read-only admin's config-callback menu has no Delete", !vTexts.includes("Delete"), JSON.stringify(vTexts), 'no "Delete"');
    check("V10", "Read-only admin's config-callback menu has exactly 1 item", (vItems || []).length === 1, JSON.stringify(vTexts), "1");
    await closeMenu(p2);
  } else {
    check("V7", "Read-only admin's config-callback menu offers Test", false, "no trigger", "menu with Test");
    check("V8", "Read-only admin's config-callback menu has no Edit", false, "no trigger", "no Edit");
    check("V9", "Read-only admin's config-callback menu has no Delete", false, "no trigger", "no Delete");
    check("V10", "Read-only admin's config-callback menu has exactly 1 item", false, "no trigger", "1");
  }
  check("V11", "Read-only admin sees the destination named datadog under its own name", vDest.some((r) => r.name === "datadog"), JSON.stringify(vDest.map((r) => r.name)), 'includes "datadog"');
  check("V12", "Read-only admin sees the same Not-active verdicts", JSON.stringify(vDest.filter((r) => r.scope.includes("Not active")).map((r) => r.name).sort()) === JSON.stringify(["uitrk-adapter-reject", "uitrk-no-description"]), JSON.stringify(vDest.filter((r) => r.scope.includes("Not active")).map((r) => r.name).sort()), '["uitrk-adapter-reject","uitrk-no-description"]');
  note("viewerConsoleErrors", consoleErrors.slice(0, 30));
  check("C2", "No uncaught page errors for the read-only admin", !consoleErrors.some((e) => e.startsWith("PAGEERROR")), JSON.stringify(consoleErrors.filter((e) => e.startsWith("PAGEERROR")).slice(0, 5)), "none");

  await ctx2.close();
  await browser.close();

  fs.writeFileSync(OUT, JSON.stringify({ results, notes }, null, 2));
  const pass = results.filter((r) => r.status === "PASS").length;
  const fail = results.filter((r) => r.status === "FAIL").length;
  console.log(`TOTAL ${results.length}  PASS ${pass}  FAIL ${fail}`);
  for (const r of results) {
    if (r.status === "FAIL") console.log(`FAIL ${r.id}: ${r.desc}\n   actual:   ${r.actual}\n   expected: ${r.expected}`);
  }
})().catch((e) => {
  console.error("RUNNER CRASH", e);
  fs.writeFileSync(OUT, JSON.stringify({ results, notes, crash: String(e && e.stack) }, null, 2));
  process.exit(1);
});
