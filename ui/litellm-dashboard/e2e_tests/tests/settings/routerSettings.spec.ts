import { test, expect } from "@playwright/test";
import { ADMIN_STORAGE_PATH } from "../../constants";
import { navigateToPage } from "../../helpers/navigation";
import { Page } from "../../fixtures/pages";
import { Role, users } from "../../fixtures/users";

const PRIMARY = "fake-openai-gpt-4";
const FALLBACK = "fake-anthropic-claude";

/**
 * Wipe any fallbacks for the primary model so the test is idempotent across
 * retries and local reruns (the proxy persists router_settings to the DB).
 */
async function clearFallbackForPrimary(request: import("@playwright/test").APIRequestContext) {
  const masterKey = users[Role.ProxyAdmin].password;
  const auth = { Authorization: `Bearer ${masterKey}` };

  const current = await request.get("http://localhost:4000/get/config/callbacks", { headers: auth });
  if (!current.ok()) return;
  const body = await current.json();
  const router = body?.router_settings ?? {};
  const existing: Array<Record<string, string[]>> = Array.isArray(router.fallbacks) ? router.fallbacks : [];
  const next = existing.filter((entry) => !(entry && PRIMARY in entry));
  if (next.length === existing.length) return;

  await request.post("http://localhost:4000/config/update", {
    headers: auth,
    data: { router_settings: { ...router, fallbacks: next } },
  });
}

test.describe("Router Settings - Fallbacks", () => {
  test.use({ storageState: ADMIN_STORAGE_PATH });

  test.beforeEach(async ({ request }) => {
    await clearFallbackForPrimary(request);
  });

  test.afterEach(async ({ request }) => {
    await clearFallbackForPrimary(request);
  });

  test("Add a fallback and verify it appears in the table", async ({ page }) => {
    await navigateToPage(page, Page.RouterSettings);

    // Four tabs: Loadbalancing / Routing Groups / Fallbacks / General — click Fallbacks
    await page.getByRole("tab", { name: "Fallbacks" }).click();

    // The model options come from /model_group/info, which AddFallbacks
    // fires only after the modal mounts. Wait for that response so the
    // dropdown is populated before we try to pick from it — without this
    // the test races on CI (local SLOWMO masks the gap).
    const modelsLoaded = page.waitForResponse(
      (res) => res.url().includes("/model_group/info") && res.status() === 200,
      { timeout: 15_000 },
    );
    await page.getByRole("button", { name: /Add Fallbacks/i }).click();
    await modelsLoaded;

    const modal = page.locator(".ant-modal:visible");
    await expect(modal).toBeVisible({ timeout: 5_000 });

    // FallbackGroupConfig.tsx renders both selects with `showSearch`. The
    // most stable interaction is: click to open + focus, type the model name to
    // narrow the listbox to a single highlighted option, then press Enter.
    // Verify each selection landed by watching the dialog's own state transition
    // (the tab title updates to the picked primary; the fallback chain list
    // populates) rather than by asserting on the dropdown popup, which sits in
    // a custom getPopupContainer and is awkward to scope reliably.
    const primarySelect = modal.locator(".ant-select").filter({ hasText: "Select primary model" });
    await primarySelect.click();
    await page.keyboard.type(PRIMARY);
    await page.keyboard.press("Enter");
    await expect(modal.getByRole("tab", { name: PRIMARY })).toBeVisible({ timeout: 10_000 });

    const fallbackSelect = modal.locator(".ant-select").filter({ hasText: "Select fallback models" });
    await fallbackSelect.click();
    await page.keyboard.type(FALLBACK);
    await page.keyboard.press("Enter");
    await page.keyboard.press("Escape");
    // The Fallback Chain helper text reads "(N/10 used)"; once it ticks to 1 the
    // selection has been recorded.
    await expect(modal.getByText("(1/10 used)")).toBeVisible({ timeout: 10_000 });

    // Save
    await modal.getByRole("button", { name: /Save All Configurations/i }).click();

    // Success toast
    await expect(page.getByText(/fallback configuration\(s\) added successfully/i).first())
      .toBeVisible({ timeout: 10_000 });

    // Modal closes, and a single row contains BOTH the primary and the fallback
    // model — stronger than asserting each name appears somewhere in tbody,
    // which could be satisfied by leftover rows from prior runs.
    await expect(modal).not.toBeVisible({ timeout: 5_000 });

    const newRow = page.locator("table tbody tr")
      .filter({ hasText: PRIMARY })
      .filter({ hasText: FALLBACK });
    await expect(newRow).toHaveCount(1, { timeout: 10_000 });
  });
});
