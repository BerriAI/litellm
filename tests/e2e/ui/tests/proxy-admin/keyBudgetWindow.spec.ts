import { test as base, expect } from "@playwright/test";
import { ADMIN_STORAGE_PATH, E2E_TEAM_CRUD_ID } from "../../constants";
import { Page } from "../../fixtures/pages";
import { dismissFeedbackPopup, navigateToPage, openKeyDetail } from "../../helpers/navigation";
import { captureRequestBody } from "../../helpers/roundTrip";
import { CHAT_MODEL_A, createVirtualKey, deleteVirtualKey, readKeyInfo, uniqueSuffix } from "../../helpers/traffic";

interface ScopedKey {
  alias: string;
  token: string;
}

const test = base.extend<{ scopedKey: ScopedKey }>({
  scopedKey: async ({ page }, use) => {
    const alias = `e2e-budget-window-${uniqueSuffix()}`;
    const created = await createVirtualKey(page.request, {
      key_alias: alias,
      team_id: E2E_TEAM_CRUD_ID,
      models: [CHAT_MODEL_A],
    });
    await use({ alias, token: created.token });
    await deleteVirtualKey(page.request, created.token);
  },
});

test.describe("Proxy Admin - Key budget window", () => {
  test.use({ storageState: ADMIN_STORAGE_PATH });

  test("a monthly spend cap survives a reload, and clearing the window keeps the cap", async ({ page, scopedKey }) => {
    const { alias, token } = scopedKey;

    const before = await readKeyInfo(page.request, token);
    expect(before.max_budget, "a freshly generated key starts with no budget").toBeNull();

    await navigateToPage(page, Page.ApiKeys);
    await dismissFeedbackPopup(page);
    await openKeyDetail(page, alias);

    await page.getByRole("tab", { name: "Settings" }).click();
    await page.getByRole("button", { name: "Edit Settings" }).click();

    await page.getByRole("spinbutton", { name: "Max Budget (USD)" }).fill("12.5");
    await page.getByLabel("Reset Budget", { exact: true }).click();
    await page.getByRole("option", { name: "monthly", exact: true }).click();
    await page.getByRole("button", { name: "Save Changes" }).click();

    await expect
      .poll(async () => (await readKeyInfo(page.request, token)).max_budget, {
        message: "the $12.50 cap never reached /key/info",
        timeout: 20_000,
      })
      .toBe(12.5);
    await expect
      .poll(async () => (await readKeyInfo(page.request, token)).budget_duration, {
        message: "the monthly reset window never reached /key/info",
        timeout: 20_000,
      })
      .toBe("30d");

    const capped = await readKeyInfo(page.request, token);
    const resetAt = new Date(capped.budget_reset_at ?? "");
    expect(Number.isNaN(resetAt.getTime()), "a monthly window left the key with no budget_reset_at").toBe(false);
    expect(resetAt.getTime(), "budget_reset_at was set in the past").toBeGreaterThan(Date.now());
    expect(resetAt.getUTCDate(), "a monthly window resets on the 1st, a daily one would not").toBe(1);

    await page.reload();
    await expect(
      page.getByRole("paragraph").filter({ hasText: "of $12.50" }),
      "the reloaded key detail does not render the $12.50 cap",
    ).toBeVisible({ timeout: 15_000 });

    await page.getByRole("tab", { name: "Settings" }).click();
    await expect(
      page.getByTestId("budget-reset-value"),
      "the reloaded key detail does not name the 30d reset window",
    ).toHaveText(/Every 30d/, { timeout: 15_000 });

    await page.getByRole("button", { name: "Edit Settings" }).click();
    await page.getByLabel("Reset Budget", { exact: true }).click();
    await page.getByRole("option", { name: "Never resets", exact: true }).click();

    const cleared = await captureRequestBody(page, { method: "POST", urlIncludes: "/key/update" }, async () => {
      await page.getByRole("button", { name: "Save Changes" }).click();
    });
    expect(cleared).toHaveProperty("budget_duration");
    expect(cleared.budget_duration, "clearing the window must send budget_duration: null explicitly").toBeNull();

    await expect
      .poll(async () => (await readKeyInfo(page.request, token)).budget_duration, {
        message: "the reset window was never cleared on /key/info",
        timeout: 20_000,
      })
      .toBeNull();

    const after = await readKeyInfo(page.request, token);
    expect(after.budget_reset_at, "clearing the reset window left a stale next-reset timestamp").toBeNull();
    expect(after.max_budget, "clearing the reset window also wiped the spend cap").toBe(12.5);
    expect(after.models, "editing the budget left the key's models untouched").toEqual(before.models);
    expect(after.team_id, "editing the budget left the key's team untouched").toEqual(before.team_id);
  });
});
