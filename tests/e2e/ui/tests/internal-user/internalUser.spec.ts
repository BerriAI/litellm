import { test, expect } from "@playwright/test";
import {
  E2E_INTERNAL_USER_KEY_ALIAS,
  E2E_TEAM_CRUD_ALIAS,
  E2E_TEAM_CRUD_ID,
  E2E_TEAM_KEYGEN_ALIAS,
  INTERNAL_USER_STORAGE_PATH,
} from "../../constants";
import { Page } from "../../fixtures/pages";
import { navigateToPage, clickTeamId } from "../../helpers/navigation";
import { CHAT_MODEL_A, MOCK_RESPONSE_TEXT, masterKey } from "../../helpers/traffic";
import { keySourceSelect, onlyVisible, openPlayground, selectModel, sendMessage } from "../../helpers/playground";

test.describe("Internal User", () => {
  test.use({ storageState: INTERNAL_USER_STORAGE_PATH });

  test("Create Key modal shows the team dropdown populated with the user's teams", async ({ page }) => {
    await navigateToPage(page, Page.ApiKeys);

    await page.getByRole("button", { name: /Create New Key/i }).click();
    await expect(page.getByText("Key Ownership")).toBeVisible({ timeout: 10_000 });

    // Open the team dropdown — seeded internal user is a member of
    // e2e-team-crud and e2e-team-org, so we expect at least the CRUD alias.
    const teamSelect = page.getByTestId("team-dropdown").getByRole("combobox");
    await teamSelect.click();
    await page.keyboard.type(E2E_TEAM_CRUD_ALIAS);
    await expect(page.getByRole("option", { name: E2E_TEAM_CRUD_ALIAS }).first()).toBeVisible({ timeout: 5_000 });
  });

  test("Team info page omits the Settings tab for non-admin members", async ({ page }) => {
    await navigateToPage(page, Page.Teams);

    await clickTeamId(page, E2E_TEAM_CRUD_ID);

    // Overview / My User / Virtual Keys are always visible; Settings is gated
    // on canEditTeam and must NOT render for a regular team member.
    await expect(page.getByRole("tab", { name: "Overview" })).toBeVisible({ timeout: 5_000 });
    await expect(page.getByRole("tab", { name: "Settings" })).not.toBeVisible();
    await expect(page.getByRole("tab", { name: "Members" })).not.toBeVisible();
  });

  test("Internal user creates a team key and uses it in the Playground", async ({ page, request }) => {
    const suffix = Date.now();
    const auth = { Authorization: `Bearer ${masterKey()}` };

    let apiKey = "";
    try {
      await navigateToPage(page, Page.ApiKeys);

      await page.getByRole("button", { name: /Create New Key/i }).click();
      await expect(page.getByText("Key Ownership")).toBeVisible({ timeout: 10_000 });

      await expect(page.getByRole("radio", { name: "You", exact: true })).toBeVisible({ timeout: 10_000 });
      await expect(page.getByRole("radio", { name: "Another User" })).toHaveCount(0);

      const keyName = `e2e-internal-team-key-${suffix}`;
      await page.getByLabel(/Key Name/).fill(keyName);

      const teamSelect = page.getByTestId("team-dropdown").getByRole("combobox");
      await teamSelect.click();
      await page.keyboard.type(E2E_TEAM_KEYGEN_ALIAS);
      await page.locator('[data-slot="combobox-content"]:visible').getByText(E2E_TEAM_KEYGEN_ALIAS).first().click();

      await page.getByRole("combobox", { name: "Select models" }).click();
      await page.getByRole("option", { name: "All Team Models", exact: true }).click();
      await page.keyboard.press("Escape");

      await page.getByRole("button", { name: "Create Key", exact: true }).click();

      await expect(page.getByText("Save your Key")).toBeVisible({ timeout: 10_000 });
      apiKey = (await page.getByRole("dialog", { name: "Save your Key" }).locator("pre").innerText()).trim();
      expect(apiKey).toMatch(/^sk-/);
      await page.keyboard.press("Escape");

      await openPlayground(page);
      await keySourceSelect(page).click();
      await onlyVisible(page.getByRole("option", { name: "Virtual Key" })).click({ timeout: 15_000 });

      const keyInput = onlyVisible(page.getByPlaceholder("Enter custom Virtual Key"));
      await expect(keyInput).toBeVisible({ timeout: 10_000 });
      await keyInput.fill(apiKey);

      await selectModel(page, CHAT_MODEL_A);
      await sendMessage(page, `internal user team key ping ${keyName}`);

      await expect(page.getByText(MOCK_RESPONSE_TEXT, { exact: false }).first()).toBeVisible({ timeout: 60_000 });
    } finally {
      if (apiKey) {
        await request.post("/key/delete", { headers: auth, data: { keys: [apiKey] } });
      }
    }
  });

  test("Virtual Keys page does not surface litellm-dashboard team keys", async ({ page }) => {
    await navigateToPage(page, Page.ApiKeys);

    // Anchor on the user's own seeded key so the absence check below cannot
    // pass vacuously against an empty table.
    await expect(page.getByRole("row").filter({ hasText: E2E_INTERNAL_USER_KEY_ALIAS }).first()).toBeVisible({
      timeout: 10_000,
    });

    // The litellm-dashboard team is the proxy's internal bookkeeping team —
    // its keys must never leak into an internal user's Virtual Keys table.
    await expect(page.getByRole("row").filter({ hasText: "litellm-dashboard" })).toHaveCount(0);
  });
});
