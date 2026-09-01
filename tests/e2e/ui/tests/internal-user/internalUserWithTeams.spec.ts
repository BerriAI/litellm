import { test, expect } from "@playwright/test";
import {
  INTERNAL_USER_STORAGE_PATH,
  E2E_TEAM_CRUD_ALIAS,
  E2E_TEAM_KEYGEN_ALIAS,
  E2E_TEAM_ORG_ALIAS,
} from "../../constants";
import { Page } from "../../fixtures/pages";
import { navigateToPage } from "../../helpers/navigation";

/**
 * Differential partner to internalUserNoTeam.spec.ts: the seeded
 * e2e-internal-user belongs to exactly three teams, so the Create Key dropdown
 * must list all of them. Without this, the no-team spec's "zero options" assertion
 * would still pass against a bug that empties the dropdown for everyone.
 */
test.describe("Internal User with team memberships", () => {
  test.use({ storageState: INTERNAL_USER_STORAGE_PATH });

  test("Create Key team dropdown lists exactly the teams the user belongs to", async ({ page }) => {
    await navigateToPage(page, Page.ApiKeys);

    await page.getByRole("button", { name: /Create New Key/i }).click();
    await expect(page.getByText("Key Ownership")).toBeVisible({ timeout: 10_000 });

    const teamSelect = page.getByTestId("team-dropdown").getByRole("combobox");
    await teamSelect.click();

    // All seeded memberships render, and nothing else does — proving the
    // dropdown is scoped to the user's teams rather than empty or unfiltered.
    await expect(page.getByRole("option", { name: E2E_TEAM_CRUD_ALIAS })).toBeVisible({ timeout: 10_000 });
    await expect(page.getByRole("option", { name: E2E_TEAM_ORG_ALIAS })).toBeVisible();
    await expect(page.getByRole("option", { name: E2E_TEAM_KEYGEN_ALIAS })).toBeVisible();
    await expect(page.getByRole("option")).toHaveCount(3);
  });
});
