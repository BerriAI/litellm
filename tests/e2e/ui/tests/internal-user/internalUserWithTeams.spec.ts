import { test, expect } from "@playwright/test";
import {
  INTERNAL_USER_STORAGE_PATH,
  E2E_TEAM_KEYGEN_ALIAS,
} from "../../constants";
import { Page } from "../../fixtures/pages";
import { navigateToPage } from "../../helpers/navigation";

test.describe("Internal User with team memberships", () => {
  test.use({ storageState: INTERNAL_USER_STORAGE_PATH });

  test("Create Key team dropdown lists only the teams where the user can create keys", async ({ page }) => {
    await navigateToPage(page, Page.ApiKeys);

    await page.getByRole("button", { name: /Create New Key/i }).click();
    await expect(page.getByText("Key Ownership")).toBeVisible({ timeout: 10_000 });

    const teamSelect = page.getByTestId("team-dropdown").getByRole("combobox");
    await teamSelect.click();

    await expect(page.getByRole("option", { name: E2E_TEAM_KEYGEN_ALIAS })).toBeVisible({ timeout: 10_000 });
    await expect(page.getByRole("option")).toHaveCount(1);
  });
});
