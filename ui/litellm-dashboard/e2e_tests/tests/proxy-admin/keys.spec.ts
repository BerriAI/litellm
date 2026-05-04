import { test, expect } from "@playwright/test";
import {
  ADMIN_STORAGE_PATH,
  E2E_DELETE_KEY_ALIAS,
  E2E_REGENERATE_KEY_ALIAS,
  E2E_UPDATE_LIMITS_KEY_ALIAS,
  E2E_INTERNAL_USER_KEY_ALIAS,
  E2E_TEAM_CRUD_ALIAS,
} from "../../constants";
import { Page } from "../../fixtures/pages";
import { navigateToPage, dismissFeedbackPopup } from "../../helpers/navigation";

test.describe("Proxy Admin - Keys", () => {
  test.use({ storageState: ADMIN_STORAGE_PATH });

  test("Create a key in a team", async ({ page }) => {
    await navigateToPage(page, Page.ApiKeys);
    await dismissFeedbackPopup(page);

    // Click "+ Create New Key" button
    await page.getByRole("button", { name: /Create New Key/i }).click();

    // Wait for the key creation modal
    await expect(page.getByText("Key Ownership")).toBeVisible({ timeout: 10_000 });

    // Fill key name (has data-testid="base-input" in the built UI)
    const keyName = `e2e-admin-key-${Date.now()}`;
    await page.getByTestId("base-input").fill(keyName);

    // Select team — the team dropdown has placeholder "Search or select a team"
    const teamSelect = page.locator(".ant-select", { hasText: "Search or select a team" });
    await teamSelect.click();
    await page.keyboard.type(E2E_TEAM_CRUD_ALIAS);
    await page.locator(".ant-select-dropdown:visible").getByText(E2E_TEAM_CRUD_ALIAS).first().click();

    // Select models
    await page.locator(".ant-select-selection-overflow").click();
    await page.locator(".ant-select-dropdown:visible").getByText("All Team Models").click();
    await page.keyboard.press("Escape");

    // Submit
    await page.getByRole("button", { name: "Create Key", exact: true }).click();

    // Success shows "Save your Key" in a second dialog
    await expect(page.getByText("Save your Key")).toBeVisible({ timeout: 10_000 });
    await page.keyboard.press("Escape");

    // Verify the new key appears in the table
    await expect(page.getByText(keyName)).toBeVisible({ timeout: 10_000 });
  });

  test("Regenerate key", async ({ page }) => {
    await navigateToPage(page, Page.ApiKeys);
    await dismissFeedbackPopup(page);

    // Key IDs are rendered as buttons in the table
    const keyRow = page.locator("tr", { hasText: E2E_REGENERATE_KEY_ALIAS });
    await expect(keyRow).toBeVisible({ timeout: 10_000 });
    await keyRow.locator("button").first().click();

    await expect(page.getByText("Back to Keys")).toBeVisible({ timeout: 10_000 });

    await page.getByRole("button", { name: "Regenerate Key" }).click();

    // Scope to the modal — the Regenerate button has an icon whose aria-label
    // ("sync") is concatenated into the button's accessible name, and the
    // "Regenerate Key" button is still in the DOM behind the modal.
    const modal = page.locator(".ant-modal:visible");
    await modal.getByRole("button", { name: /Regenerate/ }).click();

    // Success view shows a Copy button in the footer (text varies between modal versions)
    await expect(modal.getByRole("button", { name: /Copy.*Key/ })).toBeVisible({ timeout: 20_000 });
  });

  test("Update key TPM and RPM limits", async ({ page }) => {
    await navigateToPage(page, Page.ApiKeys);
    await dismissFeedbackPopup(page);

    const keyRow = page.locator("tr", { hasText: E2E_UPDATE_LIMITS_KEY_ALIAS });
    await expect(keyRow).toBeVisible({ timeout: 10_000 });
    await keyRow.locator("button").first().click();

    await expect(page.getByText("Back to Keys")).toBeVisible({ timeout: 10_000 });

    await page.getByRole("tab", { name: "Settings" }).click();
    await page.getByRole("button", { name: "Edit Settings" }).click();

    await page.getByRole("spinbutton", { name: "TPM Limit" }).fill("123");
    await page.getByRole("spinbutton", { name: "RPM Limit" }).fill("456");
    await page.getByRole("button", { name: "Save Changes" }).click();

    await expect(
      page.getByRole("paragraph").filter({ hasText: "TPM: 123" })
    ).toBeVisible({ timeout: 10_000 });
    await expect(
      page.getByRole("paragraph").filter({ hasText: "RPM: 456" })
    ).toBeVisible({ timeout: 10_000 });
  });

  test("Delete key", async ({ page }) => {
    await navigateToPage(page, Page.ApiKeys);
    await dismissFeedbackPopup(page);

    const keyRow = page.locator("tr", { hasText: E2E_DELETE_KEY_ALIAS });
    await expect(keyRow).toBeVisible({ timeout: 10_000 });
    await keyRow.locator("button").first().click();

    await expect(page.getByText("Back to Keys")).toBeVisible({ timeout: 10_000 });

    await page.getByRole("button", { name: "Delete Key" }).click();

    const modal = page.locator(".ant-modal:visible");
    await expect(modal).toBeVisible({ timeout: 5_000 });
    await modal.locator("input").fill(E2E_DELETE_KEY_ALIAS);

    const deleteButton = modal.getByRole("button", { name: "Delete", exact: true });
    await expect(deleteButton).toBeEnabled();
    await deleteButton.click();

    await expect(page.getByText(/Key deleted/i).first()).toBeVisible({ timeout: 10_000 });
  });

  test("See internal user keys in team", async ({ page }) => {
    await navigateToPage(page, Page.ApiKeys);
    await dismissFeedbackPopup(page);

    await expect(page.getByText(E2E_INTERNAL_USER_KEY_ALIAS)).toBeVisible({ timeout: 10_000 });
  });
});
