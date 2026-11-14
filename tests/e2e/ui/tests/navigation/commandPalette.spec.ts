import { test, expect, type Page as PlaywrightPage } from "@playwright/test";
import { ADMIN_STORAGE_PATH } from "../../constants";
import { Page } from "../../fixtures/pages";
import { dismissFeedbackPopup, navigateToPage } from "../../helpers/navigation";
import { createVirtualKey, deleteVirtualKey, uniqueSuffix } from "../../helpers/traffic";

const paletteSearch = (page: PlaywrightPage) => page.getByRole("combobox", { name: "Search" });
const headerTrigger = (page: PlaywrightPage) => page.getByRole("button", { name: "Search keys…", exact: true });

test.describe("Command palette", () => {
  test.use({ storageState: ADMIN_STORAGE_PATH });

  test.beforeEach(async ({ page }) => {
    await navigateToPage(page, Page.ApiKeys);
    await dismissFeedbackPopup(page);
    await expect(headerTrigger(page)).toBeVisible();
  });

  test("opens a key found by its alias", async ({ page }) => {
    const alias = `e2e-palette-key-${uniqueSuffix()}`;
    const created = await createVirtualKey(page.request, { key_alias: alias });
    try {
      await page.keyboard.press("ControlOrMeta+K");
      await paletteSearch(page).fill(alias);
      await expect(page.getByRole("option", { name: new RegExp(`^${alias}`) })).toBeVisible();
      await page.keyboard.press("Enter");

      await expect(page).toHaveURL(new RegExp(`[?&]key=${created.token}`));
      await expect(paletteSearch(page)).toBeHidden();
    } finally {
      await deleteVirtualKey(page.request, created.token);
    }
  });

  test("does not open over the Create New Key dialog", async ({ page }) => {
    await page.getByRole("button", { name: /Create New Key/i }).click();
    const keyName = page.getByLabel(/Key Name/);
    await keyName.click();

    await page.keyboard.press("ControlOrMeta+K");
    await page.keyboard.type("still-in-the-dialog");

    await expect(keyName).toHaveValue("still-in-the-dialog");
    await expect(paletteSearch(page)).toBeHidden();
  });

  test("opens from the header trigger and closes with Escape", async ({ page }) => {
    await headerTrigger(page).click();
    await expect(paletteSearch(page)).toBeFocused();

    await page.keyboard.press("Escape");
    await expect(paletteSearch(page)).toBeHidden();
  });
});
