import { expect, test, type Locator, type Page as PlaywrightPage } from "@playwright/test";
import { ADMIN_STORAGE_PATH } from "../../constants";
import { Page } from "../../fixtures/pages";
import { navigateToPage } from "../../helpers/navigation";
import { DECISIONS_MODEL } from "../../helpers/traffic";

const centerY = async (locator: Locator): Promise<number> => {
  const box = await locator.boundingBox();
  expect(box, "element has no layout box").not.toBeNull();
  return box!.y + box!.height / 2;
};

const offCenterBy = async (inner: Locator, outer: Locator): Promise<number> =>
  Math.abs((await centerY(inner)) - (await centerY(outer)));

const dismissSearchHint = async (page: PlaywrightPage): Promise<void> => {
  const dismiss = page.getByRole("button", { name: "Dismiss search hint" });
  if (await dismiss.isVisible()) {
    await dismiss.click();
  }
};

test.describe("Decisions playground Response header", () => {
  test.use({ storageState: ADMIN_STORAGE_PATH, viewport: { width: 1440, height: 900 } });

  test("keeps the title and its badges on one row centered in the header", async ({ page }) => {
    await navigateToPage(page, Page.LlmPlayground);
    await page.getByRole("tab", { name: /Decisions/ }).click();

    const header = page.getByRole("region", { name: "Decisions results" }).locator('[data-slot="card-header"]');
    const title = header.getByText("Response", { exact: true });
    await expect(title).toBeVisible({ timeout: 20_000 });
    expect(await offCenterBy(title, header), "empty-state title is off center").toBeLessThanOrEqual(1);

    await page.getByRole("button", { name: "Load example" }).click();
    await page.getByRole("menuitem", { name: "Moderate a forum comment" }).click();
    const modelPicker = page.getByRole("combobox", { name: "Decision model" });
    await modelPicker.click();
    await modelPicker.fill(DECISIONS_MODEL);
    await page.getByRole("option", { name: DECISIONS_MODEL, exact: true }).click();
    await dismissSearchHint(page);
    await page.getByRole("button", { name: "Send", exact: true }).click();

    const usage = header.getByText("12 in / 3 out");
    await expect(usage).toBeVisible({ timeout: 30_000 });
    expect(await offCenterBy(title, header), "title is off center after Send").toBeLessThanOrEqual(1);
    for (const badge of [header.getByText(/^\d+ ms$/), usage]) {
      expect(await offCenterBy(badge, title), "badge wrapped off the title's row").toBeLessThanOrEqual(1);
    }
  });
});
