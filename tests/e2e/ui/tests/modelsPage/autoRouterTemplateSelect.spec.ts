import { expect, test, type Locator, type Page as PlaywrightPage } from "@playwright/test";
import { ADMIN_STORAGE_PATH } from "../../constants";
import { navigateToPage } from "../../helpers/navigation";
import { Page } from "../../fixtures/pages";

/**
 * Opens Add Auto Router and returns the Template select's trigger, which is the
 * shallowest real page that renders SelectContent with tall multi-line options.
 */
async function openTemplateSelect(page: PlaywrightPage) {
  await navigateToPage(page, Page.Models);
  await page.getByRole("tab", { name: "Auto-Routers" }).click();
  await page.getByRole("button", { name: "Add Auto Router" }).click();

  const trigger = page.getByTestId("template-selector");
  await expect(trigger).toBeVisible();
  return trigger;
}

async function boxes(trigger: Locator, options: Locator) {
  const triggerBox = await trigger.boundingBox();
  const optionsBox = await options.boundingBox();
  return triggerBox && optionsBox ? { triggerBox, optionsBox } : null;
}

function pollOptionsOpenBelowTrigger(trigger: Locator, options: Locator) {
  return expect.poll(async () => {
    const box = await boxes(trigger, options);
    return box && box.optionsBox.y >= box.triggerBox.y + box.triggerBox.height;
  });
}

function pollOptionsCoverTrigger(trigger: Locator, options: Locator) {
  return expect.poll(async () => {
    const box = await boxes(trigger, options);
    return (
      box &&
      box.optionsBox.y < box.triggerBox.y + box.triggerBox.height &&
      box.optionsBox.y + box.optionsBox.height > box.triggerBox.y
    );
  });
}

test.describe("Auto Router template select anchoring", () => {
  test.use({ storageState: ADMIN_STORAGE_PATH });

  test("opens the options below the trigger when there is room below it", async ({ page }) => {
    await page.setViewportSize({ width: 1280, height: 900 });
    const trigger = await openTemplateSelect(page);
    await trigger.scrollIntoViewIfNeeded();

    await trigger.click();
    const options = page.getByRole("listbox");
    await expect(options).toBeVisible();

    await pollOptionsOpenBelowTrigger(trigger, options).toBe(true);
  });

  test("keeps the trigger uncovered when the options open with no room below it", async ({ page }) => {
    await page.setViewportSize({ width: 1280, height: 560 });
    const trigger = await openTemplateSelect(page);
    await trigger.scrollIntoViewIfNeeded();

    await trigger.click();
    const options = page.getByRole("listbox");
    await expect(options).toBeVisible();

    await pollOptionsCoverTrigger(trigger, options).toBe(false);
  });
});
