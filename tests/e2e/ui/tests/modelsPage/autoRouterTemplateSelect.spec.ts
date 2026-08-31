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

function pollOptionsCoverTrigger(trigger: Locator, options: Locator) {
  return expect.poll(async () => {
    const triggerBox = await trigger.boundingBox();
    const optionsBox = await options.boundingBox();
    if (!triggerBox || !optionsBox) return null;
    return optionsBox.y < triggerBox.y + triggerBox.height && optionsBox.y + optionsBox.height > triggerBox.y;
  });
}

test.describe("Auto Router template select anchoring", () => {
  test.use({ storageState: ADMIN_STORAGE_PATH });

  for (const { room, height } of [
    { room: "with room below it", height: 900 },
    { room: "with no room below it", height: 560 },
  ]) {
    test(`keeps the trigger uncovered when the options open ${room}`, async ({ page }) => {
      await page.setViewportSize({ width: 1280, height });
      const trigger = await openTemplateSelect(page);
      await trigger.scrollIntoViewIfNeeded();

      await trigger.click();
      const options = page.getByRole("listbox");
      await expect(options).toBeVisible();

      await pollOptionsCoverTrigger(trigger, options).toBe(false);
    });
  }
});
