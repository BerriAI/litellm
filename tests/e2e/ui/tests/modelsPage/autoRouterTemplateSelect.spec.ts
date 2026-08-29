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

function pollPixelsBelowTrigger(trigger: Locator, popup: Locator) {
  return expect.poll(async () => {
    const triggerBox = await trigger.boundingBox();
    const popupBox = await popup.boundingBox();
    if (!triggerBox || !popupBox) return null;
    return popupBox.y - (triggerBox.y + triggerBox.height);
  });
}

function pollPopupOverlapsTrigger(trigger: Locator, popup: Locator) {
  return expect.poll(async () => {
    const triggerBox = await trigger.boundingBox();
    const popupBox = await popup.boundingBox();
    if (!triggerBox || !popupBox) return null;
    return popupBox.y < triggerBox.y + triggerBox.height && popupBox.y + popupBox.height > triggerBox.y;
  });
}

test.describe("Auto Router template select anchoring", () => {
  test.use({ storageState: ADMIN_STORAGE_PATH });

  test("opens the options below the trigger rather than over it", async ({ page }) => {
    await page.setViewportSize({ width: 1280, height: 900 });
    const trigger = await openTemplateSelect(page);

    await trigger.click();
    const popup = page.locator('[data-slot="select-content"]');
    await expect(popup).toBeVisible();

    // Item-aligned mode reports "none" and puts the active item over the trigger.
    await expect(popup).toHaveAttribute("data-side", "bottom");
    await pollPixelsBelowTrigger(trigger, popup).toBeGreaterThanOrEqual(0);
  });

  test("flips above the trigger instead of covering it when there is no room below", async ({ page }) => {
    await page.setViewportSize({ width: 1280, height: 560 });
    const trigger = await openTemplateSelect(page);
    await trigger.scrollIntoViewIfNeeded();

    await trigger.click();
    const popup = page.locator('[data-slot="select-content"]');
    await expect(popup).toBeVisible();

    await pollPopupOverlapsTrigger(trigger, popup).toBe(false);
  });
});
