import { test, expect, Page } from "@playwright/test";
import { ADMIN_STORAGE_PATH } from "../../constants";

test.skip("Internal Users Page", () => {
  test.use({ storageState: ADMIN_STORAGE_PATH });

  async function goToInternalUsers(page: Page) {
    await page.goto("/ui");

    const internalUserTab = page.getByRole("menuitem", { name: "Internal User" });
    await expect(internalUserTab).toBeVisible();
    await internalUserTab.click();

    const firstRow = page.locator("tbody tr").first();
    await expect(firstRow).toBeVisible();
    await expect(page.locator(".ant-skeleton")).toHaveCount(0);
  }

  test("renders internal users table correctly", async ({ page }) => {
    await goToInternalUsers(page);

    const rows = page.locator("tbody tr");
    const rowCount = await rows.count();
    expect(rowCount).toBeGreaterThan(0);

    const userIdHeader = page.getByRole("columnheader", { name: "User ID" });
    await expect(userIdHeader).toBeVisible();

    const virtualKeysHeader = page.getByRole("columnheader", { name: "Virtual Keys" });
    await expect(virtualKeysHeader).toBeVisible();
  });

  test("pagination controls work correctly", async ({ page }) => {
    await goToInternalUsers(page);

    const paginationInfo = page.locator(".text-sm.text-gray-700");
    const prevButton = page.getByRole("button", { name: "Previous" });
    const nextButton = page.getByRole("button", { name: "Next" });

    const infoText = (await paginationInfo.textContent()) || "";

    // On first page, Previous should be disabled
    if (infoText.includes("1 -")) {
      await expect(prevButton).toBeDisabled();
    }

    await page.waitForTimeout(1000);
    // Check if there are more pages
    const hasMorePages = infoText.includes("of") && !infoText.endsWith("25 of 25");
    if (hasMorePages) {
      await expect(nextButton).toBeEnabled();
    }
  });
});
