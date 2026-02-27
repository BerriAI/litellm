import { test, expect, Page } from "@playwright/test";
import { ADMIN_STORAGE_PATH } from "../../constants";
test.skip("Internal Users Search", () => {
  test.use({ storageState: ADMIN_STORAGE_PATH });

  async function goToInternalUsers(page: Page) {
    await page.goto("/ui");

    const tab = page.getByRole("menuitem", { name: "Internal User" });
    await expect(tab).toBeVisible();
    await tab.click();

    await expect(page.locator("tbody tr").first()).toBeVisible();
    await expect(page.locator(".ant-skeleton")).toHaveCount(0);
  }

  test("can search users by email", async ({ page }) => {
    await goToInternalUsers(page);

    const rows = page.locator("tbody tr");
    const searchInput = page.getByPlaceholder("Search by email...");

    await expect(searchInput).toBeVisible();

    // Ensure initial data is loaded
    const initialCount = await rows.count();
    expect(initialCount).toBeGreaterThan(0);

    // 🔹 Apply filter + wait for backend response
    await Promise.all([
      page.waitForResponse(
        (res) =>
          res.url().includes("/user/list") &&
          res.url().includes("user_email=test%40") && // encoded "test@"
          res.status() === 200,
      ),
      searchInput.fill("test@"),
    ]);
    await page.waitForTimeout(5000);
    const filteredCount = await rows.count();
    await expect(filteredCount).toBeLessThan(initialCount);

    // 🔹 Clear filter + wait for unfiltered request
    await Promise.all([
      page.waitForResponse(
        (res) => res.url().includes("/user/list") && !res.url().includes("user_email=") && res.status() === 200,
      ),
      searchInput.clear(),
    ]);

    const resetCount = await rows.count();
    await expect(resetCount).toBe(initialCount);
  });

  test("can filter users by user ID and SSO ID", async ({ page }) => {
    await goToInternalUsers(page);
    const rows = page.locator("tbody tr");

    // Ensure initial data is loaded
    const initialCount = await rows.count();
    expect(initialCount).toBeGreaterThan(0);

    const filtersButton = page.getByRole("button", {
      name: "Filters",
      exact: true,
    });
    await filtersButton.click();

    const userIdInput = page.getByPlaceholder("Filter by User ID");
    const ssoIdInput = page.getByPlaceholder("Filter by SSO ID");
    await Promise.all([
      page.waitForResponse(
        (res) => res.url().includes("/user/list") && res.url().includes("user_ids=user") && res.status() === 200,
      ),
      userIdInput.fill("user"),
    ]);

    await Promise.all([
      page.waitForResponse(
        (res) =>
          res.url().includes("/user/list") &&
          res.url().includes("user_ids=user") &&
          res.url().includes("sso_user_ids=sso") &&
          res.status() === 200,
      ),
      ssoIdInput.fill("sso"),
    ]);
    const combinedFilteredCount = await rows.count();
    await expect(combinedFilteredCount).toBeLessThan(initialCount);
  });
});
