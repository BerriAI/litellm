import { test, expect } from "@playwright/test";
import { ADMIN_STORAGE_PATH, Page } from "../../constants";
import { navigateToPage } from "../../helpers/navigation";

test.describe("Proxy Admin Role", () => {
  test.use({ storageState: ADMIN_STORAGE_PATH });

  test("Can create keys", async ({ page }) => {
    await navigateToPage(page, Page.ApiKeys);
    await expect(page.getByRole("button", { name: /Create New Key/i })).toBeVisible();
  });

  test("Can list teams via API", async ({ page }) => {
    const response = await page.request.get("/team/list", {
      headers: {
        Authorization: `Bearer ${process.env.LITELLM_MASTER_KEY || "sk-1234"}`,
      },
    });
    expect(response.status()).toBe(200);
  });
});
