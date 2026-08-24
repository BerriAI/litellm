import { expect, test } from "@playwright/test";
import { ADMIN_STORAGE_PATH } from "../../constants";

test.describe("Models and Endpoints responsive header", () => {
  test.use({
    storageState: ADMIN_STORAGE_PATH,
    viewport: { width: 900, height: 720 },
  });

  test("keeps the refresh action on the same row as the tabs", async ({
    page,
  }) => {
    await page.goto("/ui");
    await page
      .getByRole("complementary")
      .getByRole("link", { name: "Models + Endpoints" })
      .click();

    const tabs = page.getByRole("tablist");
    const refresh = page.getByRole("button", { name: "Refresh models" });
    await expect(tabs).toBeVisible();
    await expect(refresh).toBeVisible();

    const tabsBox = await tabs.boundingBox();
    const refreshBox = await refresh.boundingBox();
    expect(tabsBox).not.toBeNull();
    expect(refreshBox).not.toBeNull();

    const tabsCenterY = tabsBox!.y + tabsBox!.height / 2;
    const refreshCenterY = refreshBox!.y + refreshBox!.height / 2;
    expect(Math.abs(tabsCenterY - refreshCenterY)).toBeLessThanOrEqual(2);
  });
});
