import { expect, test } from "@playwright/test";
import { ADMIN_STORAGE_PATH } from "../../constants";

test.describe("Models and Endpoints responsive header", () => {
  test.use({
    storageState: ADMIN_STORAGE_PATH,
    viewport: { width: 900, height: 720 },
  });

  test("keeps the refresh action fully visible in the header above the tabs", async ({ page }) => {
    await page.goto("/ui");
    await page
      .getByRole("complementary")
      .getByRole("link", { name: "Model Management" })
      .click();

    const content = page.getByRole("main");
    const tabs = page.getByRole("tablist");
    const refresh = page.getByRole("button", { name: "Refresh models" });
    await expect(tabs).toBeVisible();
    await expect(refresh).toBeVisible();

    const contentBox = await content.boundingBox();
    const tabsBox = await tabs.boundingBox();
    const refreshBox = await refresh.boundingBox();
    expect(contentBox).not.toBeNull();
    expect(tabsBox).not.toBeNull();
    expect(refreshBox).not.toBeNull();

    expect(refreshBox!.x, "refresh is cut off by the left edge of the content area").toBeGreaterThanOrEqual(
      contentBox!.x,
    );
    expect(
      refreshBox!.x + refreshBox!.width,
      "refresh is cut off by the right edge of the content area",
    ).toBeLessThanOrEqual(contentBox!.x + contentBox!.width);
    await expect(refresh).toBeInViewport({ ratio: 1 });

    expect(refreshBox!.y + refreshBox!.height, "refresh overlaps or sits below the tabs").toBeLessThanOrEqual(
      tabsBox!.y,
    );

    await refresh.click({ trial: true });
  });
});
