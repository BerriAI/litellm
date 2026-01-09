import test, { expect } from "@playwright/test";
import { Role } from "../../fixtures/roles";
import { ADMIN_STORAGE_PATH } from "../../constants";

const sidebarButtons = {
  [Role.ProxyAdmin]: [
    "Virtual Keys",
    "Playground",
    "Models",
    "Usage",
    "Teams",
    "Internal User",
    "Settings",
    "Experimental",
    "API Reference",
    "AI Hub",
  ],
};

const roles = [{ role: Role.ProxyAdmin, storage: ADMIN_STORAGE_PATH }];

for (const { role, storage } of roles) {
  test.describe(`${role} sidebar`, () => {
    test.use({ storageState: storage });

    test("can see and navigate all sidebar buttons", async ({ page }) => {
      await page.goto("/ui");
      for (const button of sidebarButtons[role as keyof typeof sidebarButtons]) {
        const tab = page.getByRole("menuitem", { name: button });
        await expect(tab).toBeVisible();
        await tab.click();
      }
    });
  });
}
