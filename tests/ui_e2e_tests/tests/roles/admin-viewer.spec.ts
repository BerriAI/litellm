import { test, expect } from "@playwright/test";
import { Role } from "../../constants";
import { loginAs } from "../../helpers/login";

test.describe("Admin Viewer Role", () => {
  test("Should not see Test Key page", async ({ page }) => {
    await loginAs(page, Role.ProxyAdminViewer);
    await expect(page.getByRole("menuitem", { name: "Test Key" })).not.toBeVisible();
  });
});
