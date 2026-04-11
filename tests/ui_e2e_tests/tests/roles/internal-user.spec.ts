import { test, expect } from "@playwright/test";
import { Page, Role } from "../../constants";
import { loginAs } from "../../helpers/login";
import { navigateToPage } from "../../helpers/navigation";

test.describe("Internal User Role", () => {
  test("Should not see litellm-dashboard keys", async ({ page }) => {
    await loginAs(page, Role.InternalUser);
    await navigateToPage(page, Page.ApiKeys);
    await expect(page.getByText("litellm-dashboard")).not.toBeVisible();
  });
});
