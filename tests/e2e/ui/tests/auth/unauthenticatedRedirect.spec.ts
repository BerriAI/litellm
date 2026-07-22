import { test, expect } from "@playwright/test";
import { PROXY_BASE_URL } from "../../constants";

test.describe("Authentication Checks", () => {
  test("should redirect unauthenticated user from a protected page", async ({ page }) => {
    const protectedPageUrl = `${PROXY_BASE_URL}/ui?page=llm-playground`;
    await page.goto(protectedPageUrl, { waitUntil: "domcontentloaded" });
    await expect(page).toHaveURL(/\/ui\/login/);
    await expect(page.getByRole("heading", { name: "Login" })).toBeVisible();
  });
});
