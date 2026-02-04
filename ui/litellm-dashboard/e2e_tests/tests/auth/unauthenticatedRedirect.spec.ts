import { test, expect } from "@playwright/test";

test.describe("Authentication Checks", () => {
  test("should redirect unauthenticated user from a protected page", async ({ page }) => {
    const protectedPageUrl = "http://localhost:4000/ui?page=llm-playground";
    const expectedRedirectUrl = "http://localhost:4000/ui/login/";
    await page.goto(protectedPageUrl, { waitUntil: "domcontentloaded" });
    await expect(page).toHaveURL(expectedRedirectUrl);
    await expect(page.getByRole("heading", { name: "Login" })).toBeVisible();
  });
});
