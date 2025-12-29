import { test, expect } from "@playwright/test";

test("basic test to verify playwright setup", async ({ page }) => {
  // Navigate to the base URL
  await page.goto("/");

  // Wait for the page to load and check if we can find some basic content
  // This is a very basic test just to verify the setup works
  await expect(page).toHaveTitle(/.*LiteLLM.*/);
});
