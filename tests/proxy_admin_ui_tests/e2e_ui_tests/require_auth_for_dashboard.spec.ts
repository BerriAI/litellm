// tests/auth.spec.ts
import { test, expect } from "@playwright/test";

test.describe("Authentication Checks", () => {
  test("should redirect unauthenticated user from a protected page", async ({
    page,
  }) => {
    test.setTimeout(30000);

    page.on("console", (msg) => console.log("PAGE LOG:", msg.text()));

    const protectedPageUrl = "http://localhost:4000/ui?page=llm-playground";
    const expectedRedirectUrl = "http://localhost:4000/ui/login/";

    console.log(
      `Attempting to navigate to protected page: ${protectedPageUrl}`
    );

    await page.goto(protectedPageUrl);

    console.log(`Navigation initiated. Current URL: ${page.url()}`);

    try {
      await page.waitForURL(expectedRedirectUrl, { timeout: 10000 });
      console.log(`Waited for URL. Current URL is now: ${page.url()}`);
    } catch (error) {
      console.error(
        `Timeout waiting for URL: ${expectedRedirectUrl}. Current URL: ${page.url()}`
      );
      await page.screenshot({ path: "redirect-fail-screenshot.png" });
      throw error;
    }

    await expect(page).toHaveURL(expectedRedirectUrl);
    console.log(`Assertion passed: Page URL is ${expectedRedirectUrl}`);
  });
});
