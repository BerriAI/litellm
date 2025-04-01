/*
Test view internal user page
*/

import { test, expect } from "@playwright/test";

test("view internal user page", async ({ page }) => {
  // Go to the specified URL
  await page.goto("http://localhost:4000/ui");

  // Enter "admin" in the username input field
  await page.fill('input[name="username"]', "admin");

  // Enter "gm" in the password input field
  await page.fill('input[name="password"]', "sk-1234");

  // Optionally, you can add an assertion to verify the login button is enabled
  const loginButton = page.locator('input[type="submit"]');
  await expect(loginButton).toBeEnabled();

  // Optionally, you can click the login button to submit the form
  await loginButton.click();

  const tabElement = page.locator("span.ant-menu-title-content", {
    hasText: "Internal User",
  });
  await tabElement.click();

  // try to click on button
  // <button class="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded-l focus:outline-none" disabled="">← Prev</button>
  // wait 1-2 seconds
  // await page.waitForTimeout(10000);

  // Test all expected fields are present
  // number of keys owned by user
  const keysBadges = await page
    .locator("p.tremor-Badge-text.text-sm.whitespace-nowrap", {
      hasText: "Keys",
    })
    .all();

  await page.pause();

  /*
  const keysCountArray = await keysBadges.evaluateAll((elements) => {
    elements.map((el) => {
      console.log(el);
      parseInt(el.textContent.split(" ")[0], 10);
    });
  });
  */

  let keysCountArray: number[] = [];

  for (const element of keysBadges) {
    keysCountArray.push(
      parseInt((await element.innerText()).split(" ")[0], 10)
    );
  }

  const hasNonZeroKeys = keysCountArray.some((count) => count > 0);
  expect(hasNonZeroKeys).toBe(true);

  // test pagination
  const prevButton = page.locator(
    "button.px-3.py-1.text-sm.border.rounded-md.hover\\:bg-gray-50.disabled\\:opacity-50.disabled\\:cursor-not-allowed",
    { hasText: "Previous" }
  );
  await expect(prevButton).toBeDisabled();

  let paginationText = await page
    .locator(".flex.items-center.space-x-2")
    .locator(".text-sm.text-gray-700")
    .innerText();

  let paginationTextContents = paginationText.split(" ");

  let totalPages = parseInt(
    paginationTextContents[paginationTextContents.length - 1],
    10
  );

  if (totalPages > 1) {
    const nextButton = page.locator(
      "button.px-3.py-1.text-sm.border.rounded-md.hover\\:bg-gray-50.disabled\\:opacity-50.disabled\\:cursor-not-allowed",
      { hasText: "Next" }
    );
    await expect(nextButton).toBeEnabled();
  }
});
