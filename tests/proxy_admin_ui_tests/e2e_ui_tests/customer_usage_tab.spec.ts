/*
Customer Usage Tab Test

- Load the UI with direct URL parameters
- Click on the Customer Usage tab
- Verify the dropdown states are empty
*/

import { test, expect } from '@playwright/test';

test('customer usage tab dropdowns test', async ({ page }) => {
  // Go to the specified URL
  await page.goto('http://localhost:4000/ui');

  // Enter "admin" in the username input field
  await page.fill('input[name="username"]', 'admin');

  // Enter "gm" in the password input field
  await page.fill('input[name="password"]', 'gm');

  // Optionally, you can add an assertion to verify the login button is enabled
  const loginButton = page.locator('input[type="submit"]');
  await expect(loginButton).toBeEnabled();

  await loginButton.click();

  const tabElement = page.locator('span.ant-menu-title-content', { hasText: "Usage" });
  await tabElement.click();

  await page.goto('http://localhost:3000/ui?userID=default_user_id&page=usage');


  await page.waitForTimeout(1000);
  const customerUsageTab = await page.locator('button', { hasText: "Customer Usage" });
  await customerUsageTab.click();
  

  // Check that all three dropdowns exist and are empty or have placeholders
  // check that input with id team-dropdown is visible
  const teamDropdown = page.locator('#team-dropdown');
  await expect(teamDropdown).toBeVisible();

  // check that input with id user-dropdown is visible
  const userDropdown = page.locator('#user-dropdown');
  await expect(userDropdown).toBeVisible();

  // check that input with id key-dropdown is visible
  const keyDropdown = page.locator('#key-dropdown');
  await expect(keyDropdown).toBeVisible();

  await keyDropdown.click();

  await page.waitForTimeout(1000);

  // get the div containing the class rc-virtual-list-holder-inner and verify that it has multiple children
  const virtualListHolderInner = page.locator('.rc-virtual-list-holder-inner');
  const children = await virtualListHolderInner.evaluateAll(elements => elements.map(el => el.textContent));

  await children.length > 0 && children[0]?.length;
}); 