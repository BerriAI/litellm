/*

Login to Admin UI
Basic UI Test

Click on all the tabs ensure nothing is broken
*/

import { test, expect } from '@playwright/test';

test('internal user', async ({ page }) => {
  // Go to the specified URL
  await page.goto('http://localhost:4000/ui');

  // Enter "internal user id" in the username input field
  await page.fill('input[name="username"]', 'sw12@berri.ai');

  // Enter "gm" in the password input field
  await page.fill('input[name="password"]', 'gm');

  // Optionally, you can add an assertion to verify the login button is enabled
  const loginButton = page.locator('input[type="submit"]');
  await expect(loginButton).toBeEnabled();

  // Optionally, you can click the login button to submit the form
  await loginButton.click();

  // Navigate to the 'Usage' tab
  const usageTab = page.locator('p.text-tremor-default.text-tremor-content.dark\\:text-dark-tremor-content', { hasText: 'Usage' });
  await usageTab.click();

  // wait 3 seconds 
  await page.waitForTimeout(1000);











});
