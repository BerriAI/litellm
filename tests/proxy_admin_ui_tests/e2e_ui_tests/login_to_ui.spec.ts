/*

Login to Admin UI
Basic UI Test

Click on all the tabs ensure nothing is broken
*/

import { test, expect } from '@playwright/test';

test('admin login test', async ({ page }) => {
  // Go to the specified URL
  await page.goto('http://localhost:4000/ui');

  // Enter "admin" in the username input field
  await page.fill('input[name="username"]', 'admin');

  // Enter "gm" in the password input field
  await page.fill('input[name="password"]', 'gm');

  // Optionally, you can add an assertion to verify the login button is enabled
  const loginButton = page.locator('input[type="submit"]');
  await expect(loginButton).toBeEnabled();

  // Optionally, you can click the login button to submit the form
  await loginButton.click();
  const tabs = [
    'Virtual Keys',
    'Test Key',
    'Models',
    'Usage',
    'Teams',
    'Internal User',
    'Logging & Alerts',
    'Caching',
    'Budgets',
    'Router Settings',
    'Pass-through',
    'Admin Settings',
    'API Reference',
    'Model Hub'
  ];

  for (const tab of tabs) {
    const tabElement = page.locator('p.text-tremor-default.text-tremor-content.dark\\:text-dark-tremor-content', { hasText: tab });
    await tabElement.click();
  }
});
