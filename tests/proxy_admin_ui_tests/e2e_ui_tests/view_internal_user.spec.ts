/*
Test view internal user page
*/

import { test, expect } from '@playwright/test';

test('view internal user page', async ({ page }) => {
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

  const tabElement = page.locator('p.text-tremor-default.text-tremor-content.dark\\:text-dark-tremor-content', { hasText: 'Internal User' });
  await tabElement.click();

  // try to click on button 
  // <button class="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded-l focus:outline-none" disabled="">← Prev</button>

  const prevButton = page.locator('button.bg-blue-500.hover\\:bg-blue-700.text-white.font-bold.py-2.px-4.rounded-l.focus\\:outline-none', { hasText: 'Prev' });
  await expect(prevButton).toBeDisabled();

  const nextButton = page.locator('button.bg-blue-500.hover\\:bg-blue-700.text-white.font-bold.py-2.px-4.rounded-r.focus\\:outline-none', { hasText: 'Next' });
  await expect(nextButton).toBeEnabled();
});
