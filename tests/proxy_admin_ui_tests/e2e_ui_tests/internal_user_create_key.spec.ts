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


  // Create a new key for the user
  // click on span '+ Create New Key'

  await page.locator('span:has-text("+ Create New Key")').click();


  // set a random string for input where id="key_alias"
  const key_alias = Math.random().toString(36).slice(-8);
  await page.locator('input#key_alias').fill('key_alias' + key_alias);

  // select model = "fake-gpt-3.5-turbo" for the 
  // Select the model "fake-gpt-3.5-turbo"
  await page.click('#models'); // Open the select dropdown
  await page.click('div[title="fake-gpt-3.5-turbo"]'); // Select the specific option

  // click this button <button type="submit" class="ant-btn css-1qhpsh8 ant-btn-default"><span>Create Key</span></button>
  await page.click('button[type="submit"]');

  // store the content under here as "key" - <pre style="overflow-wrap: break-word; white-space: normal;">sk-9vEIOyLVd2Hmjrif9DQHJA</pre>
  const key = await page.textContent('pre');

  console.log(key);










});
