import { VirtualKeysPage } from "../page-object-models/virtual-keys.page";
import { test, expect } from "./../fixtures/fixtures";
import { loginDetailsSet } from "./../utils/utils";

/* 4644_Test_Adding_a_Model */
test("Adding a Model as Self", async ({ loginPage, virtualKeysPage, page }) => {
  let username = "admin";
  let password = "sk-1234";
  if (loginDetailsSet()) {
    console.log("Login details exist in .env file.");
    username = process.env.UI_USERNAME as string;
    password = process.env.UI_PASSWORD as string;
  }
  await loginPage.goto();
  /* await page.screenshot({ path: "./test-results/4644_Test_Adding_a_Model/go-to-login-page.png" }); */

  await loginPage.login(username, password);
  await expect(page.getByRole("button", { name: "User" })).toBeVisible();
  /* await page.screenshot({ path: "./test-results/4644_Test_Adding_a_Model/dashboard.png" }); */

  await virtualKeysPage.logout();
  await expect(
    page.getByRole("heading", { name: "LiteLLM Login" })
  ).toBeVisible();
  /* await page.screenshot({ path: "./test-results/4644_Test_Adding_a_Model/logout.png" }); */
});

/*
await page.goto('http://localhost:4000/sso/key/generate');
  await page.getByRole('textbox', { name: 'Username:' }).click();
  await page.getByRole('textbox', { name: 'Username:' }).fill('admin');
  await page.getByRole('textbox', { name: 'Password:' }).click();
  await page.getByRole('textbox', { name: 'Password:' }).fill('sk-1234');
  await page.getByRole('button', { name: 'Submit' }).click();
  await page.getByRole('menu').getByText('Models').click();
  await page.getByRole('tab', { name: 'Add Model' }).click();
  await page.getByRole('combobox', { name: '* Provider question-circle :' }).click();
  await page.locator('span').filter({ hasText: 'OpenAI' }).click();
  await page.locator('.ant-select-selection-overflow').click();
  await page.getByTitle('omni-moderation-latest', { exact: true }).locator('div').click();
  await page.locator('div').filter({ hasText: /^Model Mappings$/ }).click();
  await page.getByRole('textbox', { name: 'Type...' }).click();
  await page.locator('#model_mappings').getByText('omni-moderation-latest').click();
  await page.getByRole('textbox', { name: 'Type...' }).dblclick();
  await page.getByRole('textbox', { name: 'Type...' }).press('ControlOrMeta+c');
  await page.getByRole('textbox', { name: '* API Key question-circle :' }).click();
  await page.getByRole('textbox', { name: '* API Key question-circle :' }).fill('sk-1234');
  await page.getByRole('button', { name: 'Add Model' }).click();
  await page.getByRole('tab', { name: 'All Models' }).click();
  await page.getByRole('paragraph').filter({ hasText: 'gpt-4o' }).click();
  await page.locator('pre').filter({ hasText: 'gpt-4o' }).click();
  await page.getByText('omni-moderation-late...').first().click();
  await page.getByText('test-model-name').first().click();
  await page.getByText('omni-moderation-late...').nth(1).click();
  await page.getByRole('row', { name: '0f7f03f... omni-moderation-' }).getByRole('paragraph').nth(1).click();
  await page.getByRole('row', { name: '0f7f03f... omni-moderation-' }).locator('pre').first().click();
  await page.getByText('omni-moderation-2024...').click();
  await page.getByRole('row', { name: 'ed5bd9d... omni-moderation-' }).getByRole('paragraph').nth(1).click();
  await page.getByRole('row', { name: 'ed5bd9d... omni-moderation-' }).locator('pre').first().click();
  await page.getByRole('paragraph').filter({ hasText: 'test-model-name' }).click();
  await page.getByRole('row', { name: 'c0be84c... test-model-name' }).getByRole('paragraph').nth(1).click();
  await page.getByRole('row', { name: 'c0be84c... test-model-name' }).locator('pre').first().click();
  await page.getByText('omni-moderation-late...').nth(2).click();
  await page.getByRole('row', { name: '7f57e27... omni-moderation-' }).getByRole('paragraph').nth(1).click();
  await page.getByRole('button', { name: '7f57e27...' }).click();
  await page.getByRole('button', { name: 'Back to Models' }).click();
  await page.locator('pre').filter({ hasText: 'omni-moderation-late...' }).click();
*/
