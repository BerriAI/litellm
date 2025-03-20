import { test, expect } from "./../fixtures/fixtures";
import { loginDetailsSet } from "./../utils/utils";

test("4644_Test_Adding_a_Model", async ({
  loginPage,
  dashboardLinks,
  modelsPage,
  page,
}) => {
  let username = "admin";
  let password = "sk-1234";
  if (loginDetailsSet()) {
    console.log("Login details exist in .env file.");
    username = process.env.UI_USERNAME as string;
    password = process.env.UI_PASSWORD as string;
  }

  console.log("1. Navigating to 'Login' page and logging in");
  await loginPage.goto();
  await page.screenshot({
    path: "./test-results/4644_Test_Adding_a_Model/00_go-to-login-page.png",
  });

  await loginPage.login(username, password);
  await expect(page.getByRole("button", { name: "User" })).toBeVisible();
  await page.screenshot({
    path: "./test-results/4644_Test_Adding_a_Model/01_dashboard.png",
  });

  //start
  // 2. Navigate to 'Models' page.
  console.log("2. Navigating to 'Models' page");
  await dashboardLinks.getModelsPageLink().click();
  await page.screenshot({
    path: "./test-results/4644_Test_Adding_a_Model/02_navigate-to-models-page.png",
  });
  // 3. Select 'Add Model' in the header of this page.
  console.log("3. Selecting 'Add Model' in the header of 'Models' page");
  await modelsPage.getAddModelTab().click();
  await page.screenshot({
    path: "./test-results/4644_Test_Adding_a_Model/03_navigate-to-add-models-tab.png",
  });
  // 4. Select OpenAI from 'Provider' dropdown.
  console.log("4. Selecting OpenAI from 'Provider' dropdown");
  await modelsPage.getProviderCombobox().click();
  await modelsPage.getProviderComboboxOption("OpenAI").click();
  await page.screenshot({
    path: "./test-results/4644_Test_Adding_a_Model/04_select-openai-provider.png",
  });
  // 5. Select model name from 'LiteLLM Model Name(s)' dropdown, and verify that the selected models appear in the 'LiteLLM Model' column in the 'Model Mappings' section. **Note**: Don't select 'All OpenAI Models (Wildcard)'.
  console.log("5. Selecting model name from 'LiteLLM Model Name(s)' dropdown");
  await modelsPage.getLitellModelNameCombobox().click();
  await modelsPage.getLitellmModelNameComboboxOption("gpt-4o").click();
  await page.screenshot({
    path: "./test-results/4644_Test_Adding_a_Model/05_select-gpt-4o-model.png",
  });
  // 6. Add API Key.
  console.log("6. Adding API Key");
  await modelsPage.getAPIKeyInputBox().fill("sk-1234");
  await page.screenshot({
    path: "./test-results/4644_Test_Adding_a_Model/06_enter-api-key.png",
  });
  // 7. Click 'Add Model'.
  console.log("7. Clicking 'Add Model'");
  await modelsPage.getAddModelSubmitButton(); //.click();
  await page.screenshot({
    path: "./test-results/4644_Test_Adding_a_Model/07_add-model.png",
  });
  // 8. Navigate to 'All Models' and verify that the models added show up in view with the Public name that was given to them.
  console.log("8. Navigating to 'All Models'");
  await modelsPage.getAllModelsTab().click();
  await page.screenshot({
    path: "./test-results/4644_Test_Adding_a_Model/08_navigate-to-all-models-tab.png",
  });
  //stop

  console.log("9. Logging out");
  await page.pause();
  await dashboardLinks.logout();
  await expect(
    page.getByRole("heading", { name: "LiteLLM Login" })
  ).toBeVisible();
  await page.screenshot({
    path: "./test-results/4644_Test_Adding_a_Model/09_logout.png",
  });
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
