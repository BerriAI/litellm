import { test, expect } from "@playwright/test";

test("add and verify model", async ({ page }) => {
  // Go to the UI and login (same as view_internal_user.spec.ts)
  await page.goto("http://localhost:4000/ui");
  await page.fill('input[name="username"]', "admin");
  await page.fill('input[name="password"]', "gm");
  const loginButton = page.locator('input[type="submit"]');
  await loginButton.click();

  // Generate a unique model name for verification
  const randomId = Math.floor(Math.random() * 10000);
  const testModelName = `test-model-${randomId}`;

  // Navigate to Models tab
  const modelsTab = page.locator("span.ant-menu-title-content", {
    hasText: "Models",
  });
  await modelsTab.click();

  // Click the Add Model tab (using role='tab' to be more specific)
  const addModelTab = page.getByRole('tab', { name: 'Add Model' });
  await addModelTab.click();

  // Validate that there is an h2 with text Add new model
  const addModelHeader = page.locator("h2", { hasText: "Add new model" });
  await expect(addModelHeader).toBeVisible();

  // Fill in the form
  // Select Provider (using the Ant Design Select component)
  // First click the select to open the dropdown
  await page.click('.ant-select-selector');
  // Wait for dropdown to be visible
  await page.waitForSelector('.ant-select-dropdown');
  // Click the OpenAI option
  await page.locator('.ant-select-item-option').filter({ hasText: 'OpenAI' }).click();

  // Select LiteLLM Model Name - All OpenAI Models (Wildcard)
  // Get all select elements and click the second one
  const selects = page.locator('.ant-select-selector');
  await selects.nth(1).click();  // Click the second select
  await page.waitForSelector('.ant-select-dropdown');  // Wait for dropdown
  await page.locator('.ant-select-item-option-content').filter({ hasText: 'gpt' }).nth(0).click();


  // Enter Model Name (using id selector for stability)
  const modelNameInput = page.locator('#model_name');
  await expect(modelNameInput).toBeVisible();
  await modelNameInput.fill(testModelName);

  // Enter API Key
  const apiKeyInput = page.locator('#api_key');
  await expect(apiKeyInput).toBeVisible();
  await apiKeyInput.fill('dummy-api-key');

  // Click Add Model button (the submit button)
  const submitButton = page.locator('button[type="submit"]').filter({ hasText: 'Add Model' });
  await submitButton.click();

  // Wait for success message or redirect
  await page.waitForTimeout(1000);

  // Navigate to All Models view
  const allModelsTab = page.getByRole('tab', { name: 'All Models' });
  await allModelsTab.click();

  // Wait for the table to be visible and verify the model exists
  const modelRow = page.locator('tr', { has: page.getByText(testModelName) });
  await expect(modelRow).toBeVisible({ timeout: 10000 });

  // Find and click the delete button in the same row
  const deleteButton = modelRow.locator('div.tremor-Grid-root').locator('span').nth(2);
  await deleteButton.click();

  // Wait for and click the confirm delete button in the modal
  const confirmDeleteButton = page.locator('button.ant-btn-dangerous');
  await expect(confirmDeleteButton).toBeVisible();
  await confirmDeleteButton.click();

  // Wait 3 seconds then refresh the oage and verify the model is no longer in the table
  await page.waitForTimeout(3000);
  await page.reload();
  await expect(modelRow).not.toBeVisible({ timeout: 10000 });
});