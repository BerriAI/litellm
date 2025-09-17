import { test, expect } from "@playwright/test";

test("create a guardrail", async ({ page }) => {
    await page.goto("/ui");

    await page.getByRole('menuitem', { name: 'Guardrails' }).click();
    await page.getByRole('button', { name: /\+?\s*Add New Guardrail/i }).click();
    await page.locator('#guardrail_name').fill('test-guardrail');

    const combo = page.locator('input#provider[role="combobox"]');

    await combo.click();
    await combo.fill('Presidio');
    await page.keyboard.press('ArrowDown');
    await page.keyboard.press('Enter');

    await page.getByRole('button', { name: 'Next' }).click();
    await page.getByRole('button', { name: 'Select All & Mask' }).click();
    await page.getByRole('button', { name: 'Next' }).click();
    await page.getByRole('button', { name: 'Create Guardrail' }).click();
    await page.getByRole('cell', { name: 'test-guardrail'}).isVisible();
    await page.waitForTimeout(4000);
});

test("create a virtual key with a guardrail", async ({page}) => {
    await page.goto("/ui");

    await page.getByRole('menuitem', { name: 'Virtual Keys' }).click();
    await page.getByRole('button', {name: 'Create New Key'}).click();

    await page.locator('#key_alias').fill('test-key');

    await page
        .locator('div.ant-select[aria-describedby="models_help"] .ant-select-selector')
        .click();
    await page.keyboard.press('Enter');

    await page.getByText("Optional Settings").click();

    await page.locator(
        '.ant-select:has(.ant-select-selection-placeholder:has-text("Select or enter guardrails")) .ant-select-selector'
    ).click();
    await page.locator('.ant-select-item-option-content', { hasText: 'test-guardrail' }).click();

    await page.getByRole("button", { name: 'Create Key'}).click();
})

test("validate guardrail configuration ability on virtual key view page", async ({page}) => {
    await page.goto(`/ui`);

    const loading = page.getByText("Loading keys")
    await expect(loading).toBeHidden();

    const row = page.getByRole('row', { name: 'test-key'});

    const firstCell = row.getByRole('cell').first();
    await firstCell.getByRole("button").click();

    await page.getByRole("tab", {name: "Settings"}).click();
    await expect(page.getByText("test-guardrail")).toBeVisible();

    // test remove guardrail
    await page.getByRole("button", {name: "Edit Settings"}).click();
    const chip = page.locator('span.ant-select-selection-item[title="test-guardrail"]');
    await chip.hover(); // AntD shows the remove icon on hover
    await chip.locator('span.ant-select-selection-item-remove').click();

    await page.getByRole("button", {name: "Save Changes"}).click();
    await expect(page.getByText("test-guardrail")).toBeHidden();
})
