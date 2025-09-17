import { chromium, FullConfig } from '@playwright/test';

export default async function globalSetup(_config: FullConfig) {
    const browser = await chromium.launch();
    const page = await browser.newPage();

    await page.goto('http://localhost:4000/ui');
    await page.fill('input[name="username"]', 'admin');
    await page.fill('input[name="password"]', 'gm');
    await page.locator('input[type="submit"]').click();

    // Wait for something that proves you’re logged in:
    await page.getByRole('menuitem', { name: 'Guardrails' }).waitFor();

    // Save the logged-in state for reuse
    await page.context().storageState({ path: 'storage/admin.json' });
    await browser.close();
}
