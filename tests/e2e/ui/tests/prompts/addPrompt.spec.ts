import { test, expect } from "@playwright/test";

import { ADMIN_STORAGE_PATH } from "../../constants";
import { Page as DashboardPage } from "../../fixtures/pages";
import { navigateToPage } from "../../helpers/navigation";
import { readBack } from "../../helpers/roundTrip";
import { masterKey, uniqueSuffix } from "../../helpers/traffic";

test.use({ storageState: ADMIN_STORAGE_PATH });

test.describe("Prompt upload form", () => {
  test("uploads a prompt file and reads the created prompt back", async ({
    page,
  }) => {
    const promptId = `e2e-prompt-${uniqueSuffix()}`;
    await navigateToPage(page, DashboardPage.Prompts);
    await page.getByRole("button", { name: "Upload .prompt File" }).click();

    try {
      await expect(
        page.getByRole("dialog", { name: "Add New Prompt" }),
      ).toBeVisible();
      await page.getByLabel("Prompt ID").fill(promptId);
      await page.locator('input[type="file"]').setInputFiles({
        name: "e2e.prompt",
        mimeType: "text/plain",
        buffer: Buffer.from(
          'model: fake-openai-gpt-4\ntemplate: "Hello {{name}}"\n',
        ),
      });
      await expect(page.getByText("Selected: e2e.prompt")).toBeVisible();
      await page.getByRole("button", { name: "Create Prompt" }).click();

      await expect
        .poll(async () => {
          const response = await page.request.get(
            `/prompts/${encodeURIComponent(promptId)}/info?environment=development`,
            {
              headers: { Authorization: `Bearer ${masterKey()}` },
            },
          );
          return response.ok();
        })
        .toBe(true);
      await expect(page.getByText(promptId, { exact: true })).toBeVisible();
    } finally {
      await page.request.delete(
        `/prompts/${encodeURIComponent(promptId)}?environment=development`,
        {
          headers: { Authorization: `Bearer ${masterKey()}` },
        },
      );
    }
  });
});
