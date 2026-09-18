import { test, expect } from "@playwright/test";

import { ADMIN_STORAGE_PATH } from "../../constants";
import { Page as DashboardPage } from "../../fixtures/pages";
import { navigateToPage } from "../../helpers/navigation";
import { masterKey, uniqueSuffix } from "../../helpers/traffic";

test.use({ storageState: ADMIN_STORAGE_PATH });

test.describe("Prompt upload form", () => {
  test("uploads a prompt file and reads the created prompt back", async ({
    page,
  }) => {
    const promptId = `e2e-prompt-${uniqueSuffix()}`;
    const promptContent = "Hello {{name}}";
    const cleanup = async (): Promise<boolean> => {
      try {
        const response = await page.request.delete(
          `/prompts/${encodeURIComponent(promptId)}?environment=development`,
          {
            headers: { Authorization: `Bearer ${masterKey()}` },
          },
        );
        return response.ok();
      } catch {
        return false;
      }
    };
    const testOutcome = await (async () => {
      try {
        await navigateToPage(page, DashboardPage.Prompts);
        await page.getByRole("button", { name: "Upload .prompt File" }).click();
        await expect(
          page.getByRole("dialog", { name: "Add New Prompt" }),
        ).toBeVisible();
        await page.getByLabel("Prompt ID").fill(promptId);
        await page.locator('input[type="file"]').setInputFiles({
          name: "e2e.prompt",
          mimeType: "text/plain",
          buffer: Buffer.from(
            `model: fake-openai-gpt-4\ntemplate: "${promptContent}"\n`,
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
        await expect
          .poll(async () => {
            const response = await page.request.get(
              `/prompts/${encodeURIComponent(promptId)}/info?environment=development`,
              {
                headers: { Authorization: `Bearer ${masterKey()}` },
              },
            );
            if (!response.ok()) return undefined;
            const promptInfo = (await response.json()) as {
              raw_prompt_template?: { content?: string };
            };
            return promptInfo.raw_prompt_template?.content;
          })
          .toContain(promptContent);
        await expect(page.getByText(promptId, { exact: true })).toBeVisible();
        return { passed: true as const };
      } catch (error) {
        return { passed: false as const, error };
      }
    })();

    try {
      if (!testOutcome.passed) throw testOutcome.error;
    } finally {
      const cleanupSucceeded = await cleanup();
      if (testOutcome.passed) expect(cleanupSucceeded).toBe(true);
    }
  });
});
