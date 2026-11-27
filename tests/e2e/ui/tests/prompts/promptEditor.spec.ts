import { test, expect } from "@playwright/test";

import { ADMIN_STORAGE_PATH } from "../../constants";
import { Page as DashboardPage } from "../../fixtures/pages";
import { navigateToPage } from "../../helpers/navigation";
import { runWithCleanup } from "../../helpers/roundTrip";
import { masterKey, uniqueSuffix } from "../../helpers/traffic";

test.use({ storageState: ADMIN_STORAGE_PATH });

test("publishes a prompt and persists edits made in Prompt Studio", async ({ page }) => {
  const promptId = `e2e-studio-${uniqueSuffix()}`;
  const developerMessage = page.getByPlaceholder("e.g., You are a helpful assistant...");
  const promptRow = page.getByText(promptId, { exact: true });
  const headers = { Authorization: `Bearer ${masterKey()}` };
  const promptPath = `/prompts/${promptId}`;
  const params = { environment: "development" };
  const savedDeveloperMessage = async () => {
    const response = await page.request.get(`${promptPath}/info`, { headers, params });
    if (!response.ok()) return undefined;
    const info = (await response.json()) as { raw_prompt_template?: { content?: string } };
    return info.raw_prompt_template?.content?.match(/^Developer: (.*)$/m)?.[1];
  };

  await runWithCleanup(
    async () => {
      await navigateToPage(page, DashboardPage.Prompts);
      await page.getByRole("button", { name: "Add New Prompt", exact: true }).click();
      await developerMessage.fill("Use examples.");
      await page.getByRole("button", { name: "Save", exact: true }).click();
      const publish = page.getByRole("dialog", { name: "Publish Prompt" });
      await publish.getByRole("textbox", { name: "Name", exact: true }).fill(promptId);
      await publish.getByRole("button", { name: "Publish", exact: true }).click();
      await expect.poll(savedDeveloperMessage).toBe("Use examples.");

      await expect(async () => {
        await navigateToPage(page, DashboardPage.Prompts);
        await expect(promptRow).toBeVisible({ timeout: 2_000 });
      }).toPass({ timeout: 30_000 });
      await promptRow.click();
      await page.getByRole("button", { name: "Prompt Studio", exact: true }).click();
      await expect(developerMessage).toHaveValue("Use examples.");
      await developerMessage.press("End");
      await developerMessage.pressSequentially(" Keep it brief.");
      await page.getByRole("button", { name: "Update", exact: true }).click();
      await expect.poll(savedDeveloperMessage).toBe("Use examples. Keep it brief.");
    },
    async () => {
      await expect.poll(async () => {
        const response = await page.request.delete(promptPath, { headers, params });
        if (response.ok()) return true;
        if (response.status() !== 404) return false;
        const versions = await page.request.get(`${promptPath}/versions`, { headers, params });
        return versions.status() === 404;
      }, { timeout: 30_000, message: `Remove prompt ${promptId}` }).toBe(true);
      return true;
    },
  );
});
