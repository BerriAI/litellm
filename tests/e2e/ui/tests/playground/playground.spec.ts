import { test, expect } from "@playwright/test";
import { ADMIN_STORAGE_PATH } from "../../constants";
import { CHAT_MODEL_A, CHAT_MODEL_B, MOCK_RESPONSE_TEXT, createVirtualKey } from "../../helpers/traffic";
import { keySourceSelect, onlyVisible, openPlayground, selectModel, sendMessage } from "../../helpers/playground";

/**
 * The one flow that exercises the dashboard's own LLM call path rather than an admin CRUD endpoint,
 * so it covers the UI's auth header, endpoint selection and streaming render.
 */
test.describe("Playground", () => {
  test.use({ storageState: ADMIN_STORAGE_PATH });

  for (const model of [CHAT_MODEL_A, CHAT_MODEL_B]) {
    test(`chats with ${model} using the current UI session`, async ({ page }) => {
      await openPlayground(page);

      // "Current UI Session" is the default: the logged-in admin's key, nothing pasted.
      await expect(keySourceSelect(page)).toContainText("Current UI Session");

      await selectModel(page, model);
      const prompt = `playground ping for ${model}`;
      await sendMessage(page, prompt);

      // Our prompt is echoed into the transcript, and the mock server replies.
      await expect(page.getByText(prompt, { exact: false }).first()).toBeVisible({ timeout: 20_000 });
      await expect(page.getByText(MOCK_RESPONSE_TEXT, { exact: false }).first()).toBeVisible({ timeout: 60_000 });
    });
  }

  test("chats using a pasted virtual key instead of the UI session", async ({ page, request }) => {
    const { key } = await createVirtualKey(request, {
      key_alias: `e2e-playground-${Date.now()}`,
    });

    await openPlayground(page);

    // Switch the source to "Virtual Key" and paste the key we just minted.
    await keySourceSelect(page).click();
    await onlyVisible(page.getByRole("option", { name: "Virtual Key" })).click({ timeout: 15_000 });

    const keyInput = onlyVisible(page.getByPlaceholder("Enter custom Virtual Key"));
    await expect(keyInput).toBeVisible({ timeout: 10_000 });
    await keyInput.fill(key);

    await selectModel(page, CHAT_MODEL_A);
    await sendMessage(page, "playground ping via virtual key");

    await expect(page.getByText(MOCK_RESPONSE_TEXT, { exact: false }).first()).toBeVisible({ timeout: 60_000 });
  });
});
