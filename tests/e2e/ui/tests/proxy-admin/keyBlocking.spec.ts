import { test as base, expect } from "@playwright/test";
import { ADMIN_STORAGE_PATH } from "../../constants";
import { Page } from "../../fixtures/pages";
import { dismissFeedbackPopup, navigateToPage, openKeyDetail } from "../../helpers/navigation";
import {
  CHAT_MODEL_A,
  MOCK_RESPONSE_TEXT,
  attemptChatCompletion,
  createVirtualKey,
  deleteVirtualKey,
  readKeyInfo,
  sendChatCompletion,
  uniqueSuffix,
} from "../../helpers/traffic";

interface ScopedKey {
  alias: string;
  token: string;
  apiKey: string;
}

const test = base.extend<{ scopedKey: ScopedKey }>({
  scopedKey: async ({ page }, use) => {
    const alias = `e2e-block-key-${uniqueSuffix()}`;
    const created = await createVirtualKey(page.request, {
      key_alias: alias,
      models: [CHAT_MODEL_A],
    });
    await use({ alias, token: created.token, apiKey: created.key });
    await deleteVirtualKey(page.request, created.token);
  },
});

test.describe("Proxy Admin - Key blocking", () => {
  test.use({ storageState: ADMIN_STORAGE_PATH });

  test("blocking a key stops it serving and unblocking restores it", async ({ page, scopedKey }) => {
    const { alias, token, apiKey } = scopedKey;

    await sendChatCompletion(page.request, {
      model: CHAT_MODEL_A,
      prompt: `pre-block ${alias}`,
      apiKey,
    });

    await navigateToPage(page, Page.ApiKeys);
    await dismissFeedbackPopup(page);
    await openKeyDetail(page, alias);

    await page.getByRole("button", { name: "More key actions" }).click();
    await page.getByRole("menuitem", { name: "Block Key" }).click();
    const blockDialog = page.getByRole("dialog", { name: "Block Key" });
    await expect(blockDialog, "the Block Key confirmation never opened").toBeVisible({ timeout: 10_000 });
    await blockDialog.getByRole("button", { name: "Block", exact: true }).click();

    await expect
      .poll(async () => (await readKeyInfo(page.request, token)).blocked, {
        message: "the key never came back blocked from /key/info",
        timeout: 20_000,
      })
      .toBe(true);

    await expect
      .poll(
        async () =>
          await attemptChatCompletion(page.request, {
            model: CHAT_MODEL_A,
            prompt: "blocked",
            apiKey,
          }),
        {
          message: "a blocked key was still served by /v1/chat/completions",
          timeout: 30_000,
        },
      )
      .toMatchObject({ status: 401, body: expect.stringContaining("blocked") });

    await page.reload();
    await expect(
      page.getByText("Blocked", { exact: true }),
      "the reloaded key detail does not show the key as blocked",
    ).toBeVisible({ timeout: 15_000 });

    await page.getByRole("button", { name: "More key actions" }).click();
    await page.getByRole("menuitem", { name: "Unblock Key" }).click();
    const unblockDialog = page.getByRole("dialog", { name: "Unblock Key" });
    await expect(unblockDialog, "the Unblock Key confirmation never opened").toBeVisible({ timeout: 10_000 });
    await unblockDialog.getByRole("button", { name: "Unblock", exact: true }).click();

    await expect
      .poll(async () => (await readKeyInfo(page.request, token)).blocked, {
        message: "the key never came back unblocked from /key/info",
        timeout: 20_000,
      })
      .toBe(false);

    await expect
      .poll(
        async () =>
          await attemptChatCompletion(page.request, {
            model: CHAT_MODEL_A,
            prompt: "unblocked",
            apiKey,
          }),
        {
          message: "an unblocked key is still refused by /v1/chat/completions",
          timeout: 30_000,
        },
      )
      .toMatchObject({ status: 200, body: expect.stringContaining(MOCK_RESPONSE_TEXT) });
  });
});
