import { test, expect } from "@playwright/test";
import { ADMIN_STORAGE_PATH } from "../../constants";
import { Page } from "../../fixtures/pages";
import { navigateToPage, dismissFeedbackPopup } from "../../helpers/navigation";
import { CHAT_MODEL_A, MOCK_RESPONSE_TEXT, masterKey } from "../../helpers/traffic";

test.describe("Second proxy admin", () => {
  test.use({ storageState: { cookies: [], origins: [] } });

  test("an invited admin can log in, mint a key, and call a model with it", async ({ page, browser, request }) => {
    const suffix = Date.now();
    const email = `second-admin-${suffix}@test.local`;
    const password = "E2e-Second-Admin-Pass-1!";
    const auth = { Authorization: `Bearer ${masterKey()}` };

    const inviteAdminUser = async (): Promise<string> => {
      const adminContext = await browser.newContext({ storageState: ADMIN_STORAGE_PATH });
      try {
        const adminPage = await adminContext.newPage();
        await navigateToPage(adminPage, Page.Users);
        await dismissFeedbackPopup(adminPage);

        await adminPage.getByRole("button", { name: "+ Invite User", exact: true }).click();
        const dialog = adminPage.getByRole("dialog", { name: "Invite User" });
        await expect(dialog).toBeVisible({ timeout: 5_000 });

        await dialog.getByLabel("User Email").fill(email);

        await dialog.getByLabel(/Global Proxy Role/).click();
        await adminPage.getByRole("option", { name: /Admin \(All Permissions\)/ }).click();

        const createdResponse = adminPage.waitForResponse(
          (res) => res.url().includes("/user/new") && res.request().method() === "POST",
        );
        await dialog.getByRole("button", { name: "Invite User" }).click();
        const createdBody = await (await createdResponse).json();
        const createdUserId = (createdBody.data?.user_id ?? createdBody.user_id) as string;
        expect(createdUserId, "created user id from /user/new").toBeTruthy();

        await expect(adminPage.getByText("API user Created").first()).toBeVisible({ timeout: 10_000 });
        return createdUserId;
      } finally {
        await adminContext.close();
      }
    };

    const userId = await inviteAdminUser();
    try {
      const passwordRes = await request.post("/user/update", {
        headers: auth,
        data: { user_email: email, password },
      });
      expect(passwordRes.ok(), `setting password failed (${passwordRes.status()}): ${await passwordRes.text()}`).toBe(
        true,
      );

      await page.goto("/ui/login");
      await page.getByPlaceholder("Enter your username").fill(email);
      await page.getByPlaceholder("Enter your password").fill(password);
      await page.getByRole("button", { name: "Login", exact: true }).click();
      await expect(page.locator("a", { hasText: "Virtual Keys" })).toBeVisible({ timeout: 30_000 });
      await dismissFeedbackPopup(page);

      await navigateToPage(page, Page.ApiKeys);
      await page.getByRole("button", { name: /Create New Key/i }).click();
      await expect(page.getByText("Key Ownership")).toBeVisible({ timeout: 10_000 });

      await page.getByLabel(/Key Name/).fill(`e2e-second-admin-key-${suffix}`);

      await page.getByRole("combobox", { name: "Select models" }).click();
      await page.getByRole("option", { name: "All Proxy Models", exact: true }).click();
      await page.keyboard.press("Escape");

      await page.getByRole("button", { name: "Create Key", exact: true }).click();

      await expect(page.getByText("Save your Key")).toBeVisible({ timeout: 10_000 });
      const apiKey = (await page.getByRole("dialog", { name: "Save your Key" }).locator("pre").innerText()).trim();
      expect(apiKey).toMatch(/^sk-/);
      await page.keyboard.press("Escape");

      const response = await page.request.post("/chat/completions", {
        headers: { Authorization: `Bearer ${apiKey}` },
        data: {
          model: CHAT_MODEL_A,
          messages: [{ role: "user", content: `second admin ping ${suffix}` }],
        },
      });
      expect(response.status()).toBe(200);
      const body = await response.json();
      expect(body.choices?.[0]?.message?.content).toBe(MOCK_RESPONSE_TEXT);
    } finally {
      if (userId) {
        await request.post("/user/delete", { headers: auth, data: { user_ids: [userId] } });
      }
    }
  });
});
