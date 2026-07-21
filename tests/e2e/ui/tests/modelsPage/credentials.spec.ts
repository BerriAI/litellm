import { test, expect } from "@playwright/test";
import { ADMIN_STORAGE_PATH } from "../../constants";
import { Role, users } from "../../fixtures/users";

test.describe("Edit LLM credential", () => {
  test.use({ storageState: ADMIN_STORAGE_PATH });

  const masterKey = users[Role.ProxyAdmin].password;
  const SEED_API_KEY = "sk-e2e-credential-ABCDEFGHIJKLMNOP";
  const SEED_API_BASE = "https://api.openai.com/v1";
  const NEW_API_BASE = "https://proxy.e2e.example.com/v1";

  let credentialName: string;

  test.beforeEach(async ({ page }) => {
    credentialName = `e2e-cred-${Date.now()}`;
    const res = await page.request.post("/credentials", {
      headers: { Authorization: `Bearer ${masterKey}` },
      data: {
        credential_name: credentialName,
        credential_values: { api_key: SEED_API_KEY, api_base: SEED_API_BASE },
        credential_info: { custom_llm_provider: "openai" },
      },
    });
    expect(res.ok(), `POST /credentials for ${credentialName}`).toBe(true);
  });

  test.afterEach(async ({ page }) => {
    await page.request.delete(`/credentials/${credentialName}`, {
      headers: { Authorization: `Bearer ${masterKey}` },
    });
  });

  test("changing only the api base does not overwrite the stored api key with its masked value", async ({ page }) => {
    await page.goto("/ui");
    await page.getByText("Models + Endpoints").click();
    await page.getByRole("tab", { name: "LLM Credentials" }).click();

    const row = page.locator("tr", { hasText: credentialName });
    await expect(row).toBeVisible({ timeout: 15_000 });
    await row.getByRole("button").first().click();

    const modal = page.locator(".ant-modal-content").filter({ hasText: "Edit Credential" });
    await expect(modal).toBeVisible({ timeout: 10_000 });

    const apiKeyField = modal.locator("#api_key");
    const apiBaseField = modal.locator("#api_base");
    await expect(apiKeyField).toBeVisible({ timeout: 15_000 });

    await expect(apiKeyField, "form pre-fills the api key with the backend's masked value").toHaveValue(/\*{2,}/);
    await expect(apiKeyField).not.toHaveValue(SEED_API_KEY);

    await apiBaseField.fill(NEW_API_BASE);

    const patchPromise = page.waitForRequest(
      (req) => req.method() === "PATCH" && req.url().includes(`/credentials/${credentialName}`),
    );
    await modal.getByRole("button", { name: "Update Credential" }).click();
    const patchReq = await patchPromise;
    const patchBody = JSON.parse(patchReq.postData() ?? "{}");

    expect(patchBody.credential_values.api_base, "UI sends the edited api base").toBe(NEW_API_BASE);
    expect("api_key" in patchBody.credential_values, "UI must not send the masked api key back on update").toBe(false);

    await expect(page.getByText("Credential updated successfully")).toBeVisible({ timeout: 10_000 });

    const infoRes = await page.request.get(`/credentials/by_name/${credentialName}`, {
      headers: { Authorization: `Bearer ${masterKey}` },
    });
    expect(infoRes.ok()).toBe(true);
    const cred = await infoRes.json();
    expect(cred.credential_values.api_base, "edited api base persisted to the backend").toBe(NEW_API_BASE);
    expect(cred.credential_values.api_key, "a stored api key is still present (returned masked)").toMatch(/\*{2,}/);
  });
});
