import { test, expect, Page } from "@playwright/test";

const ALLOWED_MODEL = process.env.E2E_UI_ALLOWED_MODEL ?? "gemini-2.5-flash";
const DENIED_MODEL = process.env.E2E_UI_DENIED_MODEL ?? "gpt-5.5";

const MASTER_KEY = process.env.LITELLM_MASTER_KEY ?? "sk-1234";
const adminAuth = { Authorization: `Bearer ${MASTER_KEY}` };

async function loginAsProxyAdmin(page: Page): Promise<void> {
  await page.goto("/ui/login");
  await page.getByPlaceholder("Enter your username").fill("admin");
  await page.getByPlaceholder("Enter your password").fill(MASTER_KEY);
  await page.getByRole("button", { name: "Login", exact: true }).click();
  await expect(page.getByText("Virtual Keys")).toBeVisible({ timeout: 30_000 });
}

async function dismissFeedbackPopup(page: Page): Promise<void> {
  const dismiss = page.getByText("Don't ask me again");
  if (await dismiss.isVisible({ timeout: 1_500 }).catch(() => false)) {
    await dismiss.click();
  }
}

test.describe("Proxy Admin - create virtual key via the Admin UI", () => {
  test("admin creates a model-scoped key and the gateway enforces its scope", async ({ page, request }) => {
    await loginAsProxyAdmin(page);

    await page.goto("/ui?page=api-keys");
    await dismissFeedbackPopup(page);

    await page.getByTestId("create-key-button").click();
    await expect(page.getByText("Key Ownership")).toBeVisible({ timeout: 15_000 });

    const alias = `e2e-ui-key-${Date.now()}`;
    await page.getByTestId("base-input").fill(alias);

    await page.locator(".ant-select-selection-overflow").click();
    const option = page.locator(".ant-select-dropdown:visible").getByRole("option", { name: ALLOWED_MODEL, exact: true });
    await option.waitFor({ state: "attached" });
    await option.evaluate((el: HTMLElement) => el.click());
    await page.keyboard.press("Escape");

    await page.getByRole("button", { name: "Create Key", exact: true }).click();
    await expect(page.getByText("Save your Key")).toBeVisible({ timeout: 15_000 });

    const secret = (await page.locator(".ant-modal:visible pre").innerText()).trim();
    expect(secret).toMatch(/^sk-/);

    await page.keyboard.press("Escape");
    await expect(page.getByText(alias)).toBeVisible({ timeout: 15_000 });

    try {
      const info = await request.get(`/key/info?key=${encodeURIComponent(secret)}`, { headers: adminAuth });
      expect(info.status()).toBe(200);
      const infoBody = await info.json();
      expect(infoBody.info.key_alias).toBe(alias);
      expect(infoBody.info.models).toEqual([ALLOWED_MODEL]);

      const chat = (model: string) =>
        request.post("/chat/completions", {
          headers: { Authorization: `Bearer ${secret}` },
          data: { model, messages: [{ role: "user", content: "ping" }], max_tokens: 16 },
        });

      await expect
        .poll(async () => (await chat(ALLOWED_MODEL)).status(), { timeout: 60_000, intervals: [2_000] })
        .toBe(200);

      const denied = await chat(DENIED_MODEL);
      expect(denied.status()).toBe(403);
      expect(await denied.text()).toContain("key_model_access_denied");
    } finally {
      await request.post("/key/delete", { headers: adminAuth, data: { keys: [secret] } });
    }
  });
});
