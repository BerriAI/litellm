import { chromium, expect, request } from "@playwright/test";
import { users, Role, STORAGE_PATHS } from "./fixtures/users";
import { seedGateway } from "./fixtures/apiSeed";
import { PROXY_BASE_URL } from "./constants";
import * as fs from "fs";

async function globalSetup() {
  const browser = await chromium.launch();
  const rootPath = process.env.SERVER_ROOT_PATH ?? "";
  const masterKey = process.env.LITELLM_MASTER_KEY || "sk-1234";

  await seedGateway(`${PROXY_BASE_URL}${rootPath}`, masterKey);

  // The Projects sidebar item is hidden unless the enterprise-gated
  // enable_projects_ui setting is on, and the seeded DB starts with it off.
  // The proxy runs with LITELLM_LICENSE in CI, so enable it the same way
  // the admin UI toggle does; the projects migration smoke needs the link.
  const api = await request.newContext();
  const settingsRes = await api.patch(`${PROXY_BASE_URL}${rootPath}/update/ui_settings`, {
    headers: { Authorization: `Bearer ${masterKey}` },
    data: { enable_projects_ui: true },
  });
  if (!settingsRes.ok()) {
    throw new Error(`Enabling enable_projects_ui failed (${settingsRes.status()}): ${await settingsRes.text()}`);
  }
  await api.dispose();

  for (const role of Object.values(Role)) {
    const { email, password } = users[role];
    const storagePath = STORAGE_PATHS[role];
    const page = await browser.newPage();
    try {
      await page.goto(`${PROXY_BASE_URL}${rootPath}/ui/login`);
      await page.getByPlaceholder("Enter your username").fill(email);
      await page.getByPlaceholder("Enter your password").fill(password);
      await page.getByRole("button", { name: "Login", exact: true }).click();
      await page.waitForURL((url) => url.pathname.startsWith(`${rootPath}/ui`) && !url.pathname.includes("/login"), {
        timeout: 30_000,
      });
      await expect(page.locator("a", { hasText: "Virtual Keys" })).toBeVisible({ timeout: 30_000 });
      // Dismiss feedback popup if present
      const dismiss = page.getByText("Don't ask me again");
      if (await dismiss.isVisible({ timeout: 1_500 }).catch(() => false)) {
        await dismiss.click();
      }
      // The login flow stores a post-login return URL in the litellm_return_url
      // cookie. If the snapshot captures it before the app consumes it, every
      // test inheriting this storageState gets yanked to that stale URL the
      // first time it mounts a page (the e2e suite's main flake source).
      await page.context().clearCookies({ name: "litellm_return_url" });
      await page.context().storageState({ path: storagePath });
    } catch (e) {
      fs.mkdirSync("test-results", { recursive: true });
      await page.screenshot({ path: `test-results/global-setup-${role}-failure.png`, fullPage: true });
      console.error(`Global setup failed for role ${role}. Screenshot saved. URL: ${page.url()}`);
      throw e;
    } finally {
      await page.close();
    }
  }

  await browser.close();
}

export default globalSetup;
