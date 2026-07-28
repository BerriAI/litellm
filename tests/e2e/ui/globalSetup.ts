import { chromium, expect, request, APIRequestContext } from "@playwright/test";
import { users, Role, STORAGE_PATHS, DB_ROLE_USERS } from "./fixtures/users";
import { ARTIFACT_DIR, UI_BASE_URL } from "./constants";
import * as fs from "fs";
import * as path from "path";

async function ensureDbRoleUser(
  api: APIRequestContext,
  baseUrl: string,
  masterKey: string,
  entry: { role: Role; userId: string; dbRole: string },
): Promise<void> {
  const { email, password } = users[entry.role];
  const auth = { Authorization: `Bearer ${masterKey}` };
  const created = await api.post(`${baseUrl}/user/new`, {
    headers: auth,
    data: {
      user_id: entry.userId,
      user_email: email,
      user_role: entry.dbRole,
      password,
      auto_create_key: false,
    },
  });
  if (created.ok()) return;
  const body = await created.text();
  if (created.status() === 400 && body.includes("already exists")) {
    const updated = await api.post(`${baseUrl}/user/update`, {
      headers: auth,
      data: { user_id: entry.userId, user_role: entry.dbRole, password },
    });
    if (!updated.ok()) {
      throw new Error(
        `Ensuring UI role user ${entry.userId} failed on /user/update (${updated.status()}): ${await updated.text()}`,
      );
    }
    return;
  }
  throw new Error(`Ensuring UI role user ${entry.userId} failed on /user/new (${created.status()}): ${body}`);
}

async function globalSetup() {
  const browser = await chromium.launch();
  const rootPath = process.env.SERVER_ROOT_PATH ?? "";

  // Create the artifact root before anything writes into it. storageState() does
  // not create missing parents, so pointing E2E_UI_ARTIFACT_DIR at a path that
  // does not exist yet would fail with ENOENT on the first role's snapshot,
  // before any test ran. Playwright creates its own outputDir lazily, so this is
  // the only place that has to do it. recursive:true makes it idempotent, and
  // keeps the default "." a no-op.
  fs.mkdirSync(ARTIFACT_DIR, { recursive: true });

  // The Projects sidebar item is hidden unless the enterprise-gated
  // enable_projects_ui setting is on, and the seeded DB starts with it off.
  // The proxy runs with LITELLM_LICENSE in CI, so enable it the same way
  // the admin UI toggle does; the projects migration smoke needs the link.
  const masterKey = process.env.LITELLM_MASTER_KEY || "sk-1234";
  const api = await request.newContext();
  const settingsRes = await api.patch(`${UI_BASE_URL}${rootPath}/update/ui_settings`, {
    headers: { Authorization: `Bearer ${masterKey}` },
    data: { enable_projects_ui: true },
  });
  if (!settingsRes.ok()) {
    throw new Error(`Enabling enable_projects_ui failed (${settingsRes.status()}): ${await settingsRes.text()}`);
  }
  for (const entry of DB_ROLE_USERS) {
    await ensureDbRoleUser(api, `${UI_BASE_URL}${rootPath}`, masterKey, entry);
  }
  await api.dispose();

  for (const role of Object.values(Role)) {
    const { email, password } = users[role];
    const storagePath = STORAGE_PATHS[role];
    const page = await browser.newPage();
    try {
      await page.goto(`${UI_BASE_URL}${rootPath}/ui/login`);
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
      // Best-effort diagnostics only: this handler must never replace the real
      // failure with its own. Writing the screenshot used to throw ENOENT/EROFS
      // on the read-only cwd in the e2e image, which masked every underlying
      // login error and made the run look like a filesystem bug.
      try {
        const failureDir = path.join(ARTIFACT_DIR, "test-results");
        fs.mkdirSync(failureDir, { recursive: true });
        await page.screenshot({
          path: path.join(failureDir, `global-setup-${role}-failure.png`),
          fullPage: true,
        });
        console.error(`Global setup failed for role ${role}. Screenshot saved. URL: ${page.url()}`);
      } catch (diagnosticError) {
        console.error(
          `Global setup failed for role ${role} at URL: ${page.url()}. ` +
            `Could not save a screenshot: ${diagnosticError}`,
        );
      }
      throw e;
    } finally {
      await page.close();
    }
  }

  await browser.close();
}

export default globalSetup;
