import { chromium, expect, request } from "@playwright/test";
import { users, Role, STORAGE_PATHS } from "./fixtures/users";
import { BASE_URL } from "./constants";
import * as fs from "fs";

// The suite historically relied on fixtures/seed.sql, which only exists on a
// locally provisioned database. Create the password users through the proxy API
// instead so the suite can run against any target, seeded or not.
//
// /user/new accepts a `password` field but does not persist it (login then
// reports "User has no password set"), so the password is set in a follow-up
// /user/update. Both calls are tolerant of the user already existing, which
// keeps repeat runs and parallel shards idempotent.
async function ensurePasswordUser(
  api: Awaited<ReturnType<typeof request.newContext>>,
  masterKey: string,
  email: string,
  password: string,
  userRole: string,
) {
  const auth = { Authorization: `Bearer ${masterKey}` };
  const userId = `e2e-${email.split("@")[0]}`;

  // 400 here means the user already exists, which is fine.
  await api.post(`${BASE_URL}/user/new`, {
    headers: auth,
    data: { user_id: userId, user_email: email, user_role: userRole },
  });

  const res = await api.post(`${BASE_URL}/user/update`, {
    headers: auth,
    data: { user_id: userId, password },
  });
  if (!res.ok()) {
    throw new Error(`Setting the password for ${email} failed (${res.status()}): ${await res.text()}`);
  }
}

async function globalSetup() {
  const browser = await chromium.launch();
  const rootPath = process.env.SERVER_ROOT_PATH ?? "";

  // The Projects sidebar item is hidden unless the enterprise-gated
  // enable_projects_ui setting is on, and the seeded DB starts with it off.
  // The proxy runs with LITELLM_LICENSE in CI, so enable it the same way
  // the admin UI toggle does; the projects migration smoke needs the link.
  const masterKey = process.env.LITELLM_MASTER_KEY || "sk-1234";
  const api = await request.newContext();
  const settingsRes = await api.patch(`${BASE_URL}${rootPath}/update/ui_settings`, {
    headers: { Authorization: `Bearer ${masterKey}` },
    data: { enable_projects_ui: true },
  });
  if (!settingsRes.ok()) {
    throw new Error(`Enabling enable_projects_ui failed (${settingsRes.status()}): ${await settingsRes.text()}`);
  }
  await api.dispose();

  // ProxyAdmin logs in as "admin" with the master key and needs no user row;
  // every other role is a real user that must exist with a password.
  const seededRoles: Record<string, string> = {
    [Role.ProxyAdminViewer]: "proxy_admin_viewer",
    [Role.InternalUser]: "internal_user",
    [Role.InternalUserViewer]: "internal_user_viewer",
    [Role.TeamAdmin]: "internal_user",
  };
  const seedApi = await request.newContext();
  for (const [role, userRole] of Object.entries(seededRoles)) {
    const { email, password } = users[role as Role];
    await ensurePasswordUser(seedApi, masterKey, email, password, userRole);
  }
  await seedApi.dispose();

  for (const role of Object.values(Role)) {
    const { email, password } = users[role];
    const storagePath = STORAGE_PATHS[role];
    const page = await browser.newPage();
    try {
      await page.goto(`${BASE_URL}${rootPath}/ui/login`);
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
