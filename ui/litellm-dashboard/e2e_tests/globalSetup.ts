import { chromium } from "@playwright/test";
import { users } from "./fixtures/users";
import { Role } from "./fixtures/roles";
import { ADMIN_STORAGE_PATH, INTERNAL_USER_VIEWER_STORAGE_PATH } from "./constants";

async function loginAndSaveState(
  browser,
  user,
  storagePath: string
) {
  const context = await browser.newContext();
  const page = await context.newPage();

  await page.goto("http://localhost:4000/ui/login");
  await page.getByPlaceholder("Enter your username").fill(user.email);
  await page.getByPlaceholder("Enter your password").fill(user.password);
  await page.getByRole("button", { name: "Login" }).click();
  await page.getByText('AI GATEWAY').waitFor();

  await context.storageState({ path: storagePath });
  await context.close();
}

async function globalSetup() {
  const browser = await chromium.launch();
  await loginAndSaveState(browser, users[Role.ProxyAdmin], ADMIN_STORAGE_PATH);
  await loginAndSaveState(browser, users[Role.InternalUserViewer], INTERNAL_USER_VIEWER_STORAGE_PATH);
  await browser.close();
}

export default globalSetup;
