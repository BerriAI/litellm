import { chromium } from "@playwright/test";
import { users } from "./fixtures/users";
import { Role } from "./fixtures/roles";

async function globalSetup() {
  const browser = await chromium.launch();
  const page = await browser.newPage();
  await page.goto("http://localhost:4000/ui/login");
  await page.getByPlaceholder("Enter your username").fill(users[Role.ProxyAdmin].email);
  await page.getByPlaceholder("Enter your password").fill(users[Role.ProxyAdmin].password);
  const loginButton = page.getByRole("button", { name: "Login", exact: true });
  await loginButton.click();
  await page.waitForSelector("text=Virtual Keys");
  await page.context().storageState({ path: "admin.storageState.json" });
  await browser.close();
}

export default globalSetup;
