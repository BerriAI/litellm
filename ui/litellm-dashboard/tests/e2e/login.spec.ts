import { VirtualKeysPage } from "../page-object-models/virtual-keys.page";
import { test, expect } from "./../fixtures/fixtures";
import { loginDetailsSet } from "./../utils/utils";

// import { config } from "dotenv";
// import path from "path";
// config({ path: "./../../../../.env.example" });

test("Login", async ({ loginPage, virtualKeysPage, page }) => {
  // Check if there are credentials in local environment file and assign credentials as appropriate.
  // Login and verify user is directed to the dashboard.
  // Do playwright tests run in order.
  let username = "admin";
  let password = "sk-1234";
  if (loginDetailsSet()) {
    console.log("Login details exist in .env file.");
    username = process.env.UI_USERNAME as string;
    password = process.env.UI_PASSWORD as string;
  }
  await loginPage.goto();
  await page.screenshot({ path: "./test-results/go-to-login-page.png" });

  await loginPage.login(username, password);
  await expect(page.getByRole("button", { name: "User" })).toBeVisible();
  await page.screenshot({ path: "./test-results/dashboard.png" });

  await virtualKeysPage.logout();
  await expect(
    page.getByRole("heading", { name: "LiteLLM Login" })
  ).toBeVisible();
  await page.screenshot({ path: "./test-results/logout.png" });
});

/* to dos
1. Add ui link to local env file
2. add login details to env file
3. configure screenshots and add path to screenshots
*/
