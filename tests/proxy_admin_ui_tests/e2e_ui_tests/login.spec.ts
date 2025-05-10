import { VirtualKeysPage } from "../page-object-models/virtual-keys.page";
import { test, expect } from "./../fixtures/fixtures";
import { loginDetailsSet } from "./../utils/utils";

test("4644_Test_Basic_Sign_in_Flow", async ({
  loginPage,
  virtualKeysPage,
  page,
}) => {
  let username = "admin";
  let password = "sk-1234";
  if (loginDetailsSet()) {
    // console.log("Login details exist in .env file.");
    username = process.env.UI_USERNAME as string;
    password = process.env.UI_PASSWORD as string;
  }
  await loginPage.goto();
  // await page.screenshot({path: "./test-results/4644_Test_Basic_Sign_in_Flow/navigate-to-login-page.png",});

  await loginPage.login(username, password);
  await expect(page.getByRole("button", { name: "User" })).toBeVisible();
  // await page.screenshot({path: "./test-results/4644_Test_Basic_Sign_in_Flow/dashboard.png",});

  // console.log("Logging out");
  await page.getByRole("button", { name: "User" }).click();
  await page.getByText("Logout").click();
  await page.waitForTimeout(500);
  await expect(page.getByRole("heading", { name: "Login" })).toBeVisible();
  // await page.screenshot({path: "./test-results/4644_Test_Basic_Sign_in_Flow/logout.png",});
});
