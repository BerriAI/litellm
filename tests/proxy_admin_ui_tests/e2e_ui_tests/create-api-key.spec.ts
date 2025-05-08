import { test, expect } from "./../fixtures/fixtures";
import { loginDetailsSet } from "./../utils/utils";

test("4644_Test_Creating_An_API_Key_for_Self_for_All_Team_Models", async ({
  loginPage,
  virtualKeysPage,
  page,
}) => {
  let username = "admin";
  let password = "sk-1234";
  // let apiKey = "";
  let apiKeyID = "";
  const keyName = "test-key-name-3";

  if (loginDetailsSet()) {
    username = process.env.UI_USERNAME as string;
    password = process.env.UI_PASSWORD as string;
  }

  // console.log("1. Navigating to 'Login' page and logging in");
  await loginPage.goto();
  // await page.screenshot({path: `./test-results/4644_test_create_api_key/00_go-to-login-page.png`,});

  await loginPage.login(username, password);
  await expect(page.getByRole("button", { name: "User" })).toBeVisible();
  // await page.screenshot({path: `./test-results/4644_test_create_api_key/01_dashboard.png`,});

  // console.log("2. Clicking the '+ Create New Key' button");
  await virtualKeysPage.getCreateNewKeyButton().click();

  // console.log("3. Clicking the Owned By You radio button.");
  await virtualKeysPage.getOwnedByYouRadioButton().check();

  // console.log("4. Entering a key name in the 'Key Name' input.");
  await virtualKeysPage.getKeyNameInput().fill(keyName);
  // await page.waitForTimeout(500);
  // await page.screenshot({path: `./test-results/4644_test_create_api_key/04_enter-key-name.png`,});

  // console.log("5. Selecting All Team Models");
  await virtualKeysPage.getModelInput().click();
  await page.getByText("All Team Models").click();
  // await page.waitForTimeout(500);
  // await page.screenshot({path: `./test-results/4644_test_create_api_key/05_select-all-team-models.png`,});

  // console.log("6. Clicking 'Create Key'");
  await virtualKeysPage.getCreateKeyButton().click();
  // await page.waitForTimeout(500);
  // await page.screenshot({path: `./test-results/4644_test_create_api_key/06_create-api-key.png`,});

  // console.log("7. Copying the API key to clipboard");
  /*
  await virtualKeysPage.getCopyAPIKeyButton().click();
  apiKey = await page.evaluate(async () => {
    return await navigator.clipboard.readText();
  });
  */

  // console.log("8. Exiting Modal Window");

  await page.waitForTimeout(500);
  await page
    .getByRole("dialog")
    .filter({ hasText: "Save your KeyPlease save this" })
    .getByLabel("Close", { exact: true })
    .click();
  // await page.keyboard.press("Escape");
  await expect(
    virtualKeysPage.getVirtualKeysTableCellValue(keyName)
  ).toBeVisible();
  // await page.waitForTimeout(500);
  // await page.screenshot({path: `./test-results/4644_test_create_api_key/08_check-api-key-created.png`,});

  apiKeyID = await page
    .locator("tr.tremor-TableRow-row.h-8")
    .locator(".tremor-Button-text.text-tremor-default")
    .innerText();
  await page.getByRole("button", { name: apiKeyID }).click();
  await page.getByRole("button", { name: "Delete Key" }).click();
  await page.getByRole("button", { name: "Delete", exact: true }).click();

  // console.log("9. Logging out");
  // await page.getByRole("button", { name: "User" }).click();
  // await page.getByText("Logout").click();
  // await expect(page.getByRole("heading", { name: "Login" })).toBeVisible();
});
