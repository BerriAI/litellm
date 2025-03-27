import { test, expect } from "./../fixtures/fixtures";
import { loginDetailsSet } from "./../utils/utils";

test("4644_Test_Creating_An_API_Key_for_Self_for_All_Team_Models", async ({
  loginPage,
  dashboardLinks,
  virtualKeysPage,
  modelsPage,
  page,
}) => {
  let username = "admin";
  let password = "sk-1234";
  let apiKey = "";
  let apiKeyID = "";
  const keyName = "test-key-name-3";

  if (loginDetailsSet()) {
    console.log("Login details exist in .env file.");
    username = process.env.UI_USERNAME as string;
    password = process.env.UI_PASSWORD as string;
  }

  console.log("1. Navigating to 'Login' page and logging in");
  await loginPage.goto();
  // await page.screenshot({path: `./test-results/4644_test_adding_a_model/openai/${model}/00_go-to-login-page.png`,});

  await loginPage.login(username, password);
  await expect(page.getByRole("button", { name: "User" })).toBeVisible();
  // await page.screenshot({path: `./test-results/4644_test_adding_a_model/openai/${model}/01_dashboard.png`,});
  /*
  console.log("2. Clicking the '+ Create New Key' button");
  await virtualKeysPage.getCreateNewKeyButton().click();

  console.log("3. Clicking the Owned By You radio button.");
  await virtualKeysPage.getOwnedByYouRadioButton().check();

  console.log("4. Entering a key name in the 'Key Name' input.");
  await virtualKeysPage.getKeyNameInput().fill(keyName);
  await page.screenshot({
    path: `./test-results/4644_Test_Creating_An_API_Key_for_Self_for_All_Team_Models/04_enter-key-name.png`,
  });

  console.log("5. Selecting All Team Models");
  await virtualKeysPage.getModelInput().click();
  await page.getByText("All Team Models").click();
  await page.screenshot({
    path: `./test-results/4644_Test_Creating_An_API_Key_for_Self_for_All_Team_Models/05_select-all-team-models.png`,
  });

  console.log("6. Clicking 'Create Key'");
  await virtualKeysPage.getCreateKeyButton().click();
  await page.screenshot({
    path: `./test-results/4644_Test_Creating_An_API_Key_for_Self_for_All_Team_Models/06_create-api-key.png`,
  });

  console.log("7. Copying the API key to clipboard");
  await virtualKeysPage.getCopyAPIKeyButton().click();
  apiKey = await page.evaluate(async () => {
    return await navigator.clipboard.readText();
  });
  console.log("API Key from clipboard: " + apiKey);
  console.log("Sliced API Key from clipboard: " + apiKey.slice(0, 8));
  //   await page.pause();
  //   await page.getByRole("button", { name: "Copy API Key" }).click();

  console.log("8. Exiting Modal Window");
  await page
    .getByRole("dialog")
    .filter({ hasText: "Save your KeyPlease save this" })
    .getByLabel("Close", { exact: true })
    .click();
  await expect(
    virtualKeysPage.getVirtualKeysTableCellValue(keyName)
  ).toBeVisible();
  await page.screenshot({
    path: `./test-results/4644_Test_Creating_An_API_Key_for_Self_for_All_Team_Models/08_checkapi-key-created.png`,
  });*/

  console.log("Delete Generated API key");
  //   await page.getByRole("button", { name: apiKey.slice(0, 8) + "..." }).click();
  console.log(
    await page
      .locator("tr.tremor-TableRow-row.h-8")
      .nth(1)
      .locator(".tremor-Button-text.text-tremor-default")
      .innerText()
  );
  apiKeyID = await page
    .locator("tr.tremor-TableRow-row.h-8")
    .nth(1)
    .locator(".tremor-Button-text.text-tremor-default")
    .innerText();
  /*.evaluate((element) => {
      console.log("element" + element);
    });*/
  await page.getByRole("button", { name: apiKeyID }).click();
  await page.getByRole("button", { name: "Delete Key" }).click();
  await page.getByRole("button", { name: "Delete", exact: true }).click();
});
