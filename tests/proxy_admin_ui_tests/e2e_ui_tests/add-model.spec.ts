import { test, expect } from "./../fixtures/fixtures";
import { loginDetailsSet } from "./../utils/utils";
const providersAndModels = JSON.parse(
  JSON.stringify(require("./../data/providers-and-models.json"))
);

providersAndModels["OpenAI"].forEach((model: string) => {
  test(`4644_Test_Adding_OpenAI's_${model}_model`, async ({
    loginPage,
    dashboardLinks,
    modelsPage,
    page,
  }) => {
    const excludeLitellmModelNameDropdownValues = [
      "Custom Model Name (Enter below)",
      "All OpenAI Models (Wildcard)",
    ];
    if (!excludeLitellmModelNameDropdownValues.includes(model)) {
      let username = "admin";
      let password = "sk-1234";
      if (loginDetailsSet()) {
        // console.log("Login details exist in .env file.");
        username = process.env.UI_USERNAME as string;
        password = process.env.UI_PASSWORD as string;
      }

      // console.log("1. Navigating to 'Login' page and logging in");

      await loginPage.goto();
      // await page.screenshot({path: `./test-results/4644_test_adding_a_model/openai/${model}/00_go-to-login-page.png`,});

      await loginPage.login(username, password);
      await expect(page.getByRole("button", { name: "User" })).toBeVisible();
      // await page.screenshot({path: `./test-results/4644_test_adding_a_model/openai/${model}/01_dashboard.png`,});

      // console.log("2. Navigating to 'Models' page");
      await dashboardLinks.getModelsPageLink().click();
      // await page.screenshot({path: `./test-results/4644_test_adding_a_model/openai/${model}/02_navigate-to-models-page.png`,});

      // console.log("3. Selecting 'Add Model' in the header of 'Models' page");
      await modelsPage.getAddModelTab().click();
      // await page.screenshot({path: `./test-results/4644_test_adding_a_model/openai/${model}/03_navigate-to-add-models-tab.png`,});

      // console.log("4. Selecting OpenAI from 'Provider' dropdown");
      await modelsPage.getProviderCombobox().click();
      modelsPage.fillProviderComboboxBox("OpenAI");
      await modelsPage.getProviderCombobox().press("Enter");
      // await page.screenshot({path: `./test-results/4644_test_adding_a_model/openai/${model}/04_select-openai-provider.png`,});

      // console.log("5. Selecting model name from 'LiteLLM Model Name(s)' dropdown");
      await modelsPage.getLitellModelNameCombobox().click();
      await modelsPage.getLitellModelNameCombobox().fill(model);
      await modelsPage.getLitellModelNameCombobox().press("Enter");
      await expect(modelsPage.getLitellmModelMappingModel(model)).toBeVisible();
      // await page.screenshot({path: `./test-results/4644_test_adding_a_model/openai/${model}/05_select-${model}-model.png`,});

      // console.log("6. Adding API Key");
      await modelsPage.getAPIKeyInputBox("OpenAI").click();
      await modelsPage.getAPIKeyInputBox("OpenAI").fill("sk-1234");
      // await page.screenshot({path: `./test-results/4644_test_adding_a_model/openai/${model}/06_enter-api-key.png`,});

      // console.log("7. Clicking 'Add Model'");
      await modelsPage.getAddModelSubmitButton().click();
      // await page.screenshot({path: `./test-results/4644_test_adding_a_model/openai/${model}/07_add-model.png`,});

      // console.log("8. Navigating to 'All Models'");
      if (model.length > 20) {
        // model = model.slice(0, 20) + "...";
      }
      await modelsPage.getAllModelsTab().click();
      await expect(
        modelsPage.getAllModelsTableCellValue(`openai logo openai`)
      ).toBeVisible();
      await expect(modelsPage.getAllModelsTableCellValue(model)).toBeVisible();
      // await page.screenshot({path: `./test-results/4644_test_adding_a_model/openai/${model}/08_navigate-to-all-models-tab.png`,});

      // console.log("Clean Up - Delete Model Created");
      const modelID = await page
        .locator("tr.tremor-TableRow-row")
        .nth(2)
        .locator("td")
        .first()
        .locator("div")
        .innerText();

      await page.getByRole("cell", { name: modelID }).locator("div").click();
      await page.getByRole("button", { name: "Delete Model" }).click();
      await page.getByRole("button", { name: "Delete", exact: true }).click();

      // await page.screenshot({path: `./test-results/4644_test_adding_a_model/openai/${model}/09_logout.png`,});
    }

    // // console.log("9. Logging out");
    // await page.getByRole("button", { name: "User" }).click();
    // await page.getByText("Logout").click();
    // await expect(page.getByRole("heading", { name: "Login" })).toBeVisible();
  });
});

Object.entries(providersAndModels).forEach(([provider, model]) => {
  test(`4644_Test_the_Correct_Dropdown_Shows_When_Adding_${provider}_Models`, async ({
    loginPage,
    dashboardLinks,
    modelsPage,
    page,
  }) => {
    let username = "admin";
    let password = "sk-1234";
    const excludeLitellmModelNameDropdownValues = [
      "OpenAI",
      "OpenAI-Compatible Endpoints (Together AI, etc.)",
      "OpenAI Text Completion",
      "OpenAI-Compatible Text Completion Models (Together AI, etc.)",
      "Anthropic",
    ];
    let litellmModelNameDropdownValues: string[] = [];
    if (loginDetailsSet()) {
      // console.log("Login details exist in .env file.");
      username = process.env.UI_USERNAME as string;
      password = process.env.UI_PASSWORD as string;
    }

    // console.log("1. Navigating to 'Login' page and logging in");
    await loginPage.goto();
    // await page.screenshot({path: `./test-results/4644_test_model_dropdown/${model}/00_go-to-login-page.png`,});

    await loginPage.login(username, password);
    await expect(page.getByRole("button", { name: "User" })).toBeVisible();
    // await page.screenshot({path: `./test-results/4644_test_model_dropdown/${model}/01_dashboard.png`,});

    // console.log("2. Navigating to 'Models' page");
    await dashboardLinks.getModelsPageLink().click();
    // await page.screenshot({path: `./test-results/4644_test_model_dropdown/${model}/02_navigate-to-models-page.png`,});

    // console.log("3. Selecting 'Add Model' in the header of 'Models' page");
    await modelsPage.getAddModelTab().click();
    // await page.screenshot({path: `./test-results/4644_test_model_dropdown/${model}/03_navigate-to-add-models-tab.png`,});

    // console.log(`4. Selecting ${model} from 'Provider' dropdown`);
    await modelsPage.getProviderCombobox().click();
    modelsPage.fillProviderComboboxBox(provider);
    await modelsPage.getProviderCombobox().press("Enter");
    // await page.screenshot({path: `./test-results/4644_test_model_dropdown/${model}/04_select-openai-provider.png`,});

    //Scrape ant-selection-option and add to list
    await modelsPage.getLitellModelNameCombobox().click();
    const litellmModelOptions = await page
      .locator(".rc-virtual-list-holder-inner")
      .locator(".ant-select-item-option-content")
      .all();

    for (const element of litellmModelOptions) {
      let modelNameDropdownValue = await element.innerText();
      if (
        !excludeLitellmModelNameDropdownValues.includes(modelNameDropdownValue)
      ) {
        litellmModelNameDropdownValues.push(modelNameDropdownValue);
      }
    }

    litellmModelNameDropdownValues.forEach((element) => {
      expect(providersAndModels[provider].includes(element)).toBeTruthy();
    });

    // console.log("5. Logging out");
    // await page.getByRole("button", { name: "User" }).click();
    // await page.getByText("Logout").click();
    // await expect(page.getByRole("heading", { name: "Login" })).toBeVisible();

    // await page.screenshot({path: `./test-results/4644_test_model_dropdown/${model}/05_logout.png`,});
  });
});
