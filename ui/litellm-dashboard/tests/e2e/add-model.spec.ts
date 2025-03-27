import { test, expect } from "./../fixtures/fixtures";
import { loginDetailsSet } from "./../utils/utils";
const providersAndModels = JSON.parse(
  JSON.stringify(require("./../data/providers-and-models.json"))
);
/*
providersAndModels["OpenAI"].forEach((model: string) => {
  test(`4644_Test_Adding_OpenAI's_${model}_model`, async ({
    loginPage,
    dashboardLinks,
    modelsPage,
    page,
  }) => {
    console.log(model);

    let username = "admin";
    let password = "sk-1234";
    if (loginDetailsSet()) {
      console.log("Login details exist in .env file.");
      username = process.env.UI_USERNAME as string;
      password = process.env.UI_PASSWORD as string;
    }

    // console.log("1. Navigating to 'Login' page and logging in");
    await loginPage.goto();
    await page.screenshot({
      path: `./test-results/4644_test_adding_a_model/openai/${model}/00_go-to-login-page.png`,
    });

    await loginPage.login(username, password);
    await expect(page.getByRole("button", { name: "User" })).toBeVisible();
    await page.screenshot({
      path: `./test-results/4644_test_adding_a_model/openai/${model}/01_dashboard.png`,
    });

    // console.log("2. Navigating to 'Models' page");
    await dashboardLinks.getModelsPageLink().click();
    await page.screenshot({
      path: `./test-results/4644_test_adding_a_model/openai/${model}/02_navigate-to-models-page.png`,
    });

    // console.log("3. Selecting 'Add Model' in the header of 'Models' page");
    await modelsPage.getAddModelTab().click();
    await page.screenshot({
      path: `./test-results/4644_test_adding_a_model/openai/${model}/03_navigate-to-add-models-tab.png`,
    });

    // console.log("4. Selecting OpenAI from 'Provider' dropdown");
    await modelsPage.getProviderCombobox().click();
    modelsPage.fillProviderComboboxBox("OpenAI");
    await modelsPage.getProviderCombobox().press("Enter");
    await page.screenshot({
      path: `./test-results/4644_test_adding_a_model/openai/${model}/04_select-openai-provider.png`,
    });

    // console.log("5. Selecting model name from 'LiteLLM Model Name(s)' dropdown");
    console.log(model);
    await modelsPage.getLitellModelNameCombobox().click();
    await modelsPage.getLitellModelNameCombobox().fill(model);
    await modelsPage.getLitellModelNameCombobox().press("Enter");
    await page.screenshot({
      path: `./test-results/4644_test_adding_a_model/openai/${model}/05_select-${model}-model.png`,
    });
    await expect(modelsPage.getLitellmModelMappingModel(model)).toBeVisible();

    // console.log("6. Adding API Key");
    await modelsPage.getAPIKeyInputBox("OpenAI").click();
    await modelsPage.getAPIKeyInputBox("OpenAI").fill("sk-1234");
    await page.screenshot({
      path: `./test-results/4644_test_adding_a_model/openai/${model}/06_enter-api-key.png`,
    });

    // console.log("7. Clicking 'Add Model'");
    await modelsPage.getAddModelSubmitButton().click();
    await page.screenshot({
      path: `./test-results/4644_test_adding_a_model/openai/${model}/07_add-model.png`,
    });

    // console.log("8. Navigating to 'All Models'");
    await modelsPage.getAllModelsTab().click();
    await page.screenshot({
      path: `./test-results/4644_test_adding_a_model/openai/${model}/08_navigate-to-all-models-tab.png`,
    });
    await expect(
      modelsPage.getAllModelsTableCellValue(`openai logo openai`)
    ).toBeVisible();
    await expect(
      modelsPage.getAllModelsTableCellValue(model.slice(0, 20) + "...")
    ).toBeVisible();

    // console.log("9. Logging out");
    await page.getByRole("link", { name: "LiteLLM Brand" }).click();
    await page.screenshot({
      path: `./test-results/4644_test_adding_a_model/openai/${model}/09_logout.png`,
    });
    
      // await dashboardLinks.logout();
      // await expect(page.getByRole("heading", { name: "LiteLLM Login" })).toBeVisible();
      
  });
});*/

Object.entries(providersAndModels).forEach(([provider, model]) => {
  console.log(`${provider}: ${model}`);
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
    let litellmModelNameDropdownValues = [];
    if (loginDetailsSet()) {
      console.log("Login details exist in .env file.");
      username = process.env.UI_USERNAME as string;
      password = process.env.UI_PASSWORD as string;
    }

    console.log("1. Navigating to 'Login' page and logging in");
    await loginPage.goto();
    await page.screenshot({
      path: `./test-results/4644_test_adding_a_model/openai/${model}/00_go-to-login-page.png`,
    });

    await loginPage.login(username, password);
    await expect(page.getByRole("button", { name: "User" })).toBeVisible();
    await page.screenshot({
      path: `./test-results/4644_test_adding_a_model/openai/${model}/01_dashboard.png`,
    });

    console.log("2. Navigating to 'Models' page");
    await dashboardLinks.getModelsPageLink().click();
    await page.screenshot({
      path: `./test-results/4644_test_adding_a_model/openai/${model}/02_navigate-to-models-page.png`,
    });

    console.log("3. Selecting 'Add Model' in the header of 'Models' page");
    await modelsPage.getAddModelTab().click();
    await page.screenshot({
      path: `./test-results/4644_test_adding_a_model/openai/${model}/03_navigate-to-add-models-tab.png`,
    });

    console.log(`4. Selecting ${model} from 'Provider' dropdown`);
    await modelsPage.getProviderCombobox().click();
    modelsPage.fillProviderComboboxBox(provider);
    await modelsPage.getProviderCombobox().press("Enter");
    await page.screenshot({
      path: `./test-results/4644_test_adding_a_model/openai/${model}/04_select-openai-provider.png`,
    });

    //Scrape ant-selection-option and add to list
    await modelsPage.getLitellModelNameCombobox().click();
    const litellmModelOptions = await page
      .locator(".rc-virtual-list-holder-inner")
      .locator(".ant-select-item-option-content")
      .all();

    for (const element of litellmModelOptions) {
      //excludeLitellmModelNameDropdownValues
      let modelNameDropdownValue = await element.innerText();
      if (
        !excludeLitellmModelNameDropdownValues.includes(modelNameDropdownValue)
      ) {
        litellmModelNameDropdownValues.push(await element.innerText());
      }
    }
    litellmModelNameDropdownValues.forEach((element) => {
      console.log(element);
      expect(providersAndModels[provider].includes(element)).toBeTruthy();
    });
    /*const litellmModelOptions = await page.$$(
      ".ant-select-item-option-content"
    );

    const arrOfmodelnames: any[] = [];

    for (const element of litellmModelOptions) {
      const text = await element.getAttribute("title");
      // console.log((await element.innerHTML()) + "\n");
    }*/
  });
});
