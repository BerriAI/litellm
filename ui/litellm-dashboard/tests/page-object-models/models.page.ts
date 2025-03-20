import { Page, Locator } from "@playwright/test";

export class ModelsPage {
  // Models Page Tabs
  private readonly allModelsTab: Locator;
  private readonly addModelTab: Locator;
  // All Models Tab Locators
  // Add Model Tab Form Locators
  private readonly providerCombobox: Locator;
  //  *private openaiProviderComboboxOption: Locator;
  private readonly litellmModelNameCombobox: Locator;
  //   *private readonly litellmModelNameComboboxOption page.getByTitle('omni-moderation-latest', { exact: true }).locator('div')
  /*private readonly modelMappingPublicNameInput: Locator;*/
  private readonly apiKeyInput: Locator;
  private readonly addModelSubmitButton: Locator;

  constructor(private readonly page: Page) {
    // Models Page Tabs
    this.allModelsTab = this.page.getByRole("tab", { name: "All Models" });
    this.addModelTab = this.page.getByRole("tab", { name: "Add Model" });
    // Add Model Tab Form Locators
    this.providerCombobox = this.page.getByRole("combobox", {
      name: "* Provider question-circle :",
    });
    /**this.openaiProviderComboboxOption = this.page
      .locator("span")
      .filter({ hasText: "OpenAI" });*/
    this.litellmModelNameCombobox = this.page.locator(
      ".ant-select-selection-overflow"
    );
    /*this.modelMappingPublicNameInput = this.page
      .getByRole("row", { name: "omni-moderation-latest omni-" })
      .getByTestId("base-input");*/
    this.apiKeyInput = page.getByRole("textbox", {
      name: "* API Key question-circle :",
    });
    this.addModelSubmitButton = page.getByRole("button", { name: "Add Model" });
  }

  // 'All Model' Tab //
  getAllModelsTab(): Locator {
    return this.allModelsTab;
  }

  // Parametized Locators
  getAllModelsTableCellValue(allModelsTableCellValue: string): Locator {
    return this.page.getByRole("cell", { name: allModelsTableCellValue }); //.first();
  }

  // 'Add Model' Tab //
  getAddModelTab(): Locator {
    return this.addModelTab;
  }

  // Parametized Form Locators
  getProviderComboboxOption(providerComboboxOption: string): Locator {
    return this.page
      .locator("span")
      .filter({ hasText: providerComboboxOption });
  }

  getLitellmModelNameComboboxOption(
    litellmModelNameComboboxOption: string
  ): Locator {
    return this.page
      .getByTitle(litellmModelNameComboboxOption, { exact: true })
      .locator("div");
  }

  getLitellmModelMappingModel(litellmModelMappingModel: string): Locator {
    return this.page
      .locator("#model_mappings")
      .getByText(litellmModelMappingModel);
  }

  getLitellmModelMappingModelPublicName(
    litellmModelMappingModel: string
  ): Locator {
    return this.page
      .getByRole("row", { name: litellmModelMappingModel })
      .getByTestId("base-input");
  }

  // Non-parametized Form Locators
  getProviderCombobox(): Locator {
    return this.providerCombobox;
  }

  getLitellModelNameCombobox(): Locator {
    return this.litellmModelNameCombobox;
  }

  getAPIKeyInputBox(): Locator {
    return this.apiKeyInput;
  }

  getAddModelSubmitButton(): Locator {
    return this.addModelSubmitButton;
  }
}
