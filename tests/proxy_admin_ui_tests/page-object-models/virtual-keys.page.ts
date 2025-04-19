import type { Page, Locator } from "@playwright/test";

export class VirtualKeysPage {
  private readonly userButton: Locator;
  private readonly logoutButton: Locator;
  private readonly createNewKeyButton: Locator;
  private readonly ownedByYouRadioButton: Locator;
  private readonly keyNameInput: Locator;
  private readonly modelInput: Locator;
  private readonly createKeyButton: Locator;
  private readonly copyAPIKeyButton: Locator;

  constructor(private readonly page: Page) {
    this.userButton = this.page.getByRole("button", { name: "User" });
    this.logoutButton = this.page.getByText("Logout");
    this.createNewKeyButton = this.page.getByRole("button", {
      name: "+ Create New Key",
    });
    this.ownedByYouRadioButton = this.page.getByRole("radio", { name: "You" });
    this.keyNameInput = this.page.getByTestId("base-input");
    this.modelInput = this.page.locator(".ant-select-selection-overflow");
    this.createKeyButton = this.page.getByRole("button", {
      name: "Create Key",
    });
    this.copyAPIKeyButton = this.page.getByRole("button", {
      name: "Copy API Key",
    });
  }

  async logout() {
    await this.userButton.click();
    await this.logoutButton.click();
  }

  getUserButton(): Locator {
    return this.userButton;
  }

  getCreateNewKeyButton(): Locator {
    return this.createNewKeyButton;
  }

  getVirtualKeysTableCellValue(virtualKeysTableCellValue: string): Locator {
    return this.page.getByRole("cell", { name: virtualKeysTableCellValue });
  }

  getOwnedByYouRadioButton(): Locator {
    return this.ownedByYouRadioButton;
  }

  getKeyNameInput(): Locator {
    return this.keyNameInput;
  }

  getModelInput(): Locator {
    return this.modelInput;
  }

  getCreateKeyButton(): Locator {
    return this.createKeyButton;
  }

  getCopyAPIKeyButton(): Locator {
    return this.copyAPIKeyButton;
  }
}
