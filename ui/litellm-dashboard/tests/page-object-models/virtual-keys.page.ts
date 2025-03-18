import type { Page, Locator } from "@playwright/test";

export class VirtualKeysPage {
  private readonly userButton: Locator;
  private readonly logoutButton: Locator;

  constructor(public readonly page: Page) {
    this.userButton = this.page.getByRole("button", { name: "User" });
    this.logoutButton = this.page.getByText("Logout");
  }

  async logout() {
    await this.userButton.click();
    await this.logoutButton.click();
  }

  async getUserButton() {
    return this.userButton;
  }
}
