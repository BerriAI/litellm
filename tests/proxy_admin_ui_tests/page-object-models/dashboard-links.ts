import { Page, Locator, expect } from "@playwright/test";

export class DashboardLinks {
  private readonly userButton: Locator;
  private readonly logoutButton: Locator;
  private readonly modelsPageLink: Locator;

  constructor(private readonly page: Page) {
    this.userButton = this.page.getByRole("button", { name: "User" });
    this.logoutButton = this.page.getByText("Logout");
    this.modelsPageLink = this.page.getByRole("menuitem", {
      name: "block Models",
    });
  }

  async logout() {
    await this.userButton.click();
    await this.logoutButton.click();
  }

  getUserButton(): Locator {
    return this.userButton;
  }

  getModelsPageLink(): Locator {
    return this.modelsPageLink;
  }

  verifyLogout() {
    expect(this.page.getByRole("heading", { name: "Login" })).toBeVisible();
  }
}
