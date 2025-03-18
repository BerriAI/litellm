import type { Page, Locator } from "@playwright/test";

export class LoginPage {
  //Locators as fields
  private readonly usernameInput: Locator;
  private readonly passwordInput: Locator;
  private readonly loginSubmit: Locator;

  //Initialize locators in constructor
  constructor(public readonly page: Page) {
    this.usernameInput = this.page.getByRole("textbox", { name: "Username:" });
    this.passwordInput = this.page.getByRole("textbox", { name: "Password:" });
    this.loginSubmit = this.page.getByRole("button", { name: "Submit" });
  }

  async goto() {
    await this.page.goto("/ui");
  }

  async login(username: string, password: string) {
    await this.usernameInput.click();
    await this.usernameInput.fill(username);
    await this.passwordInput.click();
    await this.passwordInput.fill(password);
    await this.loginSubmit.click();
  }
}
