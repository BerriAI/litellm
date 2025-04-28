import type { Page, Locator } from "@playwright/test";

export class LoginPage {
  //Locators as fields
  private readonly usernameInput: Locator;
  private readonly passwordInput: Locator;
  private readonly loginSubmit: Locator;

  //Initialize locators in constructor
  constructor(private readonly page: Page) {
    this.usernameInput = this.page.getByPlaceholder("Enter your username");
    this.passwordInput = this.page.getByPlaceholder("Enter your password");
    this.loginSubmit = this.page.getByRole("button", { name: "Login" });
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
