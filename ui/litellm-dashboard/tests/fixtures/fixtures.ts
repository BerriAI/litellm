import { VirtualKeysPage } from "./../page-object-models/virtual-keys.page";
import { test as base } from "@playwright/test";
import { LoginPage } from "../page-object-models/login.page";

type Fixtures = {
  loginPage: LoginPage;
  virtualKeysPage: VirtualKeysPage;
};

export const test = base.extend<Fixtures>({
  loginPage: async ({ page }, use) => {
    await use(new LoginPage(page));
  },
  virtualKeysPage: async ({ page }, use) => {
    await use(new VirtualKeysPage(page));
  },
});
export { expect } from "@playwright/test";
