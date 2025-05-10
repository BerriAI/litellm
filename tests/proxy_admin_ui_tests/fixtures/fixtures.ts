import { DashboardLinks } from "./../page-object-models/dashboard-links";
import { VirtualKeysPage } from "./../page-object-models/virtual-keys.page";
import { test as base } from "@playwright/test";
import { LoginPage } from "../page-object-models/login.page";
import { ModelsPage } from "../page-object-models/models.page";

type Fixtures = {
  loginPage: LoginPage;
  dashboardLinks: DashboardLinks;
  virtualKeysPage: VirtualKeysPage;
  modelsPage: ModelsPage;
};

export const test = base.extend<Fixtures>({
  loginPage: async ({ page }, use) => {
    await use(new LoginPage(page));
  },
  dashboardLinks: async ({ page }, use) => {
    await use(new DashboardLinks(page));
  },
  virtualKeysPage: async ({ page }, use) => {
    await use(new VirtualKeysPage(page));
  },
  modelsPage: async ({ page }, use) => {
    await use(new ModelsPage(page));
  },
});
export { expect } from "@playwright/test";
