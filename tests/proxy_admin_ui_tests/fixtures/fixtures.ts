import { DashboardLinks } from "./../page-object-models/dashboard-links";
import { VirtualKeysPage } from "./../page-object-models/virtual-keys.page";
import { test as base } from "@playwright/test";
import { LoginPage } from "../page-object-models/login.page";
import { ModelsPage } from "../page-object-models/models.page";
import { TeamsPage } from "../page-object-models/teams.page";
import { InternalUsersPage } from "../page-object-models/internal-users.page";

type Fixtures = {
  loginPage: LoginPage;
  dashboardLinks: DashboardLinks;
  virtualKeysPage: VirtualKeysPage;
  modelsPage: ModelsPage;
  teamsPage: TeamsPage;
  internalUsersPage: InternalUsersPage;
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
  teamsPage: async ({ page }, use) => {
    await use(new TeamsPage(page));
  },
  internalUsersPage: async ({ page }, use) => {
    await use(new InternalUsersPage(page));
  },
});
export { expect } from "@playwright/test";
