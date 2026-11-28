import { expect, test, type Locator, type Page } from "@playwright/test";
import { ADMIN_STORAGE_PATH } from "../../constants";
import { masterKey, rootPath } from "../../helpers/traffic";

const appId = "e2e-app-forms";
const appName = "E2E Forms App";
const app = `/management/v1/apps/${appId}`;
const definition = "/management/v1/apps/definition";
const customer = "e2e-app-forms-customer";
const member = "e2e-app-forms-member";
const organizationUsage = `/customers/usage?org=${customer}`;
const memberScope = `org=${customer}&team=e2e-app-forms-customer-team&member=${member}`;

type Method = "POST" | "PATCH" | "PUT";

async function save(page: Page, control: Locator, method: Method, endpoint: string, body: object) {
  const [response] = await Promise.all([
    page.waitForResponse((res) =>
      res.request().method() === method && new URL(res.url()).pathname === `${rootPath()}${endpoint}`,
    ),
    control.click(),
  ]);
  expect(response.request().postData()).toBe(JSON.stringify(body));
  expect(response.ok(), `${method} ${endpoint} returned ${response.status()}: ${await response.text()}`).toBe(true);
}

async function restore(page: Page, method: Method, endpoint: string, body: object) {
  const response = await page.request.fetch(`${rootPath()}${endpoint}`, {
    method,
    headers: { Authorization: `Bearer ${masterKey()}` },
    data: body,
  });
  expect(response.ok(), `Restore ${endpoint} returned ${response.status()}: ${await response.text()}`).toBe(true);
}

const go = (page: Page, route: string) => page.goto(`${rootPath()}/ui${route}`);
const button = (scope: Page | Locator, name: string) => scope.getByRole("button", { name, exact: true });
const surcharge = (scope: Page | Locator) => scope.getByLabel("Surcharge on customer requests (%)");

test.describe("App settings and access forms", () => {
  test.use({ storageState: ADMIN_STORAGE_PATH });
  test.describe.configure({ mode: "default" });

  const grantees = [
    { name: "Team", option: "E2E Forms App e2e-app-forms-team", method: "POST", endpoint: "/team/update", identity: { team_id: "e2e-app-forms-team" } },
    { name: "User", option: "E2E Forms Grantee (e2e-app-forms-user)", method: "POST", endpoint: "/user/update", identity: { user_id: "e2e-app-forms-user" } },
    { name: "Organization", option: "E2E Forms Organization e2e-app-forms-org", method: "PATCH", endpoint: "/organization/update", identity: { organization_id: "e2e-app-forms-org" } },
  ] as const;

  for (const { name, option, method, endpoint, identity } of grantees) {
    test(`clears the chosen ${name.toLowerCase()}'s app grants`, async ({ page }) => {
      await go(page, "/customers/access");
      await page.getByRole("tab", { name, exact: true }).click();
      await page.getByRole("combobox", { name, exact: true }).click();
      await page.getByRole("option", { name: option, exact: true }).click();
      const clear = button(page, "Clear all");
      await expect(clear).toBeVisible();
      try {
        await clear.click();
        await save(page, button(page, "Save access"), method, endpoint, { ...identity, object_permission: { apps: [] } });
        await expect(page.getByText("Access updated", { exact: true })).toBeVisible();
      } finally {
        await restore(page, method, endpoint, { ...identity, object_permission: { apps: [appId] } });
      }
    });
  }

  test("toggling customer creation without an app preserves the saved resource type", async ({ page }) => {
    await go(page, "/apps");
    const section = page.getByRole("region", { name: "Define apps for your Gateway" });
    const checkbox = section.getByRole("checkbox", { name: "Requests with no app may create new customers" });
    await expect(checkbox).toBeChecked();
    try {
      await save(page, checkbox, "PUT", definition, { resource_type: "team", auto_create_customers_without_app: false });
      await expect(checkbox).not.toBeChecked();
    } finally {
      await restore(page, "PUT", definition, { resource_type: "team", auto_create_customers_without_app: true });
    }
  });

  const settingsSites = [
    { name: "app settings page", route: `/apps/settings?app_id=${appId}`, inline: true },
    { name: "applications list", route: "/apps", inline: false },
    { name: "app group's Usage tab", route: "/app-groups?app_group=e2e-app-forms-group&group_tab=usage", inline: false },
    { name: "app group's Apps tab", route: "/app-groups?app_group=e2e-app-forms-group&group_tab=apps", inline: false },
  ] as const;

  for (const { name, route, inline } of settingsSites) {
    test(`clearing the surcharge from the ${name} saves a null surcharge without changing app group`, async ({ page }) => {
      await go(page, route);
      if (!inline) await button(page, `Edit ${appName}`).click();
      const form = inline ? page : page.getByRole("dialog");
      await expect(surcharge(form)).toHaveValue("10");
      try {
        await surcharge(form).fill("");
        await save(page, button(form, "Save app settings"), "PATCH", app, {
          budget_id: null,
          auto_create_customers: true,
          customer_margin: null,
        });
        await expect(page.getByText("App settings saved.", { exact: true })).toBeVisible();
      } finally {
        await restore(page, "PATCH", app, { customer_margin: 0.10 });
      }
    });
  }

  test("an out-of-range surcharge shows a validation error without saving", async ({ page }) => {
    await go(page, `/apps/settings?app_id=${appId}`);
    const writes: string[] = [];
    page.on("request", (req) => {
      if (req.method() === "PATCH" && new URL(req.url()).pathname === `${rootPath()}${app}`) {
        writes.push(req.url());
      }
    });
    try {
      await surcharge(page).fill("1001");
      await button(page, "Save app settings").click();
      await expect(page.getByRole("alert").filter({ hasText: "Enter a surcharge between 0 and 1000 percent, or leave it empty." })).toBeVisible();
      expect(writes).toEqual([]);
    } finally {
      if (writes.length) await restore(page, "PATCH", app, { customer_margin: 0.10 });
    }
  });

  const budgetSites = [
    {
      name: "customer usage organization budget",
      route: organizationUsage,
      edit: `Edit budget for ${appName}`,
      saveLabel: "Save app budget",
      endpoint: `${app}/customer_organizations/${customer}`,
    },
    {
      name: "app usage organization budget",
      route: `/apps/usage?app_id=${appId}&from=2026-01-01&to=2026-01-31`,
      edit: "Edit budget for E2E Forms Customer",
      saveLabel: "Save app budget",
      endpoint: `${app}/customer_organizations/${customer}`,
    },
    ...["usage", "manage"].map((site) => ({
      name: `customer ${site} member budget`,
      route: `/customers/${site}?${memberScope}`,
      edit: `Edit member budget for ${appName}`,
      saveLabel: "Save member app budget",
      endpoint: `${app}/customer_members/${member}`,
    })),
  ];

  for (const { name, route, edit, saveLabel, endpoint } of budgetSites) {
    test(`updates the existing ${name}`, async ({ page }) => {
      await go(page, route);
      await button(page, edit).click();
      const dialog = page.getByRole("dialog");
      const limit = dialog.getByLabel("Max Budget (USD)", { exact: true });
      await expect(limit).toHaveValue("50");
      try {
        await limit.fill("123");
        await save(page, button(dialog, saveLabel), "PUT", endpoint, { budget: { max_budget: 123, budget_duration: null } });
        await expect(dialog).toBeHidden();
      } finally {
        await restore(page, "PUT", endpoint, { budget: { max_budget: 50, budget_duration: null } });
      }
    });
  }
});
