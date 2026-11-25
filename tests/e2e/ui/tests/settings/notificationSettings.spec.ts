import {
  expect,
  test,
  type APIRequestContext,
  type Page as BrowserPage,
} from "@playwright/test";
import { ADMIN_STORAGE_PATH } from "../../constants";
import { Page } from "../../fixtures/pages";
import { navigateToPage } from "../../helpers/navigation";
import { proxyIsPremium } from "../../helpers/premium";
import { masterKey } from "../../helpers/traffic";

test.describe.configure({ mode: "serial" });
test.use({ storageState: ADMIN_STORAGE_PATH });

const auth = () => ({ Authorization: `Bearer ${masterKey()}` });

async function api<T>(
  request: APIRequestContext,
  method: string,
  path: string,
  data?: object,
): Promise<T> {
  const response = await request.fetch(path, { method, headers: auth(), data });
  expect(response.ok(), `${method} ${path}: HTTP ${response.status()}`).toBe(
    true,
  );
  return (await response.json()) as T;
}

async function captureWrite(
  page: BrowserPage,
  method: string,
  path: string,
  action: () => Promise<void>,
) {
  const [request] = await Promise.all([
    page.waitForRequest(
      (candidate) =>
        candidate.method() === method &&
        new URL(candidate.url()).pathname === path,
    ),
    action(),
  ]);
  return request.postDataJSON();
}

async function openSettingsTab(page: BrowserPage, tab: string) {
  await navigateToPage(page, Page.LoggingAndAlerts);
  await page.getByRole("tab", { name: tab }).click();
  return page.getByRole("tabpanel", { name: tab });
}

const stubSuccess = (page: BrowserPage, path: string) =>
  page.route(`**${path}`, (route) =>
    route.request().method() === "POST"
      ? route.fulfill({ json: { message: "ok" } })
      : route.fallback(),
  );

test.describe("Performance alert thresholds", () => {
  const thresholds = {
    spend: { multiplier: 3, minimum_increase: 50 },
    traffic: { multiplier: 10, minimum_increase: 1000 },
    latency: { multiplier: 3, minimum_increase: 500 },
    reliability: { multiplier: 3, minimum_increase: 0.05 },
  };
  const seed = (request: APIRequestContext) =>
    api(request, "PATCH", "/alerts/monitoring", { thresholds });

  test.beforeEach(async ({ request }) => {
    test.skip(
      !proxyIsPremium(),
      "proxy under test is unlicensed, so editing thresholds is enterprise-gated",
    );
    await seed(request);
  });
  test.afterEach(async ({ request }) => {
    if (proxyIsPremium()) await seed(request);
  });

  test("saves edited thresholds with the error rate sent as a fraction", async ({
    page,
  }) => {
    await page.goto("/ui/alerts");
    await page.getByRole("button", { name: "Manage alerts" }).click();
    const dialog = page.getByRole("dialog");
    await dialog.getByRole("button", { name: "Edit thresholds" }).click();
    const spend = dialog.getByRole("group", { name: "Spend" });
    const reliability = dialog.getByRole("group", { name: "Error rate" });
    await spend.getByLabel("Times the usual value").fill("2.5");
    await reliability
      .getByLabel("Minimum increase (percentage points)")
      .fill("7");
    const body = await captureWrite(page, "PATCH", "/alerts/monitoring", () =>
      dialog.getByRole("button", { name: "Save thresholds" }).click(),
    );
    expect(body).toEqual({
      thresholds: {
        ...thresholds,
        spend: { multiplier: 2.5, minimum_increase: 50 },
        reliability: { multiplier: 3, minimum_increase: 0.07 },
      },
    });
    await expect(dialog.getByRole("status")).toHaveText("Thresholds saved.");
    await expect(
      dialog.getByText("At least 2.5× the usual spend and $50 higher."),
    ).toBeVisible();
    await expect(
      dialog.getByText(
        "At least 3× the usual error rate and 7 percentage points higher.",
      ),
    ).toBeVisible();
  });

  test("a multiplier of 1 is rejected without saving", async ({ page }) => {
    await page.goto("/ui/alerts");
    await page.getByRole("button", { name: "Manage alerts" }).click();
    const dialog = page.getByRole("dialog");
    await dialog.getByRole("button", { name: "Edit thresholds" }).click();
    const multiplier = dialog
      .getByRole("group", { name: "Traffic" })
      .getByLabel("Times the usual value");
    let saved = false;
    page.on("request", (request) => {
      if (
        request.method() === "PATCH" &&
        new URL(request.url()).pathname === "/alerts/monitoring"
      )
        saved = true;
    });
    await multiplier.fill("1");
    await dialog.getByRole("button", { name: "Save thresholds" }).click();
    await expect(multiplier).toHaveJSProperty(
      "validationMessage",
      "Use a multiplier greater than 1.",
    );
    await expect(
      dialog.getByRole("button", { name: "Save thresholds" }),
    ).toBeVisible();
    expect(saved).toBe(false);
  });
});

test.describe("Alert destinations", () => {
  test("MS Teams save adds ms_teams to the destinations and sends only the edited webhook", async ({
    page,
    request,
  }) => {
    const panel = await openSettingsTab(page, "MS Teams Alerts");
    const webhook = panel.locator('input[name="MS_TEAMS_WEBHOOK_URL"]');
    await expect(webhook).toBeVisible();
    const config = await api<{
      alerts?: { name: string }[];
      general_settings?: { alerting?: string[] };
    }>(request, "GET", "/get/config/callbacks");
    const current = config.general_settings?.alerting ?? [];
    await stubSuccess(page, "/config/update");
    await webhook.fill("https://example.test/teams-hook");
    const body = await captureWrite(page, "POST", "/config/update", () =>
      panel.getByRole("button", { name: "Save Changes" }).click(),
    );
    expect(body).toEqual({
      general_settings: { alerting: [...new Set([...current, "ms_teams"])] },
      environment_variables: {
        MS_TEAMS_WEBHOOK_URL: "https://example.test/teams-hook",
      },
    });
    await expect(
      page.getByText("MS Teams settings updated successfully"),
    ).toBeVisible();
  });

  test("email save sends only the edited SMTP fields", async ({ page }) => {
    const panel = await openSettingsTab(page, "Email Alerts");
    const port = panel.locator('input[name="SMTP_PORT"]');
    await expect(port).toBeVisible();
    await stubSuccess(page, "/config/update");
    await port.fill("2525");
    await panel
      .locator('input[name="SMTP_SENDER_EMAIL"]')
      .fill("alerts@example.test");
    const body = await captureWrite(page, "POST", "/config/update", () =>
      panel.getByRole("button", { name: "Save Changes" }).last().click(),
    );
    expect(body).toEqual({
      general_settings: { alerting: ["email"] },
      environment_variables: {
        SMTP_PORT: "2525",
        SMTP_SENDER_EMAIL: "alerts@example.test",
      },
    });
    await expect(
      page.getByText("Email settings updated successfully"),
    ).toBeVisible();
  });
});

test.describe("Email notification events", () => {
  type EventSettings = { settings: { event: string; enabled: boolean }[] };
  test.afterEach(async ({ request }) => {
    await api(request, "POST", "/email/event_settings/reset");
  });

  test("saves every event with the toggled one flipped", async ({
    page,
    request,
  }) => {
    await api(request, "POST", "/email/event_settings/reset");
    const before = await api<EventSettings>(
      request,
      "GET",
      "/email/event_settings",
    );
    const target = before.settings[0];
    const panel = await openSettingsTab(page, "Email Alerts");
    const toggle = panel.getByRole("switch", {
      name: target.event,
      exact: true,
    });
    await expect(toggle).toHaveAttribute(
      "aria-checked",
      String(target.enabled),
    );
    await toggle.click();
    const body = await captureWrite(
      page,
      "PATCH",
      "/email/event_settings",
      () => panel.getByRole("button", { name: "Save Changes" }).first().click(),
    );
    expect(body).toEqual({
      settings: before.settings.map((setting) =>
        setting.event === target.event
          ? { ...setting, enabled: !setting.enabled }
          : setting,
      ),
    });
    await expect(
      page.getByText("Email event settings updated successfully"),
    ).toBeVisible();
  });
});

test.describe("Alerting settings", () => {
  test("submits every alerting argument with the edited value", async ({
    page,
    request,
  }) => {
    const settings = await api<{ field_name: string; field_value: unknown }[]>(
      request,
      "GET",
      "/alerting/settings",
    );
    const panel = await openSettingsTab(page, "Alerting Settings");
    const frequency = panel
      .getByRole("row", { name: /^daily_report_frequency/ })
      .getByRole("spinbutton");
    await expect(frequency).toBeVisible();
    await page.route("**/config/field/update", (route) =>
      route.fulfill({ json: { message: "ok" } }),
    );
    await frequency.fill("7200");
    const body = await captureWrite(page, "POST", "/config/field/update", () =>
      panel.getByRole("button", { name: "Update Settings" }).click(),
    );
    const { slack_alerting: _slack, ...current } = Object.fromEntries(
      settings.map((setting) => [setting.field_name, setting.field_value]),
    );
    expect(body).toEqual({
      field_name: "alerting_args",
      field_value: { ...current, daily_report_frequency: 7200 },
      config_type: "general_settings",
    });
  });
});
