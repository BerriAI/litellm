import {
  expect,
  test,
  type APIRequestContext,
  type Locator,
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

const emailCard = (panel: Locator, title: string) =>
  panel
    .locator('[data-slot="card"]')
    .filter({ has: panel.page().getByText(title, { exact: true }) });

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
    await expect(
      dialog.getByRole("status").filter({ hasText: "Thresholds saved." }),
    ).toBeVisible();
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
      () =>
        emailCard(panel, "Email Notifications")
          .getByRole("button", { name: "Save Changes" })
          .click(),
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
