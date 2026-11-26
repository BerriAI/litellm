import { expect, test, type APIRequestContext, type Page as BrowserPage, type Request } from "@playwright/test";
import { ADMIN_STORAGE_PATH, INTERNAL_USER_STORAGE_PATH } from "../../constants";
import { Page } from "../../fixtures/pages";
import { dismissFeedbackPopup, navigateToPage } from "../../helpers/navigation";
import { masterKey } from "../../helpers/traffic";

const banner = "/update/user_banner";
const visibility = "/update/ui_settings";
const theme = "/update/ui_theme_settings";
const telemetry = "/telemetry/settings";

const unpublished = { enabled: false, message: "", severity: "info" };
const unrestricted = { enabled_ui_pages_internal_users: null };
const defaultTheme = { logo_url: null, logo_url_dark: null, favicon_url: null };
const enabledTelemetry = {
  groups: ["heartbeat", "request_success", "token_info", "request_taxonomy", "event_details", "page_navigation"],
  destination: "local_table",
};
const disabledTelemetry = { groups: [], destination: null };

test.describe.configure({ mode: "serial" });
test.use({ storageState: ADMIN_STORAGE_PATH });

async function seed(request: APIRequestContext, path: string, data: object) {
  const method = path === telemetry ? "PUT" : "PATCH";
  const response = await request.fetch(path, { method, headers: { Authorization: `Bearer ${masterKey()}` }, data });
  expect(response.ok(), `Seed ${method} ${path}: HTTP ${response.status()}`).toBe(true);
}

const isWrite = (request: Request, path: string) =>
  request.method() === (path === telemetry ? "PUT" : "PATCH") && new URL(request.url()).pathname === path;

async function submit(page: BrowserPage, path: string, action: () => Promise<void>, body: object, status?: number) {
  const [response] = await Promise.all([
    page.waitForResponse((candidate) => isWrite(candidate.request(), path)),
    action(),
  ]);
  expect(response.request().postDataJSON()).toEqual(body);
  if (status === undefined) {
    expect(response.ok(), `${path} returned HTTP ${response.status()}`).toBe(true);
  } else {
    expect(response.status()).toBe(status);
  }
}

async function openUiSettings(page: BrowserPage) {
  await navigateToPage(page, Page.AdminPanel);
  await page.getByRole("tab", { name: "UI Settings" }).click();
}

async function openVisibility(page: BrowserPage) {
  await openUiSettings(page);
  await page.getByRole("button", { name: "Configure Page Visibility" }).click();
}

async function openTelemetry(page: BrowserPage) {
  await page.goto("/ui/telemetry");
  await expect(page.getByRole("heading", { name: "Telemetry", exact: true })).toBeVisible();
  await dismissFeedbackPopup(page);
}

test.describe("User banner", () => {
  test.beforeEach(async ({ request }) => seed(request, banner, unpublished));
  test.afterEach(async ({ request }) => seed(request, banner, unpublished));

  test("publishes the announcement and removes it when unpublished", async ({ page }) => {
    await openUiSettings(page);
    const message = "Dashboard maintenance starts tonight.";
    const publish = page.getByRole("switch", { name: "Publish user banner" });
    const save = page.getByRole("button", { name: "Save banner" });
    const dismiss = page.getByRole("button", { name: /dismiss.*banner/i });
    await page.getByRole("textbox", { name: "Message" }).fill(`**${message}**`);
    await publish.click();
    await page.getByLabel("Banner severity").click();
    await page.getByRole("option", { name: "Warning", exact: true }).click();
    await submit(page, banner, () => save.click(), { enabled: true, message: `**${message}**`, severity: "warning" });
    await expect(dismiss).toBeVisible();
    await navigateToPage(page, Page.ApiKeys);
    await expect(page.getByText(message, { exact: true })).toBeVisible();
    await openUiSettings(page);
    await expect(publish).toBeChecked();
    await publish.click();
    await submit(page, banner, () => save.click(), { enabled: false, message: `**${message}**`, severity: "warning" });
    await expect(dismiss).toBeHidden();
  });

  test("a whitespace announcement cannot be published", async ({ page }) => {
    await openUiSettings(page);
    await page.getByRole("switch", { name: "Publish user banner" }).click();
    await page.getByRole("textbox", { name: "Message" }).fill("   ");
    await expect(page.getByText("Add a message before publishing.")).toBeVisible();
    await expect(page.getByRole("button", { name: "Save banner" })).toBeDisabled();
  });
});

test.describe("Internal user page visibility", () => {
  test.beforeEach(async ({ request }) => seed(request, visibility, unrestricted));
  test.afterEach(async ({ request }) => seed(request, visibility, unrestricted));

  test("keeps only the selected pages in internal navigation until reset", async ({ page, browser }) => {
    await openVisibility(page);
    await page.getByRole("checkbox", { name: /^Usage\b/ }).click();
    await page.getByRole("checkbox", { name: /^Virtual Keys\b/ }).click();
    await submit(page, visibility, () => page.getByRole("button", { name: "Save Page Visibility Settings" }).click(), {
      enabled_ui_pages_internal_users: ["new_usage", "api-keys"],
    });
    await expect(page.getByText("2 pages selected")).toBeVisible();

    const internal = await browser.newContext({ storageState: INTERNAL_USER_STORAGE_PATH, baseURL: new URL(page.url()).origin });
    try {
      const internalPage = await internal.newPage();
      await navigateToPage(internalPage, Page.ApiKeys);
      const nav = internalPage.getByRole("complementary");
      await expect(nav.getByRole("link", { name: "Usage", exact: true })).toBeVisible();
      await expect(nav.getByRole("link", { name: "Teams", exact: true })).toBeHidden();
      await internalPage.keyboard.press("ControlOrMeta+K");
      const search = internalPage.getByRole("combobox", { name: "Search" });
      await search.fill("Usage");
      await expect(internalPage.getByRole("option", { name: /^Usage\b/ })).toBeVisible();
      await search.fill("Teams");
      await expect(internalPage.getByRole("option", { name: /^Teams\b/ })).toBeHidden();

      await submit(page, visibility, () => page.getByRole("button", { name: "Reset to Default (All Pages)" }).click(), unrestricted);
      await expect(page.getByText("Not set (all pages visible)")).toBeVisible();
      await internalPage.reload();
      await expect(nav.getByRole("link", { name: "Teams", exact: true })).toBeVisible();
    } finally {
      await internal.close();
    }
  });

  test("shows the save failure and leaves the saved visibility unchanged", async ({ page }) => {
    await openVisibility(page);
    const failure = "Could not reach UI settings";
    await page.route(`**${visibility}`, (route) => route.fulfill({
      status: 503,
      json: { error: { message: failure } },
    }));
    await page.getByRole("checkbox", { name: /^Virtual Keys\b/ }).click();
    await submit(page, visibility, () => page.getByRole("button", { name: "Save Page Visibility Settings" }).click(), {
      enabled_ui_pages_internal_users: ["api-keys"],
    }, 503);
    const alert = page.getByRole("alert").filter({ hasText: "Could not update UI settings" });
    await expect(alert).toContainText(failure);
    await openVisibility(page);
    await expect(page.getByText("Not set (all pages visible)")).toBeVisible();
    await expect(page.getByRole("checkbox", { name: /^Virtual Keys\b/ })).not.toBeChecked();
  });
});

test.describe("UI theme", () => {
  test.beforeEach(async ({ request }) => seed(request, theme, defaultTheme));
  test.afterEach(async ({ request }) => seed(request, theme, defaultTheme));

  async function openTheme(page: BrowserPage) {
    const [prefill] = await Promise.all([
      page.waitForResponse((response) => {
        const request = response.request();
        return (
          request.method() === "GET" &&
          new URL(request.url()).pathname === "/get/ui_theme_settings" &&
          Boolean(request.headers().authorization)
        );
      }),
      navigateToPage(page, Page.UiTheme),
    ]);
    expect(prefill.ok()).toBe(true);
    expect(await prefill.finished()).toBeNull();
    const logo = page.getByRole("textbox", { name: "Custom Logo URL", exact: true });
    const dark = page.getByRole("textbox", { name: "Custom Logo URL (dark mode)", exact: true });
    const icon = page.getByRole("textbox", { name: "Custom Favicon URL", exact: true });
    const { values } = await prefill.json();
    await page.evaluate(() =>
      new Promise<void>((resolve) => requestAnimationFrame(() => requestAnimationFrame(() => resolve()))),
    );
    await expect(logo).toHaveValue(values.logo_url ?? "");
    await expect(dark).toHaveValue(values.logo_url_dark ?? "");
    await expect(icon).toHaveValue(values.favicon_url ?? "");
    return { logo, dark, icon };
  }

  test("saves branding, clears one image and resets all images", async ({ page }) => {
    const { logo, dark, icon } = await openTheme(page);
    const save = page.getByRole("button", { name: "Save Changes" });
    const asset = (file: string) => new URL(`/ui/assets/logos/${file}`, page.url()).href;
    const branding = {
      logo_url: asset("litellm_logo.png"),
      logo_url_dark: asset("litellm_logo_dark.png"),
      favicon_url: asset("litellm_monogram.svg"),
    };
    await logo.fill(branding.logo_url);
    await dark.fill(branding.logo_url_dark);
    await icon.fill(branding.favicon_url);
    const updated = page.getByText("Theme settings updated successfully!");
    await submit(page, theme, () => save.click(), branding);
    await expect(updated).toBeVisible();
    await page.getByRole("button", { name: "Close toast" }).click();
    await expect(updated).toBeHidden();
    await dark.fill("");
    await submit(page, theme, () => save.click(), { ...branding, logo_url_dark: null });
    await expect(updated).toBeVisible();
    await expect(logo).toHaveValue(branding.logo_url);
    await expect(dark).toHaveValue("");
    await expect(icon).toHaveValue(branding.favicon_url);
    await submit(page, theme, () => page.getByRole("button", { name: "Reset to Default" }).click(), defaultTheme);
    await expect(page.getByText("Theme settings reset to default!")).toBeVisible();
    await expect(logo).toHaveValue("");
    await expect(dark).toHaveValue("");
    await expect(icon).toHaveValue("");
  });

  test("shows the error for an invalid logo URL and keeps the draft for correction", async ({ page }) => {
    const { logo, dark, icon } = await openTheme(page);
    await logo.fill("not-a-url");
    await submit(page, theme, () => page.getByRole("button", { name: "Save Changes" }).click(), {
      logo_url: "not-a-url",
      logo_url_dark: (await dark.inputValue()) || null,
      favicon_url: (await icon.inputValue()) || null,
    }, 400);
    await expect(page.getByText("Failed to update theme settings")).toBeVisible();
    await expect(logo).toHaveValue("not-a-url");
  });
});

test.describe("Telemetry", () => {
  test.afterEach(async ({ request }) => seed(request, telemetry, disabledTelemetry));

  test("turns off dependent groups and then all telemetry without clearing the destination", async ({ page, request }) => {
    await seed(request, telemetry, enabledTelemetry);
    await openTelemetry(page);
    const success = page.getByRole("switch", { name: "Request success" });
    await expect(success).toBeChecked();
    await submit(page, telemetry, () => success.click(), {
      groups: ["heartbeat", "page_navigation"],
      destination: "local_table",
    });
    await expect(page.getByRole("switch", { name: "Token info" })).not.toBeChecked();
    await page.getByRole("tab", { name: "Admin UI" }).click();
    await expect(page.getByRole("switch", { name: "Page navigation" })).toBeChecked();
    const power = page.getByRole("switch", { name: "Telemetry", exact: true });
    await submit(page, telemetry, () => power.click(), { groups: [], destination: "local_table" });
    await expect(power).not.toBeChecked();
    await page.reload();
    await expect(power).not.toBeChecked();
    await expect(page.getByRole("radio", { name: /^Local table\b/ })).toBeChecked();
  });

  test("explains that turning telemetry on needs enterprise setup and does not write", async ({ page, request }) => {
    await seed(request, telemetry, disabledTelemetry);
    await openTelemetry(page);
    const writes: Request[] = [];
    page.on("request", (request) => {
      if (isWrite(request, telemetry)) writes.push(request);
    });
    const power = page.getByRole("switch", { name: "Telemetry", exact: true });
    await power.click();
    const dialog = page.getByRole("dialog", { name: /enterprise/i });
    await expect(dialog).toBeVisible();
    await expect(dialog.getByText(/Telemetry can't be turned on from this page|Telemetry is only available on LiteLLM Enterprise/)).toBeVisible();
    await dialog.getByRole("button", { name: "Close", exact: true }).last().click();
    await expect(dialog).toBeHidden();
    await expect(power).not.toBeChecked();
    expect(writes).toHaveLength(0);
  });
});
