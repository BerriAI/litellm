import { expect, test, type APIRequestContext, type Page as PlaywrightPage } from "@playwright/test";
import { ADMIN_STORAGE_PATH } from "../../constants";
import { Page } from "../../fixtures/pages";
import { navigateToPage } from "../../helpers/navigation";
import { masterKey } from "../../helpers/traffic";
import type { components } from "../../../../../litellm/ui/litellm-dashboard/src/lib/http/schema";

type Settings = components["schemas"]["WebSearchInterceptionSettings"];

const UPDATE_PATH = "/update/websearch_interception_settings";
const DEFAULTS: Settings = {
  enabled: false,
  enabled_providers: [],
  search_tool_name: null,
  max_agentic_loops: null,
};
const SEEDED: Settings = {
  ...DEFAULTS,
  enabled_providers: ["bedrock"],
  max_agentic_loops: 4,
};

async function putSettings(request: APIRequestContext, settings: Settings) {
  const response = await request.patch(UPDATE_PATH, {
    headers: { Authorization: `Bearer ${masterKey()}` },
    data: settings,
  });
  expect(response.ok(), await response.text()).toBe(true);
}

async function openSettings(page: PlaywrightPage, expectedLoops: string) {
  const panel = page.getByRole("tabpanel", { name: "Web Search Interception" });
  await expect(async () => {
    await navigateToPage(page, Page.AdminPanel);
    await page.getByRole("tab", { name: "Web Search Interception" }).click();
    await expect(panel.getByRole("spinbutton", { name: "Max Agentic Loops" })).toHaveValue(expectedLoops, {
      timeout: 2000,
    });
    await expect(panel.getByText("Amazon Bedrock", { exact: true })).toBeVisible({ timeout: 2000 });
  }).toPass({ timeout: 20000 });
  return panel;
}

const nextSave = (page: PlaywrightPage) =>
  page.waitForResponse(
    (response) => response.request().method() === "PATCH" && new URL(response.url()).pathname.endsWith(UPDATE_PATH),
  );

test.describe("Web Search Interception settings", () => {
  test.describe.configure({ mode: "serial" });
  test.use({ storageState: ADMIN_STORAGE_PATH });

  test.beforeEach(async ({ request }) => {
    await putSettings(request, SEEDED);
  });

  test.afterEach(async ({ request }) => {
    await putSettings(request, DEFAULTS);
  });

  test("clears a saved loop limit without dropping the provider selection", async ({ page }) => {
    const panel = await openSettings(page, "4");
    const save = panel.getByRole("button", { name: "Save Settings" });
    await expect(save).toBeDisabled();

    await panel.getByRole("spinbutton", { name: "Max Agentic Loops" }).fill("");
    const saving = nextSave(page);
    await save.click();
    const response = await saving;
    expect(response.ok()).toBe(true);
    expect(response.request().postDataJSON()).toEqual({
      ...SEEDED,
      max_agentic_loops: null,
    });
    await expect(page.getByText("Settings updated successfully.", { exact: false })).toBeVisible();

    const reopened = await openSettings(page, "");
    await expect(reopened.getByText("Amazon Bedrock", { exact: true })).toBeVisible();
  });

  test("shows a rejected loop limit and lets the user correct it", async ({ page }) => {
    const panel = await openSettings(page, "4");
    const loops = panel.getByRole("spinbutton", { name: "Max Agentic Loops" });
    const save = panel.getByRole("button", { name: "Save Settings" });

    await loops.fill("0");
    const rejecting = nextSave(page);
    await save.click();
    const rejection = await rejecting;
    expect(rejection.status()).toBe(422);
    expect(rejection.request().postDataJSON()).toEqual({
      ...SEEDED,
      max_agentic_loops: 0,
    });
    await expect(panel.getByText("Could not update settings", { exact: true })).toBeVisible();
    await expect(loops).toHaveValue("0");
    await expect(save).toBeEnabled();

    await loops.fill("8");
    const retrying = nextSave(page);
    await save.click();
    const retry = await retrying;
    expect(retry.ok()).toBe(true);
    expect(retry.request().postDataJSON()).toEqual({
      ...SEEDED,
      max_agentic_loops: 8,
    });
    await expect(page.getByText("Settings updated successfully.", { exact: false })).toBeVisible();
  });
});
