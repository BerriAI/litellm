import { expect, test, type Page as PlaywrightPage } from "@playwright/test";
import { ADMIN_STORAGE_PATH } from "../../constants";
import { seedProxySettings, writeProxySettings } from "../../helpers/proxySettings";
import { rootPath } from "../../helpers/traffic";
import type { components } from "../../../../../litellm/ui/litellm-dashboard/src/lib/http/schema";

type Settings = components["schemas"]["MCPSemanticFilterSettings"];

const SETTINGS = "mcp_semantic_filter_settings";
const UPDATE_PATH = `/update/${SETTINGS}`;
const DEFAULTS: Settings = {
  enabled: false,
  embedding_model: "text-embedding-3-small",
  top_k: 10,
  similarity_threshold: 0.3,
};
const SEEDED: Settings = { ...DEFAULTS, top_k: 7 };

async function openSettings(page: PlaywrightPage) {
  await page.goto(`${rootPath()}/ui/mcp-servers/`);
  await page.getByRole("tab", { name: "Semantic Filter" }).click();
  const panel = page.getByRole("tabpanel", { name: "Semantic Filter" });
  await expect(panel.getByRole("spinbutton", { name: "Top K Results" })).toHaveValue(String(SEEDED.top_k), {
    timeout: 35000,
  });
  const searchHint = page.getByRole("button", { name: "Dismiss search hint" });
  if (await searchHint.isVisible()) await searchHint.click();
  return panel;
}

const nextSave = (page: PlaywrightPage) =>
  page.waitForResponse(
    (response) => response.request().method() === "PATCH" && new URL(response.url()).pathname.endsWith(UPDATE_PATH),
  );

test.describe("MCP Semantic Filter settings", () => {
  test.describe.configure({ mode: "serial" });
  test.use({ storageState: ADMIN_STORAGE_PATH });

  test.beforeEach(async ({ request, page }) => {
    await seedProxySettings(request, page, SETTINGS, SEEDED);
  });

  test.afterEach(async ({ request }) => {
    await writeProxySettings(request, SETTINGS, DEFAULTS);
  });

  test("saves the clamped Top K when the user enters a value above the limit", async ({ page }) => {
    const panel = await openSettings(page);
    const topK = panel.getByRole("spinbutton", { name: "Top K Results" });
    const save = panel.getByRole("button", { name: "Save Settings" });
    await expect(save).toBeDisabled();

    await topK.fill("500");
    await topK.press("Tab");
    await expect(topK).toHaveValue("100");

    const saving = nextSave(page);
    await save.click();
    const response = await saving;
    expect(response.ok()).toBe(true);
    expect(response.request().postDataJSON()).toEqual({
      ...SEEDED,
      top_k: 100,
    });
    await expect(page.getByText("Settings updated successfully.", { exact: false })).toBeVisible();
  });

  test("shows the error from a rejected empty Top K and lets the user correct it", async ({ page }) => {
    const panel = await openSettings(page);
    const topK = panel.getByRole("spinbutton", { name: "Top K Results" });
    const save = panel.getByRole("button", { name: "Save Settings" });

    await topK.fill("");
    const rejecting = nextSave(page);
    await save.click();
    const rejection = await rejecting;
    expect(rejection.status()).toBe(422);
    expect(rejection.request().postDataJSON()).toEqual({
      ...SEEDED,
      top_k: null,
    });
    await expect(panel.getByText("Could not update settings", { exact: true })).toBeVisible();
    await expect(topK).toHaveValue("");
    await expect(save).toBeEnabled();

    await topK.fill("9");
    const retrying = nextSave(page);
    await save.click();
    const retry = await retrying;
    expect(retry.ok()).toBe(true);
    expect(retry.request().postDataJSON()).toEqual({ ...SEEDED, top_k: 9 });
    await expect(page.getByText("Settings updated successfully.", { exact: false })).toBeVisible();
  });
});
