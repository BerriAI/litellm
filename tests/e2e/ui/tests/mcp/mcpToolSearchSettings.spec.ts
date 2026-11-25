import { test, expect, type APIRequestContext, type Locator, type Page as PlaywrightPage } from "@playwright/test";
import { ADMIN_STORAGE_PATH } from "../../constants";
import { navigateToPage } from "../../helpers/navigation";
import { Page } from "../../fixtures/pages";
import { Role, users } from "../../fixtures/users";
import type { components } from "../../../../../litellm/ui/litellm-dashboard/src/lib/http/schema";

type ToolSearchSettings = components["schemas"]["MCPToolSearchSettings"];

test.describe.configure({ mode: "serial" });

const UPDATE_PATH = "/update/mcp_tool_search_settings";
const KEYWORD_DEFAULTS: ToolSearchSettings = {
  embedding_model: null,
  top_k: 5,
  similarity_threshold: 0,
  core_tools: [],
};
const SEEDED: ToolSearchSettings = { ...KEYWORD_DEFAULTS, top_k: 3, core_tools: ["seeded_tool"] };

async function putSettings(request: APIRequestContext, settings: ToolSearchSettings): Promise<void> {
  const res = await request.patch(UPDATE_PATH, {
    headers: { Authorization: `Bearer ${users[Role.ProxyAdmin].password}` },
    data: settings,
  });
  expect(res.ok(), await res.text()).toBe(true);
}

async function openToolSearch(page: PlaywrightPage, savedCoreTools: string): Promise<Locator> {
  await navigateToPage(page, Page.McpServers);
  await page.getByRole("tab", { name: "Tool Search" }).click();
  const panel = page.getByRole("tabpanel", { name: "Tool Search" });
  await expect(panel.getByLabel("Always Returned First")).toHaveValue(savedCoreTools);
  return panel;
}

const nextSave = (page: PlaywrightPage) =>
  page.waitForRequest((req) => req.method() === "PATCH" && new URL(req.url()).pathname.endsWith(UPDATE_PATH));

test.describe("MCP tool search settings", () => {
  test.use({ storageState: ADMIN_STORAGE_PATH });

  test.beforeEach(async ({ request }) => {
    await putSettings(request, SEEDED);
  });

  test.afterEach(async ({ request }) => {
    await putSettings(request, KEYWORD_DEFAULTS);
  });

  test("saves edited settings and shows them again after a reload", async ({ page }) => {
    const panel = await openToolSearch(page, "seeded_tool");
    const topK = panel.getByLabel("Top K Results");
    const coreTools = panel.getByLabel("Always Returned First");
    const save = panel.getByRole("button", { name: "Save Settings" });

    await expect(topK).toHaveValue("3");
    await expect(save).toBeDisabled();

    await topK.fill("12");
    await coreTools.fill("alpha\nbeta");
    await expect(save).toBeEnabled();

    const saved = nextSave(page);
    await save.click();
    expect((await saved).postDataJSON()).toEqual({
      embedding_model: null,
      top_k: 12,
      similarity_threshold: 0,
      core_tools: ["alpha", "beta"],
    });
    await expect(page.getByText("Settings updated successfully.", { exact: false })).toBeVisible();

    const reopened = await openToolSearch(page, "alpha\nbeta");
    await expect(reopened.getByLabel("Top K Results")).toHaveValue("12");
  });

  test("clamps an out-of-range Top K to the largest value the proxy accepts", async ({ page }) => {
    const panel = await openToolSearch(page, "seeded_tool");
    const topK = panel.getByLabel("Top K Results");

    await topK.fill("500");
    await panel.getByLabel("Always Returned First").click();
    await expect(topK).toHaveValue("100");

    const saved = nextSave(page);
    await panel.getByRole("button", { name: "Save Settings" }).click();
    expect((await saved).postDataJSON()).toMatchObject({ top_k: 100 });
  });

  test("keeps the edits and lets the user retry when the save fails", async ({ page }) => {
    await page.route(
      `**${UPDATE_PATH}`,
      (route) =>
        route.fulfill({
          status: 500,
          contentType: "application/json",
          body: JSON.stringify({ detail: { error: "tool search settings store unavailable" } }),
        }),
      { times: 1 },
    );
    const panel = await openToolSearch(page, "seeded_tool");
    const topK = panel.getByLabel("Top K Results");
    const save = panel.getByRole("button", { name: "Save Settings" });

    await topK.fill("9");
    const failed = nextSave(page);
    await save.click();
    await failed;
    await expect(page.getByText("tool search settings store unavailable")).toBeVisible();
    await expect(topK).toHaveValue("9");
    await expect(save).toBeEnabled();

    const retried = nextSave(page);
    await save.click();
    expect((await retried).postDataJSON()).toEqual({ ...SEEDED, top_k: 9 });
    await expect(page.getByText("Settings updated successfully.", { exact: false })).toBeVisible();
  });
});
