import { expect, test, type APIRequestContext, type APIResponse, type Request } from "@playwright/test";
import { ADMIN_STORAGE_PATH } from "../../constants";
import { Page } from "../../fixtures/pages";
import { navigateToPage } from "../../helpers/navigation";
import { captureRequestBody, readBack } from "../../helpers/roundTrip";
import { masterKey, uniqueSuffix } from "../../helpers/traffic";

const modelEndpoint = "/model_group/make_public";
const mcpEndpoint = "/v1/mcp/make_public";
const linksEndpoint = "/model_hub/update_useful_links";
const pluginsEndpoint = "/claude-code/plugins";
const auth = () => ({ Authorization: `Bearer ${masterKey()}` });

async function requireSuccess(response: APIResponse): Promise<APIResponse> {
  expect(response.ok(), `${response.url()} returned ${response.status()}: ${await response.text()}`).toBe(true);
  return response;
}

const post = (request: APIRequestContext, path: string, data?: object) =>
  request.post(path, { headers: auth(), data }).then(requireSuccess);

test.describe("AI Hub visibility", () => {
  test.describe.configure({ mode: "serial" });
  test.use({ storageState: ADMIN_STORAGE_PATH });

  const restore: Array<() => Promise<unknown>> = [];
  test.afterEach(async () => {
    const errors: unknown[] = [];
    for (const undo of restore.splice(0).reverse()) {
      try {
        await undo();
      } catch (error) {
        errors.push(error);
      }
    }
    if (errors.length) throw new AggregateError(errors, "Restore AI Hub fixtures");
  });

  test("publishes the selected model group and confirms success", async ({ page, request }) => {
    const catalog = await readBack<{ data: { model_group: string; is_public_model_group?: boolean }[] }>(
      page,
      "/model_group/info",
    );
    const original = catalog.data.filter((row) => row.is_public_model_group).map((row) => row.model_group);
    const name = `e2e-hub-model-${uniqueSuffix()}`;
    const created = await post(request, "/model/new", {
      model_name: name,
      litellm_params: { model: "openai/fake-gpt-4", api_key: "fake-key" },
      model_info: {},
    });
    const { model_info: { id } } = await created.json();
    restore.push(() => post(request, "/model/delete", { id }));
    restore.push(() => post(request, modelEndpoint, { model_groups: original }));

    await navigateToPage(page, Page.ModelHubTable);
    await expect(page.getByRole("tabpanel").getByText(name, { exact: true })).toBeVisible();
    await page.getByRole("button", { name: "Select Models to Make Public" }).click();
    const dialog = page.getByRole("dialog", { name: "Make Models Public" });
    await dialog.getByPlaceholder("Search model names...").fill(name);
    await dialog.getByRole("checkbox", { name: "Select All (1)", exact: true }).check();
    await dialog.getByRole("button", { name: "Next" }).click();
    const [response, body] = await Promise.all([
      page.waitForResponse((res) => new URL(res.url()).pathname === modelEndpoint && res.request().method() === "POST"),
      captureRequestBody(page, { method: "POST", urlIncludes: modelEndpoint }, () =>
        dialog.getByRole("button", { name: "Make Public" }).click(),
      ),
    ]);
    expect(response.ok()).toBe(true);
    expect(body).toEqual({ model_groups: [name] });
    await expect(page.getByText("Successfully made 1 model group(s) public!", { exact: true })).toBeVisible();
    await expect(dialog).toBeHidden();
  });

  test("clears the MCP publication list when no servers remain selected", async ({ page, request }) => {
    const catalog = await readBack<{ server_id: string; mcp_info: { is_public_explicit: boolean } }[]>(
      page,
      "/v1/mcp/server",
    );
    const original = catalog.filter((row) => row.mcp_info.is_public_explicit).map((row) => row.server_id);
    const name = `e2e_mcp_${uniqueSuffix().replaceAll("-", "_")}`;
    const created = await post(request, "/v1/mcp/server", {
      server_name: name,
      url: "https://e2e-fake-mcp.test.local/mcp",
      transport: "http",
      auth_type: "none",
    });
    const { server_id: id } = await created.json();
    restore.push(() => request.delete(`/v1/mcp/server/${id}`, { headers: auth() }).then(requireSuccess));
    restore.push(() => post(request, mcpEndpoint, { mcp_server_ids: original }));
    await post(request, mcpEndpoint, { mcp_server_ids: [id] });

    await navigateToPage(page, Page.ModelHubTable);
    await page.getByRole("tab", { name: "MCP Hub" }).click();
    await page.getByRole("button", { name: "Manage MCP Hub Visibility" }).click();
    const dialog = page.getByRole("dialog", { name: "Manage MCP Hub Visibility" });
    const checkbox = dialog.getByRole("checkbox", { name: `Publish ${name}`, exact: true });
    await expect(checkbox).toBeChecked();
    await checkbox.uncheck();
    await dialog.getByRole("button", { name: "Next" }).click();
    const body = await captureRequestBody(page, { method: "POST", urlIncludes: mcpEndpoint }, () =>
      dialog.getByRole("button", { name: "Save Publication List" }).click(),
    );
    expect(body).toEqual({ mcp_server_ids: [] });
    await expect(page.getByText("MCP Hub publication list updated")).toBeVisible();
    await expect(dialog).toBeHidden();
  });

  test("publishes and unpublishes only the skills whose visibility changed", async ({ page, request }) => {
    const suffix = uniqueSuffix();
    const [publish, unpublish, unchanged] = ["publish", "unpublish", "unchanged"].map((kind) => `e2e-${kind}-${suffix}`);
    for (const name of [publish, unpublish, unchanged]) {
      await post(request, pluginsEndpoint, {
        name,
        source: { source: "github", repo: "example/catalog-skills" },
        description: "AI Hub visibility test skill",
        version: "1.0.0",
      });
      restore.push(() => request.delete(`${pluginsEndpoint}/${name}`, { headers: auth() }).then(requireSuccess));
    }
    await post(request, `${pluginsEndpoint}/${publish}/disable`);
    await post(request, `${pluginsEndpoint}/${unpublish}/enable`);
    await post(request, `${pluginsEndpoint}/${unchanged}/enable`);

    await navigateToPage(page, Page.ModelHubTable);
    await page.getByRole("tab", { name: "Skills Hub" }).click();
    await page.getByRole("button", { name: "Select Skills to Make Public" }).click();
    const dialog = page.getByRole("dialog", { name: "Publish to Skills Hub" });
    await expect(dialog.getByRole("checkbox", { name: unchanged, exact: true })).toBeChecked();
    await dialog.getByRole("checkbox", { name: publish, exact: true }).check();
    await dialog.getByRole("checkbox", { name: unpublish, exact: true }).uncheck();
    await dialog.getByRole("button", { name: "Next" }).click();

    const writes: { path: string; body: string | null }[] = [];
    const observe = (sent: Request) => {
      const path = new URL(sent.url()).pathname;
      if (sent.method() === "POST" && /\/claude-code\/plugins\/[^/]+\/(enable|disable)$/.test(path)) {
        writes.push({ path, body: sent.postData() });
      }
    };
    page.on("request", observe);
    try {
      await dialog.getByRole("button", { name: "Publish to Hub" }).click();
      await expect(page.getByText(/Skills Hub updated/)).toBeVisible();
      await expect(dialog).toBeHidden();
    } finally {
      page.off("request", observe);
    }
    expect(writes.sort((a, b) => a.path.localeCompare(b.path))).toEqual([
      { path: `${pluginsEndpoint}/${publish}/enable`, body: null },
      { path: `${pluginsEndpoint}/${unpublish}/disable`, body: null },
    ]);
  });

  test("keeps the new useful link order after reloading", async ({ page, request }) => {
    type UsefulLinks = Record<string, string | { url: string; index: number }>;
    const { useful_links } = await readBack<{ useful_links: UsefulLinks | null }>(page, "/public/model_hub/info");
    restore.push(() => post(request, linksEndpoint, { useful_links: useful_links ?? {} }));

    const suffix = uniqueSuffix();
    const first = `First link ${suffix}`;
    const second = `Second link ${suffix}`;
    await navigateToPage(page, Page.ModelHubTable);
    for (const [name, url] of [[first, "https://example.test/first"], [second, "https://example.test/second"]]) {
      await page.getByPlaceholder("Friendly name").fill(name);
      await page.getByPlaceholder("https://example.com").fill(url);
      await page.getByRole("button", { name: "Add Link" }).click();
      await expect(page.getByRole("row").filter({ hasText: name })).toBeVisible();
    }

    await page.getByRole("button", { name: "Rearrange Order" }).click();
    await page.getByRole("row").filter({ hasText: second }).getByTestId(/^move-up-/).click();
    await page.getByRole("button", { name: "Save Order" }).click();
    await expect(page.getByText("Link order saved successfully")).toBeVisible();
    await page.reload();
    const added = page.getByRole("cell", { name: first, exact: true })
      .or(page.getByRole("cell", { name: second, exact: true }));
    await expect(added).toHaveText([second, first]);
  });
});
