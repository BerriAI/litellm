import { expect, test, type APIRequestContext, type APIResponse, type Request } from "@playwright/test";
import { ADMIN_STORAGE_PATH } from "../../constants";
import { Page } from "../../fixtures/pages";
import { navigateToPage } from "../../helpers/navigation";
import { readBack } from "../../helpers/roundTrip";
import { masterKey, uniqueSuffix } from "../../helpers/traffic";

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

    const bodies: unknown[] = [];
    page.on("request", (sent) => {
      if (sent.method() === "POST" && new URL(sent.url()).pathname === mcpEndpoint) {
        bodies.push(sent.postDataJSON());
      }
    });
    await navigateToPage(page, Page.ModelHubTable);
    await page.getByRole("tab", { name: "MCP Hub" }).click();
    await page.getByRole("button", { name: "Manage MCP Hub Visibility" }).click();
    const dialog = page.getByRole("dialog", { name: "Manage MCP Hub Visibility" });
    const checkbox = dialog.getByRole("checkbox", { name: `Publish ${name}`, exact: true });
    await expect(checkbox).toBeChecked();
    await checkbox.uncheck();
    await dialog.getByRole("button", { name: "Next" }).click();
    await dialog.getByRole("button", { name: "Save Publication List" }).click();
    await expect(page.getByText("MCP Hub publication list updated")).toBeVisible();
    await expect(dialog).toBeHidden();
    expect(bodies).toEqual([{ mcp_server_ids: [] }]);
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
    const firstUrl = "https://example.test/first";
    const secondUrl = "https://example.test/second";
    const original = Object.entries(useful_links ?? {})
      .map(([name, value]) => [name, typeof value === "string" ? { url: value, index: 0 } : value] as const)
      .sort(([, a], [, b]) => a.index - b.index)
      .map(([name, { url }], index) => [name, { url, index }]);
    const writes: unknown[] = [];
    page.on("request", (sent) => {
      if (sent.method() === "POST" && new URL(sent.url()).pathname === linksEndpoint) {
        const { useful_links, ...extra } = sent.postDataJSON();
        writes.push({ ...extra, useful_links: Object.entries(useful_links) });
      }
    });
    await navigateToPage(page, Page.ModelHubTable);
    for (const [name] of original) {
      await expect(page.getByRole("cell", { name: String(name), exact: true })).toBeVisible();
    }
    for (const [name, url] of [[first, firstUrl], [second, secondUrl]]) {
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
    const firstLink = [first, { url: firstUrl, index: original.length }];
    const secondLink = [second, { url: secondUrl, index: original.length + 1 }];
    expect(writes).toEqual([
      { useful_links: [...original, firstLink] },
      { useful_links: [...original, firstLink, secondLink] },
      {
        useful_links: [
          ...original,
          [second, { url: secondUrl, index: original.length }],
          [first, { url: firstUrl, index: original.length + 1 }],
        ],
      },
    ]);
  });
});
