import { test, expect, type Page as PlaywrightPage } from "@playwright/test";
import { masterKey } from "../../helpers/traffic";
import { navigateToPage } from "../../helpers/navigation";
import { Page } from "../../fixtures/pages";
import { ADMIN_STORAGE_PATH } from "../../constants";
import { createMcpServer, deleteMcpServerByName } from "../../helpers/mcp";
import { captureRequestBody, readBack } from "../../helpers/roundTrip";

/**
 * Editing and deleting an MCP server, verified against the API. The reported failures are all on
 * this side: renames that need repeating, deletes that need two attempts, each toasting success on
 * the failing attempt. The URL is unreachable on purpose; only persistence is under test here.
 */
const UNREACHABLE_URL = "https://e2e-fake-mcp.test.local/mcp";

/** GET /v1/mcp/server returns a bare array of servers (useMCPServers types it MCPServer[]). */
async function findServerByName(page: PlaywrightPage, serverName: string): Promise<Record<string, any> | undefined> {
  const servers = await readBack<Record<string, any>[]>(page, "/v1/mcp/server");
  return servers.find((server) => server.server_name === serverName);
}

test.describe("MCP Servers - edit and delete", () => {
  test.use({ storageState: ADMIN_STORAGE_PATH });

  let serverName: string;

  test.beforeEach(async ({ page }) => {
    serverName = await createMcpServer(page, UNREACHABLE_URL);
  });

  // The rename test leaves an unreachable server behind, which slows the MCP page for later tests.
  test.afterEach(async ({ page }) => {
    await deleteMcpServerByName(page, serverName);
  });

  test("Renaming a server's alias persists", async ({ page }) => {
    const before = await findServerByName(page, serverName);
    expect(before, `created server ${serverName} readable from /v1/mcp/server`).toBeTruthy();

    await page.getByTestId("mcp-servers-grid").getByText(serverName).first().click();
    await expect(page.getByRole("button", { name: /Back to All Servers/i })).toBeVisible({ timeout: 10_000 });

    // exact: the server view also renders a "Network Settings" tab.
    await page.getByRole("tab", { name: "Settings", exact: true }).click();

    // A card click may land straight in edit mode, so only click the button when it rendered.
    const editSettings = page.getByRole("button", { name: "Edit Settings" });
    if (await editSettings.isVisible().catch(() => false)) {
      await editSettings.click();
    }

    // The create modal stays mounted behind the view with its own #alias and Save.
    const settingsPanel = page.getByRole("tabpanel", { name: "Settings" });

    const newAlias = `${serverName}_renamed`;
    const aliasInput = settingsPanel.locator('input[id="alias"]');
    await expect(aliasInput).toBeVisible({ timeout: 10_000 });
    await aliasInput.fill(newAlias);

    const update = await captureRequestBody(page, { method: "PUT", urlIncludes: "/v1/mcp/server" }, async () => {
      await settingsPanel.getByRole("button", { name: "Save Changes" }).click();
    });
    expect(update.alias, "new alias on the wire").toBe(newAlias);
    // An unidentified target is one way a save succeeds and changes nothing.
    expect(update.server_id, "update targets the server being edited").toBe(before?.server_id);

    // The reported symptom is a first save that returns success and does not stick.
    await expect
      .poll(async () => (await findServerByName(page, serverName))?.alias, {
        message: `alias for ${serverName} did not persist after one save`,
        timeout: 15_000,
      })
      .toBe(newAlias);
  });

  test("Deleting a server removes it", async ({ page }) => {
    expect(await findServerByName(page, serverName), `created server ${serverName} exists`).toBeTruthy();

    const card = page.getByTestId("mcp-servers-grid").getByRole("button", { name: serverName });
    await card.getByRole("button", { name: "Server actions" }).click();
    await page.getByRole("menuitem", { name: "Delete" }).click();

    const dialog = page.getByRole("alertdialog");
    await expect(dialog.getByText("Delete MCP Server?")).toBeVisible({
      timeout: 5_000,
    });
    await dialog.getByRole("button", { name: "Delete", exact: true }).click();

    // One attempt has to be enough; the report is a delete that needs two.
    await expect
      .poll(async () => await findServerByName(page, serverName), {
        message: `server ${serverName} still present after one delete`,
        timeout: 15_000,
      })
      .toBeUndefined();
  });
});

test.describe("MCP Servers - stdio environment", () => {
  test.use({ storageState: ADMIN_STORAGE_PATH });
  let serverName = "";

  test.afterEach(async ({ page }) => {
    if (serverName) await deleteMcpServerByName(page, serverName);
  });

  test("Editing a stdio server preserves, replaces, and clears its environment", async ({ page }, testInfo) => {
    serverName = `e2e_stdio_env_${Date.now()}`;
    const savedEnv = {
      E2E_SETTING: "saved-value",
      E2E_OPTION: "another-value",
    };
    const created = await page.request.post("/v1/mcp/server", {
      headers: { Authorization: `Bearer ${masterKey()}` },
      data: {
        server_name: serverName,
        alias: serverName,
        transport: "stdio",
        command: "python3",
        args: ["--version"],
        env: savedEnv,
        mcp_info: {
          server_name: serverName,
          description: "Environment persistence regression",
        },
      },
    });
    expect(created.ok(), await created.text()).toBe(true);
    await navigateToPage(page, Page.McpServers);
    const openSettings = async () => {
      await page.getByTestId("mcp-servers-grid").getByText(serverName).first().click();
      await page.getByRole("tab", { name: "Settings", exact: true }).click();
      const editSettings = page.getByRole("button", { name: "Edit Settings" });
      if (await editSettings.isVisible()) await editSettings.click();
    };
    await openSettings();
    const settingsPanel = page.getByRole("tabpanel", { name: "Settings" });
    const envInput = settingsPanel.getByLabel("Environment (JSON object)");
    await expect(envInput).toBeVisible();
    await envInput.scrollIntoViewIfNeeded();
    const screenshotPath = testInfo.outputPath("stdio-environment-form.png");
    await page.screenshot({ path: screenshotPath, fullPage: true });
    await testInfo.attach("stdio-environment-form", { path: screenshotPath, contentType: "image/png" });
    await expect.soft(envInput).toHaveValue(JSON.stringify(savedEnv, null, 2));

    for (const env of [savedEnv, { E2E_SETTING: "replacement" }, {}]) {
      if (env !== savedEnv) {
        await openSettings();
        await envInput.fill(Object.keys(env).length ? JSON.stringify(env) : "");
      }
      const saved = page.waitForResponse(
        (response) => response.request().method() === "PUT" && response.url().includes("/v1/mcp/server"),
      );
      const update = await captureRequestBody(page, { method: "PUT", urlIncludes: "/v1/mcp/server" }, async () => {
        await settingsPanel.getByRole("button", { name: "Save Changes" }).click();
      });
      expect.soft(update.env).toEqual(env);
      expect((await saved).ok()).toBe(true);
      await expect(page.getByTestId("mcp-servers-grid")).toBeVisible();
      await expect.poll(async () => (await findServerByName(page, serverName))?.env).toEqual(env);
    }
  });
});
