import { test, expect, type APIRequestContext } from "@playwright/test";
import { randomUUID } from "node:crypto";
import { execFileSync } from "node:child_process";
import * as path from "node:path";
import { Page } from "../../fixtures/pages";
import { dismissFeedbackPopup, navigateToPage } from "../../helpers/navigation";

/**
 * The Tool Policies table gets a User column: the owner of the key that discovered the tool, shown
 * as alias (then email, then id) linking to the user's page, and a plain dash when the key has no
 * owner. Both rows are produced the way a customer produces them, a chat completion carrying a
 * tool through the proxy, so the column is read from the same registry the proxy writes.
 */
const unhex = (): string => randomUUID().replaceAll("-", "");

const toolCall = (model: string, toolName: string) => ({
  model,
  messages: [{ role: "user", content: "tool policy user column" }],
  tools: [
    {
      type: "function",
      function: {
        name: toolName,
        description: "integration tool",
        parameters: { type: "object", properties: {} },
      },
    },
  ],
});

test("the Tool Policies page names the user behind the key that discovered a tool", async ({
  page,
  request,
}) => {
  const master = process.env.LITELLM_MASTER_KEY ?? "sk-integration-master";
  const upstream = (
    process.env.INTEGRATION_UPSTREAM_URL ?? "http://127.0.0.1:8190"
  ).replace(/\/+$/, "");
  const auth = { Authorization: `Bearer ${master}` };
  const marker = unhex();
  const alias = `ui-owner-${marker}`;
  const model = `ui-tool-policies-${marker}`;
  const ownedTool = `ui_owned_tool_${marker}`;
  const unownedTool = `ui_unowned_tool_${marker}`;
  const support = (...args: string[]) =>
    execFileSync(
      process.env.INTEGRATION_PYTHON ?? "python",
      [
        path.resolve(
          __dirname,
          "../../../../integration/_support/tool_rows.py",
        ),
        ...args,
      ],
      { encoding: "utf8", timeout: 10_000, killSignal: "SIGKILL" },
    );

  const post = async (api: APIRequestContext, route: string, data: object) => {
    const response = await api.post(route, { headers: auth, data });
    expect(response.status(), `POST ${route}: ${await response.text()}`).toBe(
      200,
    );
    return response.json();
  };

  let modelId = "";
  let userId = "";
  const keys: string[] = [];
  try {
    modelId = (
      await post(request, "/model/new", {
        model_name: model,
        litellm_params: {
          model: `openai/${model}`,
          api_key: "sk-upstream",
          api_base: `${upstream}/v1`,
        },
      })
    ).model_id;
    userId = (
      await post(request, "/user/new", {
        user_id: `ui-user-${marker}`,
        user_alias: alias,
        user_email: `${alias}@integration.example`,
        auto_create_key: false,
      })
    ).user_id;
    const ownedKey = (
      await post(request, "/key/generate", { user_id: userId, models: [model] })
    ).key;
    const unownedKey = (
      await post(request, "/key/generate", { models: [model] })
    ).key;
    keys.push(ownedKey, unownedKey);
    for (const [key, toolName] of [
      [ownedKey, ownedTool],
      [unownedKey, unownedTool],
    ]) {
      const response = await request.post("/v1/chat/completions", {
        headers: { Authorization: `Bearer ${key}` },
        data: toolCall(model, toolName),
      });
      expect(response.status(), await response.text()).toBe(200);
    }
    await expect
      .poll(
        async () => {
          const response = await request.get("/v1/tool/list", {
            headers: auth,
          });
          if (response.status() !== 200) return [];
          const names = (
            (await response.json()).tools as { tool_name: string }[]
          ).map((tool) => tool.tool_name);
          return [ownedTool, unownedTool].filter((name) =>
            names.includes(name),
          );
        },
        {
          timeout: 70_000,
          message: "the discovered tools never reached the registry",
        },
      )
      .toEqual([ownedTool, unownedTool]);

    await page.goto("/ui/login");
    await page.getByPlaceholder("Enter your username").fill("admin");
    await page.getByPlaceholder("Enter your password").fill(master);
    await page.getByRole("button", { name: "Login", exact: true }).click();
    await expect(page).toHaveURL(
      (url) =>
        url.pathname.startsWith("/ui") && !url.pathname.includes("login"),
    );
    await navigateToPage(page, Page.ToolPolicies);
    await dismissFeedbackPopup(page);

    const table = page.locator("table").filter({ visible: true }).first();
    const headers = table.getByRole("columnheader");
    await expect(headers.filter({ hasText: /^User$/ })).toHaveCount(1, {
      timeout: 20_000,
    });
    const headerTexts = (await headers.allInnerTexts()).map((text) =>
      text.trim(),
    );
    const userColumn = headerTexts.indexOf("User");
    expect(userColumn, `columns: ${headerTexts.join(", ")}`).toBeGreaterThan(
      -1,
    );

    const search = page
      .getByTestId("datatable-search")
      .filter({ visible: true });
    await expect(search).toBeVisible({ timeout: 20_000 });
    await search.fill(unownedTool);
    const unownedRow = table
      .locator("tbody tr")
      .filter({ hasText: unownedTool });
    await expect(unownedRow).toHaveCount(1, { timeout: 30_000 });
    const unownedCell = unownedRow.getByRole("cell").nth(userColumn);
    await expect(unownedCell).toHaveText("-");
    await expect(unownedCell.getByRole("link")).toHaveCount(0);

    await search.fill(ownedTool);
    const ownedRow = table.locator("tbody tr").filter({ hasText: ownedTool });
    await expect(ownedRow).toHaveCount(1, { timeout: 30_000 });
    const ownerLink = ownedRow
      .getByRole("cell")
      .nth(userColumn)
      .getByRole("link", { name: alias, exact: true });
    await expect(ownerLink).toBeVisible();
    expect(await ownerLink.getAttribute("href")).toContain(
      `user=${encodeURIComponent(userId)}`,
    );
    await ownerLink.click();
    await expect(page).toHaveURL(
      (url) =>
        url.searchParams.get("user") === userId ||
        url.pathname.includes(userId),
    );
  } finally {
    support("clear", ownedTool, unownedTool);
    if (keys.length) await post(request, "/key/delete", { keys });
    if (userId) await post(request, "/user/delete", { user_ids: [userId] });
    if (modelId) await post(request, "/model/delete", { id: modelId });
  }
});
