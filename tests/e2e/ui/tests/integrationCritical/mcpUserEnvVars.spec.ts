import {
  test,
  expect,
  APIRequestContext,
  Locator,
  Page as PlaywrightPage,
} from "@playwright/test";
import { randomUUID } from "node:crypto";
import { Page } from "../../fixtures/pages";
import { navigateToPage } from "../../helpers/navigation";
import { captureRequestBody } from "../../helpers/roundTrip";

const master = process.env.LITELLM_MASTER_KEY ?? "sk-integration-master";
const headers = { Authorization: `Bearer ${master}` };
const TOKEN = "USER_TOKEN";

type EnvVarStatus = {
  missing_count: number;
  required: { name: string; is_set: boolean }[];
};

type Server = {
  name: string;
  id: string;
  statusUrl: string;
  status: () => Promise<EnvVarStatus>;
  remove: () => Promise<void>;
};

async function createServer(
  request: APIRequestContext,
  variables: string[],
): Promise<Server> {
  const name = `int_mcp_${randomUUID().replace(/-/g, "").slice(0, 12)}`;
  const created = await request.post("/v1/mcp/server", {
    headers,
    data: {
      server_name: name,
      url: `${process.env.INTEGRATION_UPSTREAM_URL}/mcp`,
      transport: "http",
      auth_type: "none",
      env_vars: variables.map((variable) => ({
        name: variable,
        scope: "user",
        description: `Per-user ${variable}`,
      })),
      static_headers: Object.fromEntries(
        variables.map((variable, index) => [
          `X-User-${index}`,
          `\${${variable}}`,
        ]),
      ),
    },
  });
  expect(created.ok(), await created.text()).toBe(true);
  const id = (await created.json()).server_id as string;
  const statusUrl = `/v1/mcp/server/${id}/user-env-vars`;
  return {
    name,
    id,
    statusUrl,
    status: async () => {
      const response = await request.get(statusUrl, { headers });
      expect(response.ok(), await response.text()).toBe(true);
      return response.json() as Promise<EnvVarStatus>;
    },
    remove: async () => {
      const removed = await request.delete(`/v1/mcp/server/${id}`, { headers });
      expect(
        removed.ok() || removed.status() === 404,
        await removed.text(),
      ).toBe(true);
    },
  };
}

async function openMcpServers(page: PlaywrightPage): Promise<void> {
  await page.goto("/ui/login");
  await page.getByPlaceholder("Enter your username").fill("admin");
  await page.getByPlaceholder("Enter your password").fill(master);
  await page.getByRole("button", { name: "Login", exact: true }).click();
  await expect(page).toHaveURL(
    (url) => url.pathname.startsWith("/ui") && !url.pathname.includes("login"),
  );
  await navigateToPage(page, Page.McpServers);
}

function cardFor(page: PlaywrightPage, server: Server): Locator {
  return page.getByRole("button").filter({ hasText: server.name }).first();
}

function credentialsDialog(page: PlaywrightPage): Locator {
  return page.getByRole("dialog").filter({ hasText: "Set your credentials" });
}

async function saveValues(
  page: PlaywrightPage,
  server: Server,
  values: Record<string, string>,
): Promise<void> {
  const dialog = credentialsDialog(page);
  for (const [variable, value] of Object.entries(values)) {
    await dialog.getByLabel(variable).fill(value);
  }
  const body = await captureRequestBody(
    page,
    { method: "POST", urlIncludes: server.statusUrl },
    async () => {
      await dialog.getByRole("button", { name: "Save Credentials" }).click();
    },
  );
  expect(body).toEqual({ values });
  await expect(dialog).toHaveCount(0);
}

test("per-user MCP env var stays updatable and clearable from the card after it is set", async ({
  page,
  request,
}) => {
  const server = await createServer(request, [TOKEN]);
  try {
    await openMcpServers(page);
    const card = cardFor(page, server);
    const dialog = credentialsDialog(page);
    await expect(
      card.getByText("1 user field missing", { exact: true }),
    ).toBeVisible();

    await card.getByRole("button", { name: "Set", exact: true }).click();
    await saveValues(page, server, { [TOKEN]: "first-token" });
    expect(await server.status()).toMatchObject({
      missing_count: 0,
      required: [{ name: TOKEN, is_set: true }],
    });

    await expect(
      card.getByText("1 user field missing", { exact: true }),
    ).toHaveCount(0);
    await page.reload();
    const update = card.getByRole("button", { name: "Update", exact: true });
    await expect(
      update,
      "a set per-user variable must keep an update entry point on the card",
    ).toBeVisible();
    await update.click();
    await expect(dialog.getByText("Set", { exact: true })).toBeVisible();
    await saveValues(page, server, { [TOKEN]: "rotated-token" });
    expect(await server.status()).toMatchObject({
      missing_count: 0,
      required: [{ name: TOKEN, is_set: true }],
    });

    await update.click();
    const cleared = page.waitForResponse(
      (response) =>
        response.request().method() === "DELETE" &&
        response.url().includes(server.statusUrl),
    );
    await dialog.getByRole("button", { name: "Clear", exact: true }).click();
    const confirm = page.getByRole("alertdialog", {
      name: "Clear saved credentials",
    });
    await expect(confirm).toContainText(server.name);
    await confirm
      .getByRole("button", { name: "Clear credentials", exact: true })
      .click();
    const clearResponse = await cleared;
    expect(clearResponse.ok(), await clearResponse.text()).toBe(true);
    await expect(dialog).toHaveCount(0);
    expect(await server.status()).toMatchObject({
      missing_count: 1,
      required: [{ name: TOKEN, is_set: false }],
    });
    await expect(
      card.getByText("1 user field missing", { exact: true }),
    ).toBeVisible();
    await expect(
      card.getByRole("button", { name: "Set", exact: true }),
    ).toBeVisible();
  } finally {
    await server.remove();
  }
});

test("cancelling the clear confirmation keeps the stored value and sends no delete", async ({
  page,
  request,
}) => {
  const server = await createServer(request, [TOKEN]);
  try {
    const stored = await request.post(server.statusUrl, {
      headers,
      data: { values: { [TOKEN]: "keep-me" } },
    });
    expect(stored.ok(), await stored.text()).toBe(true);
    await openMcpServers(page);
    const card = cardFor(page, server);
    const dialog = credentialsDialog(page);
    const deletes: string[] = [];
    page.on("request", (sent) => {
      if (sent.method() === "DELETE" && sent.url().includes(server.statusUrl))
        deletes.push(sent.url());
    });
    await card.getByRole("button", { name: "Update", exact: true }).click();
    await dialog.getByRole("button", { name: "Clear", exact: true }).click();
    const confirm = page.getByRole("alertdialog", {
      name: "Clear saved credentials",
    });
    await expect(confirm).toBeVisible();
    await confirm.getByRole("button", { name: "Cancel", exact: true }).click();
    await expect(confirm).toHaveCount(0);
    await expect(
      dialog,
      "cancelling the confirmation must leave the credentials modal open",
    ).toBeVisible();
    await dialog.getByRole("button", { name: "Cancel", exact: true }).click();
    await expect(dialog).toHaveCount(0);
    await card.getByRole("button", { name: "Update", exact: true }).click();
    await expect(
      confirm,
      "a cancelled confirmation must not reappear on reopen",
    ).toHaveCount(0);
    await page.keyboard.press("Escape");
    await expect(dialog).toHaveCount(0);
    expect(deletes).toEqual([]);
    expect(await server.status()).toMatchObject({
      missing_count: 0,
      required: [{ name: TOKEN, is_set: true }],
    });
    await expect(
      card.getByRole("button", { name: "Update", exact: true }),
    ).toBeVisible();
  } finally {
    await server.remove();
  }
});

test("pressing Enter on Update opens the credentials modal instead of the server editor", async ({
  page,
  request,
}) => {
  const server = await createServer(request, [TOKEN]);
  try {
    const stored = await request.post(server.statusUrl, {
      headers,
      data: { values: { [TOKEN]: "keyboard" } },
    });
    expect(stored.ok(), await stored.text()).toBe(true);
    await openMcpServers(page);
    const card = cardFor(page, server);
    const update = card.getByRole("button", { name: "Update", exact: true });
    await update.focus();
    await page.keyboard.press("Enter");
    const dialog = credentialsDialog(page);
    await expect(dialog).toBeVisible();
    await expect(
      page.getByRole("button", { name: "Back to All Servers" }),
    ).toHaveCount(0);
    await saveValues(page, server, { [TOKEN]: "keyboard-rotated" });
    await expect(
      page.getByRole("button", { name: "Back to All Servers" }),
    ).toHaveCount(0);
    await expect(card).toBeVisible();
    expect(await server.status()).toMatchObject({
      missing_count: 0,
      required: [{ name: TOKEN, is_set: true }],
    });
    await card.click();
    await expect(
      page.getByRole("button", { name: "Back to All Servers" }),
    ).toBeVisible();
  } finally {
    await server.remove();
  }
});

test("a server with two per-user variables reports the remaining gap until both are saved", async ({
  page,
  request,
}) => {
  const second = "WORKSPACE";
  const server = await createServer(request, [TOKEN, second]);
  try {
    await openMcpServers(page);
    const card = cardFor(page, server);
    const dialog = credentialsDialog(page);
    await expect(
      card.getByText("2 user fields missing", { exact: true }),
    ).toBeVisible();
    await card.getByRole("button", { name: "Set", exact: true }).click();
    await dialog.getByLabel(TOKEN).fill("only-token");
    const posts: string[] = [];
    page.on("request", (sent) => {
      if (sent.method() === "POST" && sent.url().includes(server.statusUrl))
        posts.push(sent.url());
    });
    await dialog.getByRole("button", { name: "Save Credentials" }).click();
    await expect(dialog.getByRole("alert")).toHaveText(`${second} is required`);
    expect(posts, "a missing required field must block the save").toEqual([]);
    await dialog.getByRole("button", { name: "Cancel", exact: true }).click();
    await expect(dialog).toHaveCount(0);

    const partial = await request.post(server.statusUrl, {
      headers,
      data: { values: { [TOKEN]: "only-token" } },
    });
    expect(partial.ok(), await partial.text()).toBe(true);
    await page.reload();
    await expect(
      card.getByText("1 user field missing", { exact: true }),
    ).toBeVisible();
    await expect(
      card.getByRole("button", { name: "Update", exact: true }),
    ).toHaveCount(0);
    await card.getByRole("button", { name: "Set", exact: true }).click();
    await expect(dialog.getByText("Set", { exact: true })).toHaveCount(1);
    await saveValues(page, server, {
      [TOKEN]: "",
      [second]: "workspace-value",
    });
    expect(await server.status()).toMatchObject({
      missing_count: 0,
      required: [
        { name: TOKEN, is_set: true },
        { name: second, is_set: true },
      ],
    });
    await expect(
      card.getByRole("button", { name: "Update", exact: true }),
    ).toBeVisible();
    await expect(card.getByText(/user fields? missing/)).toHaveCount(0);
  } finally {
    await server.remove();
  }
});

test("a server without per-user variables shows no credential row", async ({
  page,
  request,
}) => {
  const server = await createServer(request, []);
  const withVariable = await createServer(request, [TOKEN]);
  try {
    await openMcpServers(page);
    const plain = cardFor(page, server);
    await expect(plain).toBeVisible();
    await expect(
      cardFor(page, withVariable).getByRole("button", {
        name: "Set",
        exact: true,
      }),
    ).toBeVisible();
    await expect(plain.getByText("Per-user credentials")).toHaveCount(0);
    await expect(
      plain.getByRole("button", { name: "Set", exact: true }),
    ).toHaveCount(0);
    await expect(
      plain.getByRole("button", { name: "Update", exact: true }),
    ).toHaveCount(0);
    await expect(plain.getByText(/user fields? missing/)).toHaveCount(0);
  } finally {
    await server.remove();
    await withVariable.remove();
  }
});

test("clearing credentials for a server deleted underneath the modal reports the failure without losing the page", async ({
  page,
  request,
}) => {
  const server = await createServer(request, [TOKEN]);
  const survivor = await createServer(request, [TOKEN]);
  try {
    const stored = await request.post(server.statusUrl, {
      headers,
      data: { values: { [TOKEN]: "doomed" } },
    });
    expect(stored.ok(), await stored.text()).toBe(true);
    await openMcpServers(page);
    const card = cardFor(page, server);
    const dialog = credentialsDialog(page);
    await card.getByRole("button", { name: "Update", exact: true }).click();
    await expect(dialog).toBeVisible();
    await server.remove();
    const cleared = page.waitForResponse(
      (response) =>
        response.request().method() === "DELETE" &&
        response.url().includes(server.statusUrl),
    );
    await dialog.getByRole("button", { name: "Clear", exact: true }).click();
    await page
      .getByRole("alertdialog", { name: "Clear saved credentials" })
      .getByRole("button", {
        name: "Clear credentials",
        exact: true,
      })
      .click();
    const clearResponse = await cleared;
    expect(clearResponse.status()).toBe(404);
    await expect(page.getByText(/Failed to clear env vars/)).toBeVisible();
    await expect(
      dialog,
      "a failed clear must keep the modal open for the user",
    ).toBeVisible();
    await page.keyboard.press("Escape");
    await expect(dialog).toHaveCount(0);
    await page.reload();
    await expect(
      cardFor(page, survivor).getByRole("button", { name: "Set", exact: true }),
    ).toBeVisible();
    await expect(page.getByText(server.name)).toHaveCount(0);
  } finally {
    await server.remove();
    await survivor.remove();
  }
});
