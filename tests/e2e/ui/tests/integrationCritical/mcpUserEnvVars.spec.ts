import { test, expect } from "@playwright/test";
import { randomUUID } from "node:crypto";
import { Page } from "../../fixtures/pages";
import { navigateToPage } from "../../helpers/navigation";
import { captureRequestBody } from "../../helpers/roundTrip";

test("per-user MCP env var stays updatable and clearable from the card after it is set", async ({
  page,
  request,
}) => {
  const master = process.env.LITELLM_MASTER_KEY ?? "sk-integration-master";
  const headers = { Authorization: `Bearer ${master}` };
  const serverName = `int_mcp_${randomUUID().replace(/-/g, "").slice(0, 12)}`;
  const variable = "USER_TOKEN";
  const created = await request.post("/v1/mcp/server", {
    headers,
    data: {
      server_name: serverName,
      url: `${process.env.INTEGRATION_UPSTREAM_URL}/mcp`,
      transport: "http",
      auth_type: "none",
      env_vars: [
        {
          name: variable,
          scope: "user",
          description: "Per-user upstream token",
        },
      ],
      static_headers: { Authorization: `Bearer \${${variable}}` },
    },
  });
  expect(created.ok(), await created.text()).toBe(true);
  const serverId = (await created.json()).server_id as string;
  const statusUrl = `/v1/mcp/server/${serverId}/user-env-vars`;
  const savedStatus = async () => {
    const response = await request.get(statusUrl, { headers });
    expect(response.ok(), await response.text()).toBe(true);
    return response.json() as Promise<{
      missing_count: number;
      required: { name: string; is_set: boolean }[];
    }>;
  };
  try {
    await page.goto("/ui/login");
    await page.getByPlaceholder("Enter your username").fill("admin");
    await page.getByPlaceholder("Enter your password").fill(master);
    await page.getByRole("button", { name: "Login", exact: true }).click();
    await expect(page).toHaveURL(
      (url) =>
        url.pathname.startsWith("/ui") && !url.pathname.includes("login"),
    );
    await navigateToPage(page, Page.McpServers);
    const card = page
      .getByRole("button")
      .filter({ hasText: serverName })
      .first();
    await expect(
      card.getByText("1 user field missing", { exact: true }),
    ).toBeVisible();

    await card.getByRole("button", { name: "Set", exact: true }).click();
    const dialog = page
      .getByRole("dialog")
      .filter({ hasText: "Set your credentials" });
    await dialog.getByLabel(variable).fill("first-token");
    const firstSave = await captureRequestBody(
      page,
      { method: "POST", urlIncludes: statusUrl },
      async () => {
        await dialog.getByRole("button", { name: "Save Credentials" }).click();
      },
    );
    expect(firstSave).toEqual({ values: { [variable]: "first-token" } });
    await expect(dialog).toHaveCount(0);
    expect(await savedStatus()).toMatchObject({
      missing_count: 0,
      required: [{ name: variable, is_set: true }],
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
    await dialog.getByLabel(variable).fill("rotated-token");
    const secondSave = await captureRequestBody(
      page,
      { method: "POST", urlIncludes: statusUrl },
      async () => {
        await dialog.getByRole("button", { name: "Save Credentials" }).click();
      },
    );
    expect(secondSave).toEqual({ values: { [variable]: "rotated-token" } });
    await expect(dialog).toHaveCount(0);
    expect(await savedStatus()).toMatchObject({
      missing_count: 0,
      required: [{ name: variable, is_set: true }],
    });

    await update.click();
    const cleared = page.waitForResponse(
      (response) =>
        response.request().method() === "DELETE" &&
        response.url().includes(statusUrl),
    );
    await dialog.getByRole("button", { name: "Clear", exact: true }).click();
    const clearResponse = await cleared;
    expect(clearResponse.ok(), await clearResponse.text()).toBe(true);
    await expect(dialog).toHaveCount(0);
    expect(await savedStatus()).toMatchObject({
      missing_count: 1,
      required: [{ name: variable, is_set: false }],
    });
    await expect(
      card.getByText("1 user field missing", { exact: true }),
    ).toBeVisible();
    await expect(
      card.getByRole("button", { name: "Set", exact: true }),
    ).toBeVisible();
  } finally {
    const removed = await request.delete(`/v1/mcp/server/${serverId}`, {
      headers,
    });
    expect(removed.ok(), await removed.text()).toBe(true);
  }
});
