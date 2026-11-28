import { expect, test } from "@playwright/test";
import { ADMIN_STORAGE_PATH } from "../constants";
import { Page } from "../fixtures/pages";
import { navigateToPage } from "../helpers/navigation";
import { masterKey, rootPath, uniqueSuffix } from "../helpers/traffic";

test.use({ storageState: ADMIN_STORAGE_PATH });

test("Creates an agent and saves its new name", async ({ page, request }) => {
  const name = `e2e_agent_${uniqueSuffix().replaceAll("-", "_")}`;
  const renamed = `${name}_renamed`;
  try {
    await navigateToPage(page, Page.Agents);
    await page.getByRole("button", { name: "Add New Agent" }).click();
    const wizard = page.getByRole("dialog", { name: "Add New Agent" });
    await wizard.getByLabel("Agent Name", { exact: true }).fill(name);
    await wizard.getByLabel("Display Name", { exact: true }).fill(name);
    await wizard.getByLabel("Description", { exact: true }).fill("Agent created by the dashboard browser test");
    for (let step = 0; step < 3; step++) {
      await wizard.getByRole("button", { name: /^Next/ }).click();
    }
    await wizard.getByRole("button", { name: /Skip for now/ }).click();
    await wizard.getByRole("button", { name: /^Create Agent/ }).click();
    await wizard.getByRole("button", { name: "Done", exact: true }).click();

    const agentRow = page.getByRole("row").filter({ has: page.getByRole("cell", { name, exact: true }) });
    await agentRow.getByRole("button", { name: /^[\da-f-]{36}$/i }).click();
    await expect(page.getByRole("heading", { name, exact: true })).toBeVisible();

    await page.getByRole("tab", { name: "Settings", exact: true }).click();
    await page.getByRole("button", { name: "Edit Settings", exact: true }).click();
    await page.getByLabel("Agent Name", { exact: true }).fill(renamed);
    await page.getByRole("button", { name: "Save Changes", exact: true }).click();
    await expect(page.getByRole("heading", { name: renamed, exact: true })).toBeVisible();
  } finally {
    const url = `${rootPath()}/v1/agents`;
    const headers = { Authorization: `Bearer ${masterKey()}` };
    const response = await request.get(url, { headers, params: { health_check: "false" } });
    if (response.ok()) {
      const agents: { agent_id: string; agent_name: string }[] = await response.json();
      for (const agent of agents.filter(({ agent_name }) => agent_name === name || agent_name === renamed)) {
        await request.delete(`${url}/${agent.agent_id}`, { headers });
      }
    }
  }
});
