import { test, expect } from "@playwright/test";
import { createHash, randomUUID } from "node:crypto";
import { execFileSync } from "node:child_process";
import * as path from "node:path";
import { Page } from "../../fixtures/pages";
import { navigateToPage, openKeyDetail } from "../../helpers/navigation";
import { captureRequestBody, readBack } from "../../helpers/roundTrip";

test("project creation and explicit detachment preserve saved scope and restore serving", async ({
  page,
  request,
}) => {
  const master = process.env.LITELLM_MASTER_KEY ?? "sk-integration-master";
  const headers = { Authorization: `Bearer ${master}` };
  const prefix = `integration-browser-${randomUUID()}`;
  const resources: Array<() => Promise<void>> = [];
  const post = async (url: string, data: object) => {
    const response = await request.post(url, { headers, data });
    expect(response.ok(), `${url}: ${await response.text()}`).toBe(true);
    return response.json();
  };
  const remove = (url: string, data: object) => async () => {
    await post(url, data);
  };
  const saved = (key: string) =>
    JSON.parse(
      execFileSync(
        process.env.INTEGRATION_PYTHON ?? "python",
        [
          path.resolve(
            __dirname,
            "../../../../integration/_support/browser_state.py",
          ),
          createHash("sha256").update(key).digest("hex"),
        ],
        { encoding: "utf8", timeout: 10_000, killSignal: "SIGKILL" },
      ),
    );
  try {
    const previous = await request.get("/get/ui_settings", { headers });
    expect(previous.ok(), await previous.text()).toBe(true);
    const priorEnabled =
      (await previous.json()).values.enable_projects_ui ?? false;
    resources.push(async () => {
      const response = await request.patch("/update/ui_settings", {
        headers,
        data: { enable_projects_ui: priorEnabled },
      });
      expect(response.ok(), await response.text()).toBe(true);
    });
    const settings = await request.patch("/update/ui_settings", {
      headers,
      data: { enable_projects_ui: true },
    });
    expect(settings.ok(), await settings.text()).toBe(true);
    for (const alias of [prefix, `${prefix}-outside`]) {
      const model = await post("/model/new", {
        model_name: alias,
        litellm_params: {
          model: "openai/gpt-4o-mini",
          api_key: "synthetic-provider-key",
          api_base: `${process.env.INTEGRATION_UPSTREAM_URL}/v1`,
        },
        model_info: {},
      });
      resources.push(remove("/model/delete", { id: model.model_info.id }));
    }
    const team = await post("/team/new", {
      team_alias: prefix,
      models: [prefix],
    });
    resources.push(remove("/team/delete", { team_ids: [team.team_id] }));
    const project = await post("/project/new", {
      project_alias: prefix,
      team_id: team.team_id,
      models: [prefix],
    });
    resources.push(async () => {
      const response = await request.delete("/project/delete", {
        headers,
        data: { project_ids: [project.project_id] },
      });
      expect(response.ok(), await response.text()).toBe(true);
    });
    resources.push(async () => {
      const listing = await request.get(
        `/key/list?key_alias=${encodeURIComponent(prefix)}&return_full_object=true`,
        { headers },
      );
      expect(listing.ok(), await listing.text()).toBe(true);
      for (const key of (await listing.json()).keys.filter(
        (key: { key_alias: string }) => key.key_alias === prefix,
      ))
        await post("/key/delete", { keys: [key.token] });
    });
    await page.goto("/ui/login");
    await page.getByPlaceholder("Enter your username").fill("admin");
    await page.getByPlaceholder("Enter your password").fill(master);
    await page.getByRole("button", { name: "Login", exact: true }).click();
    await expect(page).toHaveURL(
      (url) =>
        url.pathname.startsWith("/ui") && !url.pathname.includes("login"),
    );
    await navigateToPage(page, Page.ApiKeys);
    await page.getByRole("button", { name: /Create New Key/i }).click();
    await page.getByLabel(/Key Name/).fill(prefix);
    await page.getByPlaceholder("Search or select a project").fill(prefix);
    await page.getByRole("option", { name: new RegExp(prefix) }).click();
    await page.getByRole("combobox", { name: "Select models" }).click();
    await page.getByRole("option", { name: prefix, exact: true }).click();
    await page.keyboard.press("Escape");
    const creating = page.waitForResponse(
      (response) =>
        response.request().method() === "POST" &&
        new URL(response.url()).pathname === "/key/generate",
    );
    await page.getByRole("button", { name: "Create Key", exact: true }).click();
    const created = await creating;
    expect(created.ok(), await created.text()).toBe(true);
    const createBody = created.request().postDataJSON();
    expect(createBody.project_id).toBe(project.project_id);
    expect(createBody.team_id).toBe(team.team_id);
    const key = (await created.json()).key as string;
    expect(saved(key)).toEqual([
      {
        project_id: project.project_id,
        team_id: team.team_id,
        models: [prefix],
      },
    ]);
    await expect(
      page.getByText("Save your Key", { exact: true }),
    ).toBeVisible();
    await page.keyboard.press("Escape");
    const chat = (model: string) =>
      request.post("/v1/chat/completions", {
        headers: { Authorization: `Bearer ${key}` },
        data: {
          model,
          messages: [{ role: "user", content: "synthetic browser control" }],
        },
      });
    const first = await chat(prefix);
    expect(first.status(), await first.text()).toBe(200);
    expect((await first.json()).usage.total_tokens).toBe(40);
    await post("/project/update", {
      project_id: project.project_id,
      blocked: true,
    });
    const blocked = await chat(prefix);
    expect(blocked.status(), await blocked.text()).toBe(401);
    expect((await blocked.json()).error.type).toBe("auth_error");
    await openKeyDetail(page, prefix);
    await page.getByRole("tab", { name: "Settings" }).click();
    await page.getByRole("button", { name: "Edit Settings" }).click();
    await page
      .getByRole("button", { name: "Detach from project", exact: true })
      .click();
    const update = await captureRequestBody(
      page,
      { method: "POST", urlIncludes: "/key/update" },
      async () => {
        await page.getByRole("button", { name: "Save Changes" }).click();
      },
    );
    expect(update.project_id).toBeNull();
    await expect(
      page.getByRole("button", { name: "Edit Settings" }),
    ).toBeVisible();
    await page.reload();
    const info = await readBack<{
      info: { project_id: string | null; team_id: string; models: string[] };
    }>(page, `/key/info?key=${encodeURIComponent(key)}`);
    expect(info.info.project_id).toBeNull();
    expect(saved(key)).toEqual([
      { project_id: null, team_id: team.team_id, models: [prefix] },
    ]);
    await page.getByRole("tab", { name: "Settings" }).click();
    await page.getByRole("button", { name: "Edit Settings" }).click();
    await expect(
      page.getByRole("button", { name: "Detach from project" }),
    ).toHaveCount(0);
    const restored = await chat(prefix);
    expect(restored.status(), await restored.text()).toBe(200);
    expect((await restored.json()).usage.total_tokens).toBe(40);
    const outside = await chat(`${prefix}-outside`);
    expect(outside.status(), await outside.text()).toBe(403);
    expect((await outside.json()).error.type).toBe("key_model_access_denied");
    await post("/key/delete", { keys: [key] });
    expect(saved(key)).toEqual([]);
  } finally {
    const failures = [];
    for (const cleanup of resources.reverse()) {
      try {
        await cleanup();
      } catch (error) {
        failures.push(error);
      }
    }
    expect(failures).toEqual([]);
  }
});
