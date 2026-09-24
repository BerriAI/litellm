import { test, expect } from "@playwright/test";
import { randomUUID } from "node:crypto";
import { Page } from "../../fixtures/pages";
import { navigateToPage } from "../../helpers/navigation";
import { captureRequestBody, readBack } from "../../helpers/roundTrip";

test("team logging settings persist langfuse_span_scope no_internal through the callback editor", async ({
  page,
  request,
}) => {
  const master = process.env.LITELLM_MASTER_KEY ?? "sk-integration-master";
  const headers = { Authorization: `Bearer ${master}` };
  const prefix = `integration-browser-${randomUUID()}`;
  const post = async (url: string, data: object) => {
    const response = await request.post(url, { headers, data });
    expect(response.ok(), `${url}: ${await response.text()}`).toBe(true);
    return response.json();
  };
  const team = await post("/team/new", { team_alias: prefix });
  try {
    await page.goto("/ui/login");
    await page.getByPlaceholder("Enter your username").fill("admin");
    await page.getByPlaceholder("Enter your password").fill(master);
    await page.getByRole("button", { name: "Login", exact: true }).click();
    await expect(page).toHaveURL(
      (url) => url.pathname.startsWith("/ui") && !url.pathname.includes("login"),
    );
    await navigateToPage(page, Page.Teams);
    await page.getByRole("button", { name: new RegExp(`^${prefix}`) }).click();
    await page.getByRole("tab", { name: "Settings" }).click();
    await page.getByRole("button", { name: "Edit Settings" }).click();
    await page
      .getByRole("button", { name: "Add Integration", exact: true })
      .click();
    await page
      .getByRole("combobox")
      .filter({ hasText: "Select integration" })
      .click();
    await page
      .getByRole("option", { name: /Langfuse OTEL/ })
      .click();
    const spanScope = page.getByRole("combobox", {
      name: "langfuse span scope",
    });
    await spanScope.click();
    await expect(
      page.getByRole("option", { name: "full", exact: true }),
    ).toBeVisible();
    await page
      .getByRole("option", { name: "no_internal", exact: true })
      .click();
    const update = await captureRequestBody(
      page,
      { method: "POST", urlIncludes: "/team/update" },
      async () => {
        await page.getByRole("button", { name: "Save Changes" }).click();
      },
    );
    const logging = (update.metadata as { logging: any[] }).logging;
    expect(logging).toEqual([
      {
        callback_name: "langfuse_otel",
        callback_type: "success",
        callback_vars: { langfuse_span_scope: "no_internal" },
      },
    ]);
    await expect(
      page.getByText("Team settings updated successfully"),
    ).toBeVisible();
    const info = await readBack<{
      team_info: { metadata: { logging: any[] } };
    }>(page, `/team/info?team_id=${team.team_id}`);
    expect(
      info.team_info.metadata.logging[0].callback_vars.langfuse_span_scope,
    ).toBe("no_internal");
  } finally {
    await post("/team/delete", { team_ids: [team.team_id] });
  }
});
