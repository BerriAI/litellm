import { test, expect, type APIRequestContext } from "@playwright/test";
import { randomUUID } from "node:crypto";
import { Page } from "../../fixtures/pages";
import { dismissFeedbackPopup, navigateToPage } from "../../helpers/navigation";

/**
 * Credential canary S8: what the Logs page renders for a request, including any client-side
 * merge, never shows the deployment api_key that served it.
 *
 * A deployment is registered with a fresh canary api_key pointing at the owned upstream. The
 * upstream must receive that canary as its bearer (positive control). The request carries a
 * marker in its message content with stored prompts on, and the drawer must render that marker
 * in both its pretty view and its raw request JSON view (sensitivity control: the stored request
 * really reached the page) while the page's DOM holds no copy of the canary core in either view.
 */
const unhex = (): string => randomUUID().replaceAll("-", "");

test("the Logs drawer renders the stored request without the deployment api_key", async ({
  page,
  request,
}) => {
  const master = process.env.LITELLM_MASTER_KEY ?? "sk-integration-master";
  const upstream = (
    process.env.INTEGRATION_UPSTREAM_URL ?? "http://127.0.0.1:8190"
  ).replace(/\/+$/, "");
  const auth = { Authorization: `Bearer ${master}` };
  const canaryCore = unhex();
  const deploymentKey = `lkc-B1-${canaryCore}`;
  const marker = `lkc-M0-${unhex()}`;
  const model = `canary-drawer-${unhex()}`;

  const post = async (api: APIRequestContext, path: string, data: object) => {
    const response = await api.post(path, { headers: auth, data });
    expect(response.status(), `POST ${path}: ${await response.text()}`).toBe(
      200,
    );
    return response.json();
  };

  const setting = await request.get(
    "/config/field/info?field_name=store_prompts_in_spend_logs",
    { headers: auth },
  );
  expect(setting.status(), await setting.text()).toBe(200);
  const promptsStored: boolean = (await setting.json()).field_value === true;
  let modelId = "";
  try {
    await post(request, "/config/update", {
      general_settings: { store_prompts_in_spend_logs: true },
    });
    const created = await post(request, "/model/new", {
      model_name: model,
      litellm_params: {
        model: "openai/gpt-4o-mini",
        api_key: deploymentKey,
        api_base: `${upstream}/v1`,
      },
    });
    modelId = created.model_id;
    let requestId = "";
    await expect
      .poll(
        async () => {
          const response = await request.post("/v1/chat/completions", {
            headers: auth,
            data: {
              model,
              messages: [{ role: "user", content: `drawer ${marker}` }],
            },
          });
          if (response.status() === 200) requestId = (await response.json()).id;
          return response.status();
        },
        {
          timeout: 30_000,
          message: "the new deployment never served the request",
        },
      )
      .toBe(200);

    const observed = await request.get(`${upstream}/__observations`);
    const delivered = (
      (await observed.json()).requests as {
        authorization: string;
        body: unknown;
      }[]
    ).filter((entry) => JSON.stringify(entry.body).includes(marker));
    expect(
      delivered.map((entry) => entry.authorization),
      "Positive control: the upstream never received the deployment key",
    ).toEqual([`Bearer ${deploymentKey}`]);

    await expect
      .poll(
        async () => {
          const response = await request.get(
            `/spend/logs/ui/${encodeURIComponent(requestId)}`,
            { headers: auth },
          );
          return response.status() === 200
            ? JSON.stringify(await response.json()).includes(marker)
            : false;
        },
        {
          timeout: 70_000,
          message: `the stored request for ${requestId} never carried the marker`,
        },
      )
      .toBe(true);

    await page.goto("/ui/login");
    await page.getByPlaceholder("Enter your username").fill("admin");
    await page.getByPlaceholder("Enter your password").fill(master);
    await page.getByRole("button", { name: "Login", exact: true }).click();
    await expect(page).toHaveURL(
      (url) =>
        url.pathname.startsWith("/ui") && !url.pathname.includes("login"),
    );
    await navigateToPage(page, Page.Logs);
    await dismissFeedbackPopup(page);

    const search = page
      .getByTestId("datatable-search")
      .filter({ visible: true });
    await expect(search).toBeVisible({ timeout: 20_000 });
    await search.fill(requestId);
    const row = page
      .locator("table")
      .filter({ visible: true })
      .first()
      .locator("tbody tr")
      .filter({ hasText: requestId });
    await expect(row).toHaveCount(1, { timeout: 30_000 });
    await row.click();

    const drawer = page.getByRole("dialog").first();
    await expect(drawer.getByText("Request & Response")).toBeVisible({
      timeout: 20_000,
    });
    await expect(
      drawer.getByText(marker, { exact: false }).first(),
    ).toBeVisible({ timeout: 20_000 });
    expect(
      (await page.content()).includes(canaryCore),
      "the drawer's pretty view holds the deployment api_key",
    ).toBe(false);

    await drawer.getByRole("tab", { name: "JSON", exact: true }).click();
    await drawer.getByRole("tab", { name: "Request", exact: true }).click();
    const requestJson = drawer
      .getByRole("tabpanel")
      .filter({ hasText: marker })
      .last();
    await expect(requestJson).toBeVisible({ timeout: 20_000 });
    expect(
      (await page.content()).includes(canaryCore),
      "the drawer's request JSON holds the deployment api_key",
    ).toBe(false);
  } finally {
    if (modelId) await post(request, "/model/delete", { id: modelId });
    await post(request, "/config/update", {
      general_settings: { store_prompts_in_spend_logs: promptsStored },
    });
  }
});
