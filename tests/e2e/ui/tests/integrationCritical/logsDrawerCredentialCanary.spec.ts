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
 * really reached the page) while the page's DOM holds no copy of the canary core in either view,
 * raw or base64-encoded.
 */
const unhex = (): string => randomUUID().replaceAll("-", "");

/**
 * The forms the canary core can take on the page: raw (JSON and percent encoding leave a hex
 * core unchanged), and base64 in the standard and URL-safe alphabets at each of the three byte
 * alignments it can start at. Each base64 form keeps only the characters that depend on core
 * bytes alone, so it matches whatever bytes precede or follow the core.
 */
const canaryForms = (core: string): ReadonlyMap<string, string> => {
  const forms = new Map([["raw", core]]);
  for (let offset = 0; offset < 3; offset++) {
    const bytes = Buffer.concat([Buffer.alloc(offset), Buffer.from(core)]);
    const first = offset === 0 ? 0 : 4;
    const last = Math.floor(bytes.length / 3) * 4;
    const text = bytes.toString("base64").slice(first, last);
    forms.set(`base64@${offset}`, text);
    forms.set(
      `base64url@${offset}`,
      text.replaceAll("+", "-").replaceAll("/", "_"),
    );
  }
  return forms;
};

/** The names of the canary forms found in ``text``; the raw form ignores case. */
const foundForms = (
  text: string,
  forms: ReadonlyMap<string, string>,
): string[] =>
  [...forms]
    .filter(([name, needle]) =>
      name === "raw"
        ? text.toLowerCase().includes(needle)
        : text.includes(needle),
    )
    .map(([name]) => name);

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
  const forms = canaryForms(canaryCore);
  for (const prefix of ["", "k", "k:"]) {
    const encoded = Buffer.from(`${prefix}${deploymentKey}`).toString("base64");
    expect(
      foundForms(`Basic ${encoded}`, forms),
      `the decoder misses base64 after a ${prefix.length}-byte prefix`,
    ).not.toEqual([]);
  }
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
  // A fresh database has no stored value, and the route answers 400 "... is not set".
  const settingText = await setting.text();
  expect(
    setting.status() === 200 || settingText.includes("is not set"),
    settingText,
  ).toBe(true);
  const promptsStored: boolean | null =
    setting.status() === 200
      ? JSON.parse(settingText).field_value === true
      : null;
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
      foundForms(await page.content(), forms),
      "the drawer's pretty view holds the deployment api_key",
    ).toEqual([]);

    await drawer.getByRole("tab", { name: "JSON", exact: true }).click();
    await drawer.getByRole("tab", { name: "Request", exact: true }).click();
    const requestJson = drawer
      .getByRole("tabpanel")
      .filter({ hasText: marker })
      .last();
    await expect(requestJson).toBeVisible({ timeout: 20_000 });
    expect(
      foundForms(await page.content(), forms),
      "the drawer's request JSON holds the deployment api_key",
    ).toEqual([]);
  } finally {
    if (modelId) await post(request, "/model/delete", { id: modelId });
    if (promptsStored === null) {
      await post(request, "/config/field/delete", {
        config_type: "general_settings",
        field_name: "store_prompts_in_spend_logs",
      });
    } else {
      await post(request, "/config/update", {
        general_settings: { store_prompts_in_spend_logs: promptsStored },
      });
    }
  }
});
