import { test, expect, type APIRequestContext, type Locator, type Page as PlaywrightPage } from "@playwright/test";
import { ADMIN_STORAGE_PATH } from "../../constants";
import { navigateToPage, dismissFeedbackPopup } from "../../helpers/navigation";
import { Page } from "../../fixtures/pages";
import {
  createVirtualKey,
  masterKey,
  rootPath,
  sendChatCompletion,
  waitForKeyInDailyActivity,
  waitForSpendLog,
} from "../../helpers/traffic";

/** Covers /ui/usage. The legacy /ui/old-usage view is deprecated and deliberately not covered. */

/** Stepping up from the title is exact; the page renders several other tables. */
const topKeysCard = (page: PlaywrightPage): Locator =>
  page.getByText("Top Virtual Keys", { exact: true }).locator("xpath=..");

async function openUsage(page: PlaywrightPage): Promise<Locator> {
  await navigateToPage(page, Page.NewUsage);
  await dismissFeedbackPopup(page);
  const card = topKeysCard(page);
  await expect(card).toBeVisible({ timeout: 30_000 });
  // Widen past the default top-5 so other keys in the database cannot crowd this one out.
  // The radio itself is sr-only and its label covers it, so click the label.
  await card.getByRole("radiogroup", { name: "Number of top keys to show" }).getByText("50", { exact: true }).click();
  return card;
}

/** The upstream fixtures/config.yml points its models at, so the mock server answers this too. */
const MOCK_DEPLOYMENT = "openai/fake-gpt-4";

/** A deployment whose traffic costs real money, so the key that used it outranks the $0 crowd. */
async function createPricedDeployment(
  request: APIRequestContext,
  label: string,
  registerForCleanup: string[],
): Promise<{ modelName: string }> {
  const modelName = `e2e-usage-priced-${label}`;
  // Claimed before the call: /model/new can persist the deployment and still answer non-2xx, so a
  // name recorded up front is the only registration no response shape can skip.
  registerForCleanup.push(modelName);
  const res = await request.post("/model/new", {
    headers: { Authorization: `Bearer ${masterKey()}`, "Content-Type": "application/json" },
    data: {
      model_name: modelName,
      litellm_params: {
        model: MOCK_DEPLOYMENT,
        api_base: `http://127.0.0.1:${process.env.MOCK_LLM_PORT ?? "8090"}/v1`,
        api_key: "fake-key",
        input_cost_per_token: 0.01,
        output_cost_per_token: 0.01,
      },
    },
  });
  expect(res.ok(), `POST /model/new failed (${res.status()}): ${await res.text()}`).toBe(true);

  // /model/new returns once the row is written, but the router only picks the deployment up on its
  // next refresh, so sending traffic straight away can still get "no healthy deployments". A ping
  // that fails writes no spend log, so retrying it costs the ranking this test asserts nothing.
  await expect
    .poll(
      async () => {
        const ping = await request.post(`${rootPath()}/v1/chat/completions`, {
          headers: { Authorization: `Bearer ${masterKey()}`, "Content-Type": "application/json" },
          data: { model: modelName, messages: [{ role: "user", content: "readiness ping" }] },
        });
        return ping.ok();
      },
      { message: `deployment ${modelName} never became routable`, timeout: 60_000 },
    )
    .toBe(true);

  return { modelName };
}

test.describe("Usage page", () => {
  test.use({ storageState: ADMIN_STORAGE_PATH });

  const pricedDeployments: string[] = [];

  test.afterEach(async ({ request }) => {
    // A deployment left behind keeps its custom pricing, so it goes on changing what later runs
    // route and what they cost. Runs on the failure path too, which the test body would not.
    // Resolved by name rather than by a returned id, so a create that persisted without answering
    // 2xx is still cleaned up. /model/info serves the router, and /model/new answers 2xx even when
    // its in-request router reload failed, so the search-backed listing is what covers a deployment
    // that reached the database only. Absent from both means it never persisted.
    const names = pricedDeployments.splice(0);
    if (names.length === 0) return;
    const auth = { Authorization: `Bearer ${masterKey()}`, "Content-Type": "application/json" };

    const idIn = async (path: string, label: string, name: string): Promise<string | undefined> => {
      const listed = await request.get(path, { headers: auth });
      expect(listed.ok(), `GET ${label} (${listed.status()})`).toBe(true);
      const deployments = ((await listed.json()).data ?? []) as {
        model_name?: string;
        model_info?: { id?: string };
      }[];
      return deployments.find((d) => d.model_name === name)?.model_info?.id;
    };

    for (const name of names) {
      const search = encodeURIComponent(name);
      const id =
        (await idIn(`${rootPath()}/model/info`, "/model/info", name)) ??
        (await idIn(`${rootPath()}/v2/model/info?search=${search}`, "/v2/model/info", name));
      if (id === undefined) continue;
      const deleted = await request.post(`${rootPath()}/model/delete`, {
        headers: auth,
        data: { id },
      });
      expect(deleted.ok(), `POST /model/delete for ${name} (${deleted.status()})`).toBe(true);
    }
  });

  test("Top Virtual Keys lists a key that served traffic, toggles views, and opens key info", async ({
    page,
    request,
  }) => {
    const alias = `e2e-usage-key-${Date.now()}`;
    const { key, token } = await createVirtualKey(request, {
      key_alias: alias,
    });

    // Top Virtual Keys ranks by spend, and every mock deployment costs $0, so once a run has more
    // keys than the list shows, whether this one makes the cut is down to how ties happen to sort.
    // Give it a priced deployment of its own so it earns its place.
    const { modelName } = await createPricedDeployment(request, alias, pricedDeployments);

    const requestId = await sendChatCompletion(request, {
      model: modelName,
      prompt: `usage ping for ${alias}`,
      apiKey: key,
    });
    await waitForSpendLog(request, requestId);
    // Must land in the aggregate before the page mounts — it fetches once.
    await waitForKeyInDailyActivity(request, token);

    const card = await openUsage(page);

    // Table view (the default): the key is listed by its alias.
    const row = card.getByRole("row").filter({ hasText: alias });
    await expect(row, `${alias} missing from Top Virtual Keys`).toHaveCount(1, {
      timeout: 30_000,
    });

    // Chart view swaps the table out for the bar chart, and back.
    await card.getByText("Chart View", { exact: true }).click();
    await expect(card.getByRole("table")).toHaveCount(0, { timeout: 10_000 });
    await card.getByText("Table View", { exact: true }).click();
    await expect(row).toHaveCount(1, { timeout: 10_000 });

    // The alias is already in the row behind the modal, so match the panel's own controls.
    await row.getByRole("button", { name: token }).click();
    const keyInfo = page.getByRole("tab", { name: "Overview", exact: true });
    await expect(keyInfo, "key info panel did not open").toBeVisible({
      timeout: 20_000,
    });
    await expect(page.getByRole("tab", { name: "Settings", exact: true })).toBeVisible();
    await expect(page.getByText("Back to Keys", { exact: false })).toBeVisible();
  });
});
