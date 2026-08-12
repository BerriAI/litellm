import { test, expect } from "@playwright/test";
import { ADMIN_STORAGE_PATH } from "../../constants";
import { navigateToPage } from "../../helpers/navigation";
import { Page } from "../../fixtures/pages";
import { Role, users } from "../../fixtures/users";
import { MOCK_RESPONSE_TEXT } from "../../helpers/traffic";
import { openPlayground, selectModel, sendMessage } from "../../helpers/playground";
// Type-only import of the OpenAPI-generated backend schema, erased at runtime by
// esbuild. It types the round-trips below so mistakes surface in the editor; the live
// test against the real proxy is what actually enforces the contract.
import type { components } from "../../../../../ui/litellm-dashboard/src/lib/http/schema";

// These tests mutate the proxy's shared router_settings, and the Loadbalancing save
// echoes the whole settings object, so they must not run concurrently.
test.describe.configure({ mode: "serial" });

const PRIMARY = "fake-openai-gpt-4";
const FALLBACK = "fake-anthropic-claude";

/**
 * Wipe any fallbacks for the primary model so the test is idempotent across
 * retries and local reruns (the proxy persists router_settings to the DB).
 */
async function clearFallbackForPrimary(request: import("@playwright/test").APIRequestContext) {
  const masterKey = users[Role.ProxyAdmin].password;
  const auth = { Authorization: `Bearer ${masterKey}` };

  const current = await request.get("/get/config/callbacks", { headers: auth });
  if (!current.ok()) return;
  const body = await current.json();
  const router = body?.router_settings ?? {};
  const existing: Array<Record<string, string[]>> = Array.isArray(router.fallbacks) ? router.fallbacks : [];
  const next = existing.filter((entry) => !(entry && PRIMARY in entry));
  if (next.length === existing.length) return;

  await request.post("/config/update", {
    headers: auth,
    data: { router_settings: { ...router, fallbacks: next } },
  });
}

test.describe("Router Settings - Fallbacks", () => {
  test.use({ storageState: ADMIN_STORAGE_PATH });

  test.beforeEach(async ({ request }) => {
    await clearFallbackForPrimary(request);
  });

  test.afterEach(async ({ request }) => {
    await clearFallbackForPrimary(request);
  });

  test("Add a fallback and verify it appears in the table", async ({ page }) => {
    await navigateToPage(page, Page.RouterSettings);

    // Four tabs: Loadbalancing / Routing Groups / Fallbacks / General — click Fallbacks
    await page.getByRole("tab", { name: "Fallbacks" }).click();

    // The model options come from /model_group/info, which AddFallbacks
    // fires only after the modal mounts. Wait for that response so the
    // dropdown is populated before we try to pick from it — without this
    // the test races on CI (local SLOWMO masks the gap).
    const modelsLoaded = page.waitForResponse(
      (res) => res.url().includes("/model_group/info") && res.status() === 200,
      { timeout: 15_000 },
    );
    await page.getByRole("button", { name: /Add Fallbacks/i }).click();
    await modelsLoaded;

    const modal = page.locator(".ant-modal:visible");
    await expect(modal).toBeVisible({ timeout: 5_000 });

    // FallbackGroupConfig.tsx renders both selects with `showSearch`. The
    // most stable interaction is: click to open + focus, type the model name to
    // narrow the listbox to a single highlighted option, then press Enter.
    // Verify each selection landed by watching the dialog's own state transition
    // (the tab title updates to the picked primary; the fallback chain list
    // populates) rather than by asserting on the dropdown popup, which sits in
    // a custom getPopupContainer and is awkward to scope reliably.
    const primarySelect = modal.locator(".ant-select").filter({ hasText: "Select primary model" });
    await primarySelect.click();
    await page.keyboard.type(PRIMARY);
    await page.keyboard.press("Enter");
    await expect(modal.getByRole("tab", { name: PRIMARY })).toBeVisible({
      timeout: 10_000,
    });

    const fallbackSelect = modal.locator(".ant-select").filter({ hasText: "Select fallback models" });
    await fallbackSelect.click();
    await page.keyboard.type(FALLBACK);
    await page.keyboard.press("Enter");
    await page.keyboard.press("Escape");
    // The Fallback Chain helper text reads "(N/10 used)"; once it ticks to 1 the
    // selection has been recorded.
    await expect(modal.getByText("(1/10 used)")).toBeVisible({
      timeout: 10_000,
    });

    // Save
    await modal.getByRole("button", { name: /Save All Configurations/i }).click();

    // Success toast
    await expect(page.getByText(/fallback configuration\(s\) added successfully/i).first()).toBeVisible({
      timeout: 10_000,
    });

    // Modal closes, and a single row contains BOTH the primary and the fallback
    // model — stronger than asserting each name appears somewhere in tbody,
    // which could be satisfied by leftover rows from prior runs.
    await expect(modal).not.toBeVisible({ timeout: 5_000 });

    const newRow = page.locator("table tbody tr").filter({ hasText: PRIMARY }).filter({ hasText: FALLBACK });
    await expect(newRow).toHaveCount(1, { timeout: 10_000 });
  });
});

type ConfigYAML = components["schemas"]["ConfigYAML"];
type RouterSettingsResponse = components["schemas"]["RouterSettingsResponse"];

const ADMIN_AUTH = {
  Authorization: `Bearer ${users[Role.ProxyAdmin].password}`,
};

/**
 * Apply a router_settings patch through the typed /config/update contract. The
 * server merges it over existing settings (request wins), so only the passed keys
 * change. Fails loudly if the write is rejected instead of leaving a silent bad seed.
 */
async function patchRouterSettings(
  request: import("@playwright/test").APIRequestContext,
  patch: Partial<NonNullable<ConfigYAML["router_settings"]>>,
) {
  const res = await request.post(`/config/update`, {
    headers: ADMIN_AUTH,
    data: { router_settings: patch },
  });
  expect(res.ok(), `seed /config/update failed: ${res.status()} ${await res.text()}`).toBeTruthy();
}

test.describe("Router Settings - Loadbalancing", () => {
  test.use({ storageState: ADMIN_STORAGE_PATH });

  // Pin num_retries and an empty routing_groups so the assertions are deterministic.
  // Empty already reproduces LIT-4057: the old tab serialized [] to the string "[]"
  // and the save 422'd.
  test.beforeEach(async ({ request }) => {
    await patchRouterSettings(request, { num_retries: 3, routing_groups: [] });
  });

  test.afterEach(async ({ request }) => {
    await patchRouterSettings(request, { num_retries: 3 });
  });

  test("saves the Loadbalancing tab without a 422 when routing_groups is present, and persists", async ({
    page,
    request,
  }) => {
    await navigateToPage(page, Page.RouterSettings);
    await page.getByRole("tab", { name: "Loadbalancing" }).click();

    const numRetries = page.locator('input[name="num_retries"]');
    await expect(numRetries).toHaveValue("3", { timeout: 15_000 });
    // routing_groups belongs to its own tab and must not leak into this form.
    await expect(page.locator('input[name="routing_groups"]')).toHaveCount(0);

    await numRetries.fill("5");

    // LIT-4057: the tab used to serialize routing_groups as the string "[]",
    // which the backend rejects with 422 while the UI still claimed success.
    // Assert the save actually succeeds at the network level.
    const saveResponse = page.waitForResponse(
      (res) => res.url().includes("/config/update") && res.request().method() === "POST",
      { timeout: 15_000 },
    );
    await page.getByRole("button", { name: /save changes/i }).click();
    expect((await saveResponse).status()).toBe(200);

    await expect(page.getByText(/router settings updated successfully/i).first()).toBeVisible({ timeout: 10_000 });

    // The ticket's core symptom was that a refresh showed the old value.
    await navigateToPage(page, Page.RouterSettings);
    await page.getByRole("tab", { name: "Loadbalancing" }).click();
    await expect(page.locator('input[name="num_retries"]')).toHaveValue("5", {
      timeout: 15_000,
    });

    // The typed backend read agrees the change persisted.
    await expect
      .poll(
        async () => {
          const res = await request.get(`/router/settings`, {
            headers: ADMIN_AUTH,
          });
          const data = (await res.json()) as RouterSettingsResponse;
          return data.current_values?.num_retries;
        },
        { timeout: 10_000 },
      )
      .toBe(5);
  });
});

/**
 * The test above proves the UI can record a fallback; this proves the fallback is honoured. The
 * primary is created here because every fixture model is mock-backed and cannot fail on demand.
 */
test.describe("Router Settings - Fallbacks serve the request", () => {
  test.use({ storageState: ADMIN_STORAGE_PATH });

  const BROKEN_PRIMARY = "e2e-broken-primary";
  let brokenModelId: string | null = null;

  /** Drop only this test's fallback entry, leaving any others untouched. */
  async function clearBrokenFallback(request: import("@playwright/test").APIRequestContext) {
    const current = await request.get("/get/config/callbacks", {
      headers: ADMIN_AUTH,
    });
    if (!current.ok()) return;
    const router = (await current.json())?.router_settings ?? {};
    const existing: Array<Record<string, string[]>> = Array.isArray(router.fallbacks) ? router.fallbacks : [];
    await patchRouterSettings(request, {
      fallbacks: existing.filter((entry) => !(entry && BROKEN_PRIMARY in entry)),
    } as Partial<NonNullable<ConfigYAML["router_settings"]>>);
  }

  test.beforeEach(async ({ request }) => {
    await clearBrokenFallback(request);

    // Port 9 is the discard service: nothing listens, so the connection is
    // refused immediately rather than hanging until a timeout.
    const res = await request.post("/model/new", {
      headers: ADMIN_AUTH,
      data: {
        model_name: BROKEN_PRIMARY,
        litellm_params: {
          model: "openai/broken",
          api_base: "http://127.0.0.1:9/v1",
          api_key: "fake",
          timeout: 5,
        },
      },
    });
    expect(res.ok(), `creating the broken primary failed: ${res.status()} ${await res.text()}`).toBeTruthy();
    brokenModelId = (await res.json())?.model_id ?? null;
  });

  test.afterEach(async ({ request }) => {
    await clearBrokenFallback(request);
    if (brokenModelId) {
      await request.post("/model/delete", {
        headers: ADMIN_AUTH,
        data: { id: brokenModelId },
      });
      brokenModelId = null;
    }
  });

  test("a request to an unreachable model is answered by its fallback", async ({ page, request }) => {
    const chat = async () =>
      request.post("/v1/chat/completions", {
        headers: { ...ADMIN_AUTH, "Content-Type": "application/json" },
        data: {
          model: BROKEN_PRIMARY,
          messages: [{ role: "user", content: "fallback probe" }],
        },
      });

    // The control: it proves the reply below could only have come from the fallback.
    expect((await chat()).status(), "broken primary unexpectedly succeeded on its own").toBeGreaterThanOrEqual(400);

    await patchRouterSettings(request, {
      fallbacks: [{ [BROKEN_PRIMARY]: [PRIMARY] }],
    } as Partial<NonNullable<ConfigYAML["router_settings"]>>);

    // Same call now succeeds, served by the fallback model.
    await expect
      .poll(async () => (await chat()).status(), {
        timeout: 30_000,
        message: "fallback never took effect",
      })
      .toBe(200);

    // And the playground renders a reply for a model whose own upstream is down.
    await openPlayground(page);
    await selectModel(page, BROKEN_PRIMARY);
    await sendMessage(page, "fallback probe from the playground");
    await expect(page.getByText(MOCK_RESPONSE_TEXT, { exact: false }).first()).toBeVisible({ timeout: 60_000 });
  });
});
