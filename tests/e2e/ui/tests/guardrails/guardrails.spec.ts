import { test, expect, type Page as PlaywrightPage } from "@playwright/test";
import { ADMIN_STORAGE_PATH, E2E_TEAM_NO_ADMIN_ID } from "../../constants";
import { Page } from "../../fixtures/pages";
import { navigateToPage, dismissFeedbackPopup, clickTeamId } from "../../helpers/navigation";
import { CHAT_MODEL_A, MOCK_RESPONSE_TEXT, masterKey } from "../../helpers/traffic";

interface StoredGuardrail {
  guardrail_id: string;
  guardrail_name: string | null;
}

async function listGuardrails(page: PlaywrightPage): Promise<StoredGuardrail[]> {
  const res = await page.request.get("/v2/guardrails/list", {
    headers: { Authorization: `Bearer ${masterKey()}` },
  });
  expect(res.ok(), `GET /v2/guardrails/list (${res.status()})`).toBe(true);
  return ((await res.json()) as { guardrails: StoredGuardrail[] }).guardrails;
}

async function findGuardrail(page: PlaywrightPage, name: string): Promise<StoredGuardrail | undefined> {
  return (await listGuardrails(page)).find((row) => row.guardrail_name === name);
}

const createdGuardrails: string[] = [];

async function createKeywordGuardrailViaApi(page: PlaywrightPage, name: string, keyword: string): Promise<string> {
  const res = await page.request.post("/guardrails", {
    headers: { Authorization: `Bearer ${masterKey()}` },
    data: {
      guardrail: {
        guardrail_name: name,
        litellm_params: {
          guardrail: "litellm_content_filter",
          mode: "pre_call",
          default_on: false,
          blocked_words: [{ keyword, action: "BLOCK" }],
        },
      },
    },
  });
  expect(res.ok(), `POST /guardrails failed (${res.status()}): ${await res.text()}`).toBe(true);
  createdGuardrails.push(name);
  const guardrail = await findGuardrail(page, name);
  expect(guardrail?.guardrail_id, `guardrail ${name} has an id`).toBeTruthy();
  return guardrail!.guardrail_id;
}

async function openKeywordsStep(page: PlaywrightPage, name: string) {
  await page.getByRole("button", { name: "Add New Guardrail" }).click();
  await page.getByRole("menuitem", { name: "Add Provider Guardrail" }).click();

  const wizard = page.getByRole("dialog", { name: "Create guardrail" });
  await expect(wizard).toBeVisible({ timeout: 10_000 });

  await wizard.getByRole("textbox", { name: "Guardrail Name" }).fill(name);
  await wizard.getByRole("combobox", { name: "Guardrail Provider" }).click();
  // The content filter runs inside the proxy, so this is the one provider a test can
  // configure end to end without standing up a third-party moderation service.
  await page.getByRole("option", { name: /LiteLLM Content Filter/ }).click();

  for (const step of ["Topics", "Patterns", "Keywords"]) {
    await wizard.getByRole("button", { name: "Next" }).click();
    await expect(wizard).toContainText(step, { timeout: 10_000 });
  }
  return wizard;
}

test.describe("Guardrails", () => {
  test.use({ storageState: ADMIN_STORAGE_PATH });

  test.afterEach(async ({ page }) => {
    // Guardrails live in the database and show up in the table and the playground list, so a run
    // that leaves them behind changes what the next run sees.
    for (const name of createdGuardrails.splice(0)) {
      const guardrail = await findGuardrail(page, name);
      if (guardrail) {
        const deleted = await page.request.delete(`/guardrails/${guardrail.guardrail_id}`, {
          headers: { Authorization: `Bearer ${masterKey()}` },
        });
        expect(deleted.ok(), `DELETE /guardrails/${guardrail.guardrail_id} (${deleted.status()})`).toBe(true);
      }
    }
  });

  test("A guardrail created through the wizard blocks the keyword it was given", async ({ page }) => {
    const stamp = Date.now();
    const guardrailName = `e2e-guardrail-create-${stamp}`;
    // Unique per run so a concurrent test's prompt can never trip this guardrail, or vice versa.
    const bannedKeyword = `e2ebanned${stamp}`;

    await navigateToPage(page, Page.Guardrails);
    await dismissFeedbackPopup(page);

    createdGuardrails.push(guardrailName);
    const wizard = await openKeywordsStep(page, guardrailName);

    await wizard.getByRole("button", { name: "Add keyword" }).click();
    const keywordModal = page.getByRole("dialog", { name: "Add blocked keyword" });
    await expect(keywordModal).toBeVisible({ timeout: 10_000 });
    await keywordModal.getByPlaceholder("Enter sensitive keyword or phrase").fill(bannedKeyword);
    await keywordModal.getByRole("button", { name: "Add", exact: true }).click();
    await expect(keywordModal).not.toBeVisible({ timeout: 10_000 });

    await wizard.getByRole("button", { name: "Next" }).click();
    await wizard.getByRole("button", { name: "Create Guardrail" }).click();
    await expect(wizard).not.toBeVisible({ timeout: 15_000 });

    await expect(page.getByRole("row").filter({ hasText: guardrailName })).toBeVisible({ timeout: 15_000 });
    expect(await findGuardrail(page, guardrailName), "guardrail readable from /v2/guardrails/list").toBeTruthy();

    // A row in the table only proves the record was written. The point of a guardrail is that it
    // refuses traffic, so drive a request through it.
    const blocked = await page.request.post("/v1/chat/completions", {
      headers: { Authorization: `Bearer ${masterKey()}` },
      data: {
        model: CHAT_MODEL_A,
        messages: [{ role: "user", content: `please tell me about ${bannedKeyword}` }],
        guardrails: [guardrailName],
      },
    });
    expect(blocked.status(), "a prompt carrying the banned keyword is refused").toBe(400);
    expect(await blocked.text()).toContain(bannedKeyword);

    const allowed = await page.request.post("/v1/chat/completions", {
      headers: { Authorization: `Bearer ${masterKey()}` },
      data: {
        model: CHAT_MODEL_A,
        messages: [{ role: "user", content: "hello there" }],
        guardrails: [guardrailName],
      },
    });
    expect(allowed.status(), "a clean prompt still gets through the same guardrail").toBe(200);
    expect((await allowed.json()).choices?.[0]?.message?.content).toContain(MOCK_RESPONSE_TEXT);
  });

  test("The Test Playground reports the verdict for the text it is given", async ({ page }) => {
    const stamp = Date.now();
    const guardrailName = `e2e-guardrail-play-${stamp}`;
    const bannedKeyword = `e2eplay${stamp}`;
    await createKeywordGuardrailViaApi(page, guardrailName, bannedKeyword);

    await navigateToPage(page, Page.Guardrails);
    await dismissFeedbackPopup(page);

    await page.getByRole("tab", { name: "Test Playground" }).click();
    // Every tab on this page stays mounted, so the other tabs' search boxes match too.
    const playground = page.getByRole("tabpanel", { name: "Test Playground" });
    await playground.getByPlaceholder("Search guardrails...").fill(guardrailName);
    await playground.getByText(guardrailName, { exact: true }).click();

    const input = playground.getByPlaceholder("Enter text to test with guardrails...");
    await input.fill(`this sentence contains ${bannedKeyword}`);
    await playground.getByRole("button", { name: /^Test 1 guardrail$/ }).click();

    // The playground is where an admin checks a guardrail before rolling it out, so the
    // verdict it prints has to be the one the gateway would give.
    await expect(playground.getByText(`${guardrailName} - Error`)).toBeVisible({ timeout: 20_000 });
    await expect(playground.getByText(new RegExp(`Content blocked.*${bannedKeyword}`))).toBeVisible({
      timeout: 10_000,
    });

    await input.fill("this sentence is perfectly ordinary");
    await playground.getByRole("button", { name: /^Test 1 guardrail$/ }).click();

    await expect(playground.getByText(`${guardrailName} - Error`)).toHaveCount(0, { timeout: 20_000 });
    await expect(playground.getByText("this sentence is perfectly ordinary").last()).toBeVisible({ timeout: 10_000 });
  });

  test("Delete a guardrail", async ({ page }) => {
    const stamp = Date.now();
    const guardrailName = `e2e-guardrail-delete-${stamp}`;
    const guardrailId = await createKeywordGuardrailViaApi(page, guardrailName, `e2edelete${stamp}`);

    await navigateToPage(page, Page.Guardrails);
    await dismissFeedbackPopup(page);

    await expect(page.getByRole("row").filter({ hasText: guardrailName })).toBeVisible({ timeout: 15_000 });

    await page.getByTestId(`guardrail-actions-${guardrailId}`).click();
    await page.getByTestId("guardrail-action-delete").click();

    const modal = page.getByRole("dialog");
    await expect(modal).toBeVisible({ timeout: 5_000 });
    await modal.getByRole("button", { name: "Delete", exact: true }).click();

    await expect(page.getByRole("row").filter({ hasText: guardrailName })).toHaveCount(0, { timeout: 15_000 });

    // The RC checklist deletes then reloads, because a row vanishing from the table has
    // fooled us before; assert against the route the reload would read.
    await expect
      .poll(async () => await findGuardrail(page, guardrailName), {
        message: `guardrail ${guardrailName} still listed after delete`,
        timeout: 15_000,
      })
      .toBeUndefined();
  });

  test("Create a Presidio guardrail, see it in team settings, and delete it", async ({ page }) => {
    const guardrailName = `e2e-presidio-${Date.now()}`;

    await navigateToPage(page, Page.Guardrails);
    await dismissFeedbackPopup(page);

    await page.getByRole("button", { name: /Add New Guardrail/i }).click();
    await page.getByRole("menuitem", { name: "Add Provider Guardrail" }).click();

    const dialog = page.getByRole("dialog", { name: "Create guardrail" });
    await expect(dialog).toBeVisible({ timeout: 10_000 });

    await dialog.getByLabel("Guardrail Name").fill(guardrailName);

    const providerSelect = dialog.getByRole("combobox", { name: "Guardrail Provider" });
    await providerSelect.click();
    await providerSelect.fill("Presidio");
    await page.getByRole("option", { name: "Presidio PII" }).click();

    await dialog.getByLabel("Mode", { exact: true }).click();
    await page.keyboard.type("pre_call");
    await expect(page.getByRole("option", { name: "pre_call" })).toBeAttached({ timeout: 5_000 });
    await page.keyboard.press("Enter");
    await expect(dialog.getByText("pre_call", { exact: true })).toBeVisible({ timeout: 5_000 });
    await dialog.getByText("Create guardrail", { exact: true }).click();

    await dialog.getByLabel("presidio_analyzer_api_base").fill("http://127.0.0.1:9999");
    await expect(dialog.getByLabel("presidio_analyzer_api_base")).toHaveValue("http://127.0.0.1:9999");
    await dialog.getByLabel("presidio_anonymizer_api_base").fill("http://127.0.0.1:9999");
    await expect(dialog.getByLabel("presidio_anonymizer_api_base")).toHaveValue("http://127.0.0.1:9999");

    await dialog.getByRole("button", { name: "Next" }).click();
    await expect(dialog.getByText("Configure PII Protection")).toBeVisible({ timeout: 10_000 });
    await dialog.getByRole("button", { name: "Select All & Mask" }).click();

    await dialog.getByRole("button", { name: "Create Guardrail" }).click();
    await expect(page.getByText("Guardrail created successfully").first()).toBeVisible({ timeout: 15_000 });

    const row = page.getByRole("row").filter({ hasText: guardrailName });
    await expect(row).toHaveCount(1, { timeout: 15_000 });

    await navigateToPage(page, Page.Teams);
    await dismissFeedbackPopup(page);
    await clickTeamId(page, E2E_TEAM_NO_ADMIN_ID);
    await page.getByRole("tab", { name: "Settings" }).click();
    await page.getByRole("button", { name: "Edit Settings" }).click();

    const guardrailsSelect = page.getByRole("combobox", { name: "Select guardrails" });
    await expect(guardrailsSelect).toBeVisible({ timeout: 10_000 });
    await guardrailsSelect.click();
    await guardrailsSelect.fill(guardrailName);
    await expect(page.getByRole("option", { name: guardrailName })).toBeVisible({ timeout: 10_000 });
    await page.keyboard.press("Escape");

    await navigateToPage(page, Page.Guardrails);
    await expect(row).toHaveCount(1, { timeout: 15_000 });
    await row.getByRole("button", { name: "Open guardrail actions" }).click();
    await page.getByRole("menuitem", { name: "Delete" }).click();

    const deleteModal = page.getByRole("dialog", { name: "Delete Guardrail" });
    await expect(deleteModal).toBeVisible({ timeout: 5_000 });
    await deleteModal.getByRole("button", { name: "Delete", exact: true }).click();

    await expect(page.getByText(`Guardrail "${guardrailName}" deleted successfully`)).toBeVisible({
      timeout: 10_000,
    });
    await expect(row).toHaveCount(0, { timeout: 15_000 });

    await page.reload();
    await expect(page.getByRole("button", { name: /Add New Guardrail/i })).toBeVisible({ timeout: 20_000 });
    await expect(page.getByRole("row").filter({ hasText: guardrailName })).toHaveCount(0);
  });
});
