import { test as base, expect, type APIRequestContext, type Page as PlaywrightPage } from "@playwright/test";
import { ADMIN_STORAGE_PATH, MOCK_PRESIDIO_URL } from "../../constants";
import { navigateToPage, dismissFeedbackPopup } from "../../helpers/navigation";
import { Page } from "../../fixtures/pages";
import { CHAT_MODEL_A, MOCK_RESPONSE_TEXT, masterKey, rootPath, waitForSpendLog } from "../../helpers/traffic";
import { openPlayground, selectModel, sendButton, onlyVisible } from "../../helpers/playground";

const test = base.extend<{ guardrailName: string }>({
  guardrailName: async ({ request }, use) => {
    const name = `e2e-presidio-story-${Date.now()}`;
    try {
      await use(name);
    } finally {
      const res = await request.get(`${rootPath()}/v2/guardrails/list`, {
        headers: { Authorization: `Bearer ${masterKey()}` },
      });
      expect(res.ok(), `guardrail cleanup lookup failed (${res.status()})`).toBe(true);
      const body: { guardrails: { guardrail_id: string; guardrail_name: string }[] } = await res.json();
      for (const guardrail of body.guardrails.filter((row) => row.guardrail_name === name)) {
        const deleted = await request.delete(`${rootPath()}/guardrails/${guardrail.guardrail_id}`, {
          headers: { Authorization: `Bearer ${masterKey()}` },
        });
        expect(deleted.ok(), `guardrail cleanup failed (${deleted.status()})`).toBe(true);
      }
    }
  },
});

interface SpendLog {
  request_id: string;
  proxy_server_request?: { messages?: { role: string; content: string }[] };
  metadata?: {
    guardrail_information?: {
      guardrail_name: string;
      guardrail_mode: string;
      guardrail_status: string;
      guardrail_provider: string;
    }[];
  };
}

async function readSpendLog(request: APIRequestContext, requestId: string, timeoutMs = 60_000): Promise<SpendLog> {
  await waitForSpendLog(request, requestId, timeoutMs);
  const res = await request.get(`${rootPath()}/spend/logs?request_id=${encodeURIComponent(requestId)}`, {
    headers: { Authorization: `Bearer ${masterKey()}` },
  });
  expect(res.ok(), `spend log read failed (${res.status()})`).toBe(true);
  const rows: SpendLog[] = await res.json();
  expect(rows).toHaveLength(1);
  expect(rows[0].request_id).toBe(requestId);
  return rows[0];
}

async function waitForGuardrailExecution(request: APIRequestContext, guardrailName: string): Promise<void> {
  const deadline = Date.now() + 90_000;
  while (Date.now() < deadline) {
    const res = await request.post(`${rootPath()}/v1/chat/completions`, {
      headers: { Authorization: `Bearer ${masterKey()}` },
      data: {
        model: CHAT_MODEL_A,
        messages: [{ role: "user", content: "Presidio readiness check." }],
        guardrails: [guardrailName],
        stream: false,
      },
    });
    expect(res.ok(), `Presidio readiness request failed (${res.status()}): ${await res.text()}`).toBe(true);
    const body: { id: string } = await res.json();
    expect(body.id).toBeTruthy();
    const row = await readSpendLog(request, body.id, Math.max(1, deadline - Date.now()));
    const run = row.metadata?.guardrail_information?.find(
      (record) => record.guardrail_name === guardrailName && record.guardrail_mode === "pre_call",
    );
    if (run) {
      expect(run.guardrail_provider).toBe("presidio");
      expect(run.guardrail_status).toBe("success");
      return;
    }
    await new Promise((resolve) => setTimeout(resolve, 5_000));
  }
  throw new Error(`Presidio guardrail ${guardrailName} never executed on the serving proxy`);
}

const RAW_EMAIL = "jane.doe@example.com";
const RAW_PHONE = "555-867-5309";

const visibleTestId = (page: PlaywrightPage, id: string) => page.getByTestId(id).filter({ visible: true });

const requestLogsRows = (page: PlaywrightPage) =>
  page.locator("table").filter({ visible: true }).first().locator("tbody tr");

async function createPresidioGuardrail(page: PlaywrightPage, guardrailName: string): Promise<void> {
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

  await dialog.getByLabel("presidio_analyzer_api_base").fill(MOCK_PRESIDIO_URL);
  await dialog.getByLabel("presidio_anonymizer_api_base").fill(MOCK_PRESIDIO_URL);

  await dialog.getByRole("button", { name: "Next" }).click();
  await expect(dialog.getByText("Configure PII Protection")).toBeVisible({ timeout: 10_000 });
  await dialog.getByRole("button", { name: "Select All & Mask" }).click();

  await dialog.getByRole("button", { name: "Create Guardrail" }).click();
  await expect(page.getByText("Guardrail created successfully").first()).toBeVisible({ timeout: 15_000 });
  await expect(page.getByRole("row").filter({ hasText: guardrailName })).toHaveCount(1, { timeout: 15_000 });
}

async function deleteGuardrail(page: PlaywrightPage, guardrailName: string): Promise<void> {
  await navigateToPage(page, Page.Guardrails);
  await dismissFeedbackPopup(page);
  const row = page.getByRole("row").filter({ hasText: guardrailName });
  await expect(row).toHaveCount(1, { timeout: 15_000 });
  await row.getByRole("button", { name: "Open guardrail actions" }).click();
  await page.getByRole("menuitem", { name: "Delete" }).click();
  const deleteModal = page.getByRole("dialog", { name: "Delete Guardrail" });
  await expect(deleteModal).toBeVisible({ timeout: 5_000 });
  await deleteModal.getByRole("button", { name: "Delete", exact: true }).click();
  await expect(page.getByText(`Guardrail "${guardrailName}" deleted successfully`)).toBeVisible({ timeout: 10_000 });
}

test.use({ trace: "retain-on-failure" });

test.describe("Presidio PII guardrail, end to end from the dashboard", () => {
  test.use({ storageState: ADMIN_STORAGE_PATH });

  test("masks PII sent from the Playground and shows the run in Logs", async ({ page, request, guardrailName }) => {
    const marker = `case-ref-${Math.random().toString(36).slice(2, 10)}`;
    const prompt = `${marker}. Email me at ${RAW_EMAIL} or call ${RAW_PHONE}.`;

    await createPresidioGuardrail(page, guardrailName);
    await waitForGuardrailExecution(request, guardrailName);

    await openPlayground(page);
    await selectModel(page, CHAT_MODEL_A);

    const guardrailSelect = onlyVisible(page.getByPlaceholder("Select guardrails"));
    await expect(guardrailSelect).toBeVisible({ timeout: 20_000 });
    await guardrailSelect.click();
    await guardrailSelect.fill(guardrailName);
    await onlyVisible(page.getByRole("option", { name: guardrailName, exact: true })).click({ timeout: 20_000 });
    await page.keyboard.press("Escape");

    const input = onlyVisible(page.getByPlaceholder("Type your message", { exact: false }));
    await expect(input).toBeVisible({ timeout: 15_000 });

    await input.fill(prompt);
    const [response] = await Promise.all([
      page.waitForResponse(
        (res) => res.request().method() === "POST" && /\/chat\/completions$/.test(new URL(res.url()).pathname),
      ),
      sendButton(page).click(),
    ]);
    const responseBody = await response.text();
    expect(response.ok(), `Playground request failed (${response.status()}): ${responseBody}`).toBe(true);
    expect(response.request().postDataJSON().messages).toEqual([{ role: "user", content: prompt }]);
    expect(responseBody).toContain(MOCK_RESPONSE_TEXT);
    const chunks: { id?: string; error?: unknown }[] = responseBody
      .split("\n")
      .filter((line) => line.startsWith("data: ") && line.trim() !== "data: [DONE]")
      .map((line) => JSON.parse(line.slice(6)));
    expect(chunks.some((chunk) => chunk.error), `Playground stream failed: ${responseBody}`).toBe(false);
    const requestId = chunks.find((chunk) => chunk.id)?.id ?? "";
    expect(requestId, "Playground stream has no completion ID").toBeTruthy();
    const stored = await readSpendLog(request, requestId);
    expect(stored.metadata?.guardrail_information).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          guardrail_name: guardrailName,
          guardrail_mode: "pre_call",
          guardrail_provider: "presidio",
          guardrail_status: "success",
        }),
      ]),
    );
    expect(stored.proxy_server_request?.messages).toEqual([
      { role: "user", content: `${marker}. Email me at <EMAIL_ADDRESS> or call <PHONE_NUMBER>.` },
    ]);
    const storedBody = JSON.stringify(stored);
    expect(storedBody, "the raw email reached the spend log, so the prompt was stored unmasked").not.toContain(
      RAW_EMAIL,
    );
    expect(storedBody, "the raw phone number reached the spend log, so the prompt was stored unmasked").not.toContain(
      RAW_PHONE,
    );
    expect(storedBody, "the stored prompt carries no EMAIL_ADDRESS placeholder, so nothing was masked").toContain(
      "<EMAIL_ADDRESS>",
    );
    expect(storedBody, "the stored prompt carries no PHONE_NUMBER placeholder, so nothing was masked").toContain(
      "<PHONE_NUMBER>",
    );

    await navigateToPage(page, Page.Logs);
    await dismissFeedbackPopup(page);
    const search = visibleTestId(page, "datatable-search");
    await expect(search).toBeVisible({ timeout: 20_000 });
    await search.fill(requestId);
    const row = requestLogsRows(page).filter({ hasText: requestId });
    await expect(row, `no logs row for request ${requestId}`).toHaveCount(1, { timeout: 30_000 });
    await row.click();

    const drawer = page.getByRole("dialog").first();
    await expect(onlyVisible(drawer.getByText("Guardrails & Policy Compliance"))).toBeVisible({ timeout: 20_000 });
    await expect(onlyVisible(drawer.getByText(`Pre-call guardrail: ${guardrailName}`))).toBeVisible({ timeout: 20_000 });
    const maskedPrompt = drawer.getByText(`${marker}. Email me at <EMAIL_ADDRESS> or call <PHONE_NUMBER>.`);
    await expect(onlyVisible(maskedPrompt)).toBeVisible({ timeout: 20_000 });

    await onlyVisible(drawer.getByText("2 matched")).click();
    await expect(onlyVisible(drawer.getByText("Detected Entities (2)"))).toBeVisible({ timeout: 10_000 });
    await expect(onlyVisible(drawer.getByText("EMAIL_ADDRESS", { exact: true }))).toBeVisible({ timeout: 10_000 });
    await expect(onlyVisible(drawer.getByText("PHONE_NUMBER", { exact: true }))).toBeVisible({ timeout: 10_000 });
    await expect(drawer.getByText(/^Score: \d\.\d{2}$/).filter({ visible: true })).toHaveCount(2, {
      timeout: 10_000,
    });

    await expect(drawer.getByText(RAW_EMAIL)).toHaveCount(0);
    await expect(drawer.getByText(RAW_PHONE)).toHaveCount(0);

    await deleteGuardrail(page, guardrailName);
  });
});
