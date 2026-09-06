import { test, expect, type Page as PlaywrightPage } from "@playwright/test";
import { ADMIN_STORAGE_PATH, MOCK_PRESIDIO_URL } from "../../constants";
import { navigateToPage, dismissFeedbackPopup } from "../../helpers/navigation";
import { Page } from "../../fixtures/pages";
import { CHAT_MODEL_A, masterKey, rootPath, waitForSpendLogByPrompt } from "../../helpers/traffic";
import { openPlayground, selectModel, sendButton, onlyVisible } from "../../helpers/playground";

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

test.describe("Presidio PII guardrail, end to end from the dashboard", () => {
  test.use({ storageState: ADMIN_STORAGE_PATH });

  test("masks PII sent from the Playground and shows the run in Logs", async ({ page, request }) => {
    const guardrailName = `e2e-presidio-story-${Date.now()}`;
    const marker = `case-ref-${Math.random().toString(36).slice(2, 10)}`;
    const prompt = `${marker}. Email me at ${RAW_EMAIL} or call ${RAW_PHONE}.`;

    await createPresidioGuardrail(page, guardrailName);

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

    await expect
      .poll(
        async () => {
          await input.fill(prompt);
          await sendButton(page).click();
          const res = await request.get(`${rootPath()}/spend/logs`, {
            headers: { Authorization: `Bearer ${masterKey()}` },
          });
          if (!res.ok()) return false;
          const rows: { metadata?: { applied_guardrails?: string[] } }[] = await res.json();
          return (Array.isArray(rows) ? rows : []).some((row) =>
            (row.metadata?.applied_guardrails ?? []).includes(guardrailName),
          );
        },
        {
          message: `the playground never produced a request that ran ${guardrailName}`,
          timeout: 90_000,
          intervals: [5_000],
        },
      )
      .toBe(true);

    const requestId = await waitForSpendLogByPrompt(request, marker);

    const stored = await request.get(`${rootPath()}/spend/logs?request_id=${requestId}`, {
      headers: { Authorization: `Bearer ${masterKey()}` },
    });
    expect(stored.ok(), `spend log read failed: ${stored.status()}`).toBe(true);
    const storedBody = JSON.stringify(await stored.json());
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
    await expect(onlyVisible(drawer.getByText("Score: 1.00", { exact: true }))).toBeVisible({ timeout: 10_000 });
    await expect(onlyVisible(drawer.getByText("Score: 0.75", { exact: true }))).toBeVisible({ timeout: 10_000 });

    await expect(drawer.getByText(RAW_EMAIL)).toHaveCount(0);
    await expect(drawer.getByText(RAW_PHONE)).toHaveCount(0);

    await deleteGuardrail(page, guardrailName);
  });
});
