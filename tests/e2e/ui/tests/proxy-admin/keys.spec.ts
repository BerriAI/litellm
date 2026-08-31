import { test, expect, type Page as PlaywrightPage } from "@playwright/test";
import {
  ADMIN_STORAGE_PATH,
  E2E_DELETE_KEY_ALIAS,
  E2E_REGENERATE_KEY_ALIAS,
  E2E_UPDATE_LIMITS_KEY_ALIAS,
  E2E_INTERNAL_USER_KEY_ALIAS,
  E2E_TEAM_CRUD_ALIAS,
} from "../../constants";
import { Page } from "../../fixtures/pages";
import { navigateToPage, dismissFeedbackPopup } from "../../helpers/navigation";
import { captureRequestBody, readBack } from "../../helpers/roundTrip";

/**
 * Looks a key up by alias, undefined when none carries it. `return_full_object=true` is what makes
 * the row carry token / models / tpm_limit; without it the response is aliases only.
 */
async function findKeyByAlias(page: PlaywrightPage, alias: string): Promise<Record<string, any> | undefined> {
  const body = await readBack<{ keys: Record<string, any>[] }>(
    page,
    `/key/list?key_alias=${encodeURIComponent(alias)}&return_full_object=true&size=100`,
  );
  return body.keys.find((row) => row.key_alias === alias);
}

test.describe("Proxy Admin - Keys", () => {
  test.use({ storageState: ADMIN_STORAGE_PATH });

  test("Create a key in a team", async ({ page }) => {
    await navigateToPage(page, Page.ApiKeys);
    await dismissFeedbackPopup(page);

    // Click "+ Create New Key" button
    await page.getByRole("button", { name: /Create New Key/i }).click();

    // Wait for the key creation modal
    await expect(page.getByText("Key Ownership")).toBeVisible({ timeout: 10_000 });

    const keyName = `e2e-admin-key-${Date.now()}`;
    await page.getByLabel(/Key Name/).fill(keyName);

    // Select team
    const teamSelect = page.getByTestId("team-dropdown").getByRole("combobox");
    await teamSelect.click();
    await page.keyboard.type(E2E_TEAM_CRUD_ALIAS);
    await page.locator('[data-slot="combobox-content"]:visible').getByText(E2E_TEAM_CRUD_ALIAS).first().click();

    // Select models — the popup is portaled to the body, so scope options to the page.
    await page.getByRole("combobox", { name: "Select models" }).click();
    await page.getByRole("option", { name: "All Team Models", exact: true }).click();
    await page.keyboard.press("Escape");

    // Submit
    await page.getByRole("button", { name: "Create Key", exact: true }).click();

    // Success shows "Save your Key" in a second dialog
    await expect(page.getByText("Save your Key")).toBeVisible({ timeout: 10_000 });
    await page.keyboard.press("Escape");

    // Verify the new key appears in the table
    await expect(page.getByText(keyName)).toBeVisible({ timeout: 10_000 });

    // The row above renders from the create response the UI already holds, so it proves nothing.
    const persisted = await findKeyByAlias(page, keyName);
    expect(persisted, `key ${keyName} readable from /key/list`).toBeTruthy();
    expect(typeof persisted?.team_id, "created key is owned by a team, not orphaned").toBe("string");
  });

  test("Regenerate key", async ({ page }) => {
    await navigateToPage(page, Page.ApiKeys);
    await dismissFeedbackPopup(page);

    // Capture the old token first: a modal with a Copy button only proves the UI rendered.
    const before = await findKeyByAlias(page, E2E_REGENERATE_KEY_ALIAS);
    expect(before?.token, `seeded key ${E2E_REGENERATE_KEY_ALIAS} has a token`).toBeTruthy();

    // Key IDs are rendered as buttons in the table
    const keyRow = page.locator("tr", { hasText: E2E_REGENERATE_KEY_ALIAS });
    await expect(keyRow).toBeVisible({ timeout: 10_000 });
    await keyRow.locator("button").first().click();

    await expect(page.getByText("Back to Keys")).toBeVisible({ timeout: 10_000 });

    await page.getByRole("button", { name: "Regenerate Key" }).click();

    // Scope to the modal — the Regenerate button has an icon whose aria-label
    // ("sync") is concatenated into the button's accessible name, and the
    // "Regenerate Key" button is still in the DOM behind the modal.
    const modal = page.getByRole("dialog", { name: "Regenerate Virtual Key" });
    await modal.getByRole("button", { name: /Regenerate/ }).click();

    // Success view shows a Copy button in the footer (text varies between modal versions)
    await expect(modal.getByRole("button", { name: /Copy.*Key/ })).toBeVisible({ timeout: 20_000 });

    // The token must be replaced and the alias kept; orphaning it looks identical from the modal.
    await expect
      .poll(async () => (await findKeyByAlias(page, E2E_REGENERATE_KEY_ALIAS))?.token, {
        message: `token for ${E2E_REGENERATE_KEY_ALIAS} did not change after regenerate`,
        timeout: 15_000,
      })
      .not.toBe(before?.token);
  });

  test("Update key TPM and RPM limits", async ({ page }) => {
    await navigateToPage(page, Page.ApiKeys);
    await dismissFeedbackPopup(page);

    // Snapshot first, so the end assertions can tell an isolated edit from a collateral one.
    const before = await findKeyByAlias(page, E2E_UPDATE_LIMITS_KEY_ALIAS);
    expect(before, `seeded key ${E2E_UPDATE_LIMITS_KEY_ALIAS} exists`).toBeTruthy();

    const keyRow = page.locator("tr", { hasText: E2E_UPDATE_LIMITS_KEY_ALIAS });
    await expect(keyRow).toBeVisible({ timeout: 10_000 });
    await keyRow.locator("button").first().click();

    await expect(page.getByText("Back to Keys")).toBeVisible({ timeout: 10_000 });

    await page.getByRole("tab", { name: "Settings" }).click();
    await page.getByRole("button", { name: "Edit Settings" }).click();

    await page.getByRole("spinbutton", { name: "TPM Limit" }).fill("123");
    await page.getByRole("spinbutton", { name: "RPM Limit" }).fill("456");

    const update = await captureRequestBody(page, { method: "POST", urlIncludes: "/key/update" }, async () => {
      await page.getByRole("button", { name: "Save Changes" }).click();
    });

    // The form posts limits at the top level. Compare numerically: the spinbutton yields either type.
    expect(Number(update.tpm_limit), "TPM limit on the wire").toBe(123);
    expect(Number(update.rpm_limit), "RPM limit on the wire").toBe(456);

    await expect(page.getByRole("paragraph").filter({ hasText: "TPM: 123" })).toBeVisible({ timeout: 10_000 });
    await expect(page.getByRole("paragraph").filter({ hasText: "RPM: 456" })).toBeVisible({ timeout: 10_000 });

    // Read the key back; the rendering above comes from a response the UI already holds.
    const after = await findKeyByAlias(page, E2E_UPDATE_LIMITS_KEY_ALIAS);
    expect(after, "key still readable after update").toBeTruthy();
    expect(Number(after?.tpm_limit), "TPM limit persisted").toBe(123);
    expect(Number(after?.rpm_limit), "RPM limit persisted").toBe(456);

    // Not hypothetical: bumping a key's budget wiped its MCP toolset (PR #34452), toast said success.
    expect(after?.models, "editing limits left the key's models untouched").toEqual(before?.models);
    expect(after?.team_id, "editing limits left the key's team untouched").toEqual(before?.team_id);
  });

  test("Delete key", async ({ page }) => {
    await navigateToPage(page, Page.ApiKeys);
    await dismissFeedbackPopup(page);

    const keyRow = page.locator("tr", { hasText: E2E_DELETE_KEY_ALIAS });
    await expect(keyRow).toBeVisible({ timeout: 10_000 });
    await keyRow.locator("button").first().click();

    await expect(page.getByText("Back to Keys")).toBeVisible({ timeout: 10_000 });

    await page.getByRole("button", { name: "More key actions" }).click();
    await page.getByRole("menuitem", { name: "Delete Key" }).click();

    const modal = page.getByRole("dialog", { name: "Delete Key" });
    await expect(modal).toBeVisible({ timeout: 5_000 });
    await modal.locator("input").fill(E2E_DELETE_KEY_ALIAS);

    const deleteButton = modal.getByRole("button", { name: "Delete", exact: true });
    await expect(deleteButton).toBeEnabled();
    await deleteButton.click();

    await expect(page.getByText(/Key deleted/i).first()).toBeVisible({ timeout: 10_000 });

    // The key is gone when the management API stops returning it, not when the toast says so.
    await expect
      .poll(async () => await findKeyByAlias(page, E2E_DELETE_KEY_ALIAS), {
        message: `key ${E2E_DELETE_KEY_ALIAS} still readable from /key/list after delete`,
        timeout: 15_000,
      })
      .toBeUndefined();
  });

  test("See internal user keys in team", async ({ page }) => {
    await navigateToPage(page, Page.ApiKeys);
    await dismissFeedbackPopup(page);

    await expect(page.getByText(E2E_INTERNAL_USER_KEY_ALIAS)).toBeVisible({ timeout: 10_000 });
  });

  test("Create a key with All Proxy Models (no team)", async ({ page }) => {
    await navigateToPage(page, Page.ApiKeys);
    await dismissFeedbackPopup(page);

    await page.getByRole("button", { name: /Create New Key/i }).click();

    await expect(page.getByText("Key Ownership")).toBeVisible({ timeout: 10_000 });

    const keyName = `e2e-admin-allproxy-${Date.now()}`;
    await page.getByLabel(/Key Name/).fill(keyName);

    // No team selection — leave team dropdown empty so the key is owned by the admin user

    // Select models — open the multi-select and pick the all-models meta-option.
    // With no team selected the modal offers "All Proxy Models"; the team-scoped
    // "All Team Models" option only appears once a team is picked.
    await page.getByRole("combobox", { name: "Select models" }).click();
    await page.getByRole("option", { name: "All Proxy Models", exact: true }).click();
    await page.keyboard.press("Escape");

    await page.getByRole("button", { name: "Create Key", exact: true }).click();

    await expect(page.getByText("Save your Key")).toBeVisible({ timeout: 10_000 });
    await page.keyboard.press("Escape");

    await expect(page.getByText(keyName)).toBeVisible({ timeout: 10_000 });
  });

  test("Create a key with a specific proxy model (no team)", async ({ page }) => {
    await navigateToPage(page, Page.ApiKeys);
    await dismissFeedbackPopup(page);

    await page.getByRole("button", { name: /Create New Key/i }).click();

    await expect(page.getByText("Key Ownership")).toBeVisible({ timeout: 10_000 });

    const keyName = `e2e-admin-specific-${Date.now()}`;
    await page.getByLabel(/Key Name/).fill(keyName);

    // Open the model multi-select and pick a single specific model.
    const modelName = "fake-openai-gpt-4";
    await page.getByRole("combobox", { name: "Select models" }).click();
    await page.getByRole("option", { name: modelName, exact: true }).click();
    await page.keyboard.press("Escape");

    await page.getByRole("button", { name: "Create Key", exact: true }).click();

    await expect(page.getByText("Save your Key")).toBeVisible({ timeout: 10_000 });

    // Grab the new key from the success modal (rendered inside a <pre>) and
    // verify it can call /chat/completions for the model it was scoped to.
    // The mock LLM server (fixtures/mock_llm_server/server.py) replies with
    // a fixed "This is a mock response." body.
    const apiKey = (await page.getByRole("dialog", { name: "Save your Key" }).locator("pre").innerText()).trim();
    expect(apiKey).toMatch(/^sk-/);

    const response = await page.request.post("/chat/completions", {
      headers: { Authorization: `Bearer ${apiKey}` },
      data: {
        model: modelName,
        messages: [{ role: "user", content: "ping" }],
      },
    });
    expect(response.status()).toBe(200);
    const body = await response.json();
    expect(body.choices?.[0]?.message?.content).toBe("This is a mock response.");

    await page.keyboard.press("Escape");

    await expect(page.getByText(keyName)).toBeVisible({ timeout: 10_000 });
  });
});
