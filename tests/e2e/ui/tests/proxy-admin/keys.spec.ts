import { test, expect, type Page as PlaywrightPage } from "@playwright/test";
import {
  ADMIN_STORAGE_PATH,
  E2E_REGENERATE_KEY_ALIAS,
  E2E_UPDATE_LIMITS_KEY_ALIAS,
  E2E_INTERNAL_USER_KEY_ALIAS,
  E2E_TEAM_CRUD_ALIAS,
  E2E_TEAM_CRUD_ID,
  E2E_TEAM_ADMIN_USER_ID,
  E2E_INTERNAL_USER_ID,
  E2E_INTERNAL_USER_EMAIL,
} from "../../constants";
import { Page } from "../../fixtures/pages";
import { navigateToPage, dismissFeedbackPopup } from "../../helpers/navigation";
import { captureRequestBody, readBack, runWithCleanup } from "../../helpers/roundTrip";
import { masterKey } from "../../helpers/traffic";
import { proxyIsPremium } from "../../helpers/premium";

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

/** A key this test owns, so deleting it costs the suite nothing on a retry or a second run. */
async function createDeletableKey(page: PlaywrightPage): Promise<string> {
  const alias = `e2e-delete-key-${Date.now()}`;
  const res = await page.request.post("/key/generate", {
    headers: { Authorization: `Bearer ${masterKey()}` },
    data: { key_alias: alias, team_id: E2E_TEAM_CRUD_ID },
  });
  expect(res.ok(), `POST /key/generate failed (${res.status()}): ${await res.text()}`).toBe(true);
  return alias;
}

async function readKeyStates(page: PlaywrightPage, aliases: string[]) {
  const keys = await Promise.all(aliases.map((alias) => findKeyByAlias(page, alias)));
  return keys.map(
    (key) =>
      key && {
        token: key.token,
        key_alias: key.key_alias,
        team_id: key.team_id,
        user_id: key.user_id,
        max_budget: key.max_budget,
        budget_duration: key.budget_duration,
        models: key.models,
        metadata: key.metadata,
        tpm_limit: key.tpm_limit,
        rpm_limit: key.rpm_limit,
        expires: key.expires,
      },
  );
}

async function withBulkKeys(page: PlaywrightPage, action: (aliases: string[]) => Promise<void>) {
  const prefix = `e2e-bulk-${test.info().workerIndex}-${Date.now()}`;
  const aliases = [`${prefix}-first`, `${prefix}-second`, `${prefix}-keep`];
  const headers = { Authorization: `Bearer ${masterKey()}` };
  await runWithCleanup(
    async () => {
      for (const key_alias of aliases) {
        const response = await page.request.post("/key/generate", {
          headers,
          data: {
            key_alias,
            user_id: E2E_TEAM_ADMIN_USER_ID,
            max_budget: 100,
            budget_duration: "30d",
            tpm_limit: 123,
            rpm_limit: 45,
            metadata: { bulk_edit_test: prefix },
          },
        });
        expect(response.ok(), `create disposable key ${key_alias}`).toBe(true);
      }
      const initial = await readKeyStates(page, aliases);
      expect(initial.map((key) => key?.token)).toEqual(aliases.map(() => expect.any(String)));
      await navigateToPage(page, Page.ApiKeys);
      await page.getByTestId("datatable-search").fill(prefix);
      await expect(page.getByRole("row").filter({ hasText: aliases[0] })).toBeVisible();
      await expect(page.getByRole("checkbox")).toHaveCount(0);
      await page.getByRole("button", { name: "Select keys", exact: true }).click();
      await expect(page.getByRole("button", { name: "Actions", exact: true })).toHaveCount(0);
      await action(aliases);
    },
    async () => {
      const remaining = await readKeyStates(page, aliases);
      const tokens = remaining.flatMap((key) => (typeof key?.token === "string" ? [key.token] : []));
      if (tokens.length === 0) return true;
      const response = await page.request.post("/key/delete", { headers, data: { keys: tokens } });
      return response.ok();
    },
  );
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
    await page.getByRole("option", { name: E2E_TEAM_CRUD_ALIAS }).first().click();

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
    // The Regenerate Key button renders disabled when the proxy is unlicensed, so without one this
    // fails on a product gate rather than on a regression.
    test.skip(!proxyIsPremium(), "proxy under test is unlicensed — Regenerate Key is premium-gated");
    await navigateToPage(page, Page.ApiKeys);
    await dismissFeedbackPopup(page);

    // Capture the old token first: a modal with a Copy button only proves the UI rendered.
    const before = await findKeyByAlias(page, E2E_REGENERATE_KEY_ALIAS);
    expect(before?.token, `seeded key ${E2E_REGENERATE_KEY_ALIAS} has a token`).toBeTruthy();

    const keyRow = page.getByRole("row").filter({ hasText: E2E_REGENERATE_KEY_ALIAS });
    await expect(keyRow).toBeVisible({ timeout: 10_000 });
    await keyRow.getByRole("button", { name: E2E_REGENERATE_KEY_ALIAS }).click();

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

    const keyRow = page.getByRole("row").filter({ hasText: E2E_UPDATE_LIMITS_KEY_ALIAS });
    await expect(keyRow).toBeVisible({ timeout: 10_000 });
    await keyRow.getByRole("button", { name: E2E_UPDATE_LIMITS_KEY_ALIAS }).click();

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
    // Deleting the seeded key leaves nothing for the next attempt, so the retries CI runs with are
    // guaranteed to fail and the suite cannot run twice against one database. Bring our own.
    const alias = await createDeletableKey(page);

    await navigateToPage(page, Page.ApiKeys);
    await dismissFeedbackPopup(page);

    const keyRow = page.getByRole("row").filter({ hasText: alias });
    await expect(keyRow).toBeVisible({ timeout: 10_000 });
    await keyRow.getByRole("button", { name: alias }).click();

    await expect(page.getByText("Back to Keys")).toBeVisible({ timeout: 10_000 });

    await page.getByRole("button", { name: "More key actions" }).click();
    await page.getByRole("menuitem", { name: "Delete Key" }).click();

    const modal = page.getByRole("dialog", { name: "Delete Key" });
    await expect(modal).toBeVisible({ timeout: 5_000 });
    await modal.locator("input").fill(alias);

    const deleteButton = modal.getByRole("button", { name: "Delete", exact: true });
    await expect(deleteButton).toBeEnabled();
    await deleteButton.click();

    await expect(page.getByText(/Key deleted/i).first()).toBeVisible({ timeout: 10_000 });

    // The key is gone when the management API stops returning it, not when the toast says so.
    await expect
      .poll(async () => await findKeyByAlias(page, alias), {
        message: `key ${alias} still readable from /key/list after delete`,
        timeout: 15_000,
      })
      .toBeUndefined();
  });

  test("Bulk delete only selected keys after confirmation", async ({ page }) => {
    await withBulkKeys(page, async (aliases) => {
      const before = await readKeyStates(page, aliases);
      for (const alias of aliases.slice(0, 2)) {
        await page.getByRole("row").filter({ hasText: alias }).getByRole("checkbox").check();
      }
      await page.getByRole("button", { name: "Actions", exact: true }).click();
      await page.getByRole("menuitem", { name: "Delete selected (2)" }).click();
      const modal = page.getByRole("dialog", { name: /Delete/ });
      await expect(modal.getByText(aliases[0], { exact: true })).toBeVisible();
      await expect(modal.getByText(aliases[1], { exact: true })).toBeVisible();
      await expect(modal.getByText(aliases[2], { exact: true })).toHaveCount(0);
      const deleteButton = modal.getByRole("button", { name: "Delete", exact: true });
      await expect(deleteButton).toBeDisabled();
      await modal.getByRole("textbox").fill("Confirm delete");
      await expect(deleteButton).toBeDisabled();
      await modal.getByRole("textbox").fill("confirm delete");
      await expect(deleteButton).toBeEnabled();
      await modal.getByRole("button", { name: "Cancel", exact: true }).click();
      expect(await readKeyStates(page, aliases)).toEqual(before);

      await page.getByRole("button", { name: "Actions", exact: true }).click();
      await page.getByRole("menuitem", { name: "Delete selected (2)" }).click();
      await expect(modal.getByRole("textbox")).toHaveValue("");
      await expect(deleteButton).toBeDisabled();
      await modal.getByRole("textbox").fill("confirm delete");
      const body = await captureRequestBody<{ keys: string[] }>(
        page,
        { method: "POST", urlIncludes: "/key/delete" },
        () => deleteButton.click(),
      );
      expect(body.keys).toEqual(expect.arrayContaining(before.slice(0, 2).map((key) => key?.token)));
      expect(body.keys).toHaveLength(2);
      await expect.poll(() => readKeyStates(page, aliases)).toEqual([undefined, undefined, before[2]]);
      for (const alias of aliases.slice(0, 2)) {
        await expect(page.getByRole("row").filter({ hasText: alias })).toHaveCount(0);
      }
      await page.reload();
      await expect(page.getByRole("row").filter({ hasText: aliases[2] })).toBeVisible();
      await expect(page.getByRole("checkbox")).toHaveCount(0);
    });
  });

  test("Bulk edit team, user, and budget only for selected keys", async ({ page }) => {
    await withBulkKeys(page, async (aliases) => {
      for (const [action, changes] of [
        ["Move to team", { team_id: E2E_TEAM_CRUD_ID }],
        ["Assign to user", { user_id: E2E_INTERNAL_USER_ID }],
        ["Set budget", { max_budget: 25.5, budget_duration: "7d" }],
      ] as const) {
        await test.step(action, async () => {
          const before = await readKeyStates(page, aliases);
          for (const alias of aliases.slice(0, 2)) {
            await page.getByRole("row").filter({ hasText: alias }).getByRole("checkbox").check();
          }
          await page.getByRole("button", { name: "Actions", exact: true }).click();
          await page.getByRole("menuitem", { name: action, exact: true }).click();
          const dialog = page.getByRole("dialog", { name: action, exact: true });
          if ("team_id" in changes) {
            await dialog.getByRole("combobox", { name: "Team", exact: true }).fill(E2E_TEAM_CRUD_ALIAS);
            await page.getByRole("option", { name: new RegExp(E2E_TEAM_CRUD_ALIAS) }).click();
          } else if ("user_id" in changes) {
            await dialog.getByRole("combobox", { name: "User", exact: true }).fill(E2E_INTERNAL_USER_EMAIL);
            await page.getByRole("option", { name: new RegExp(E2E_INTERNAL_USER_ID) }).click();
          } else {
            await expect(dialog.getByRole("button", { name: "Apply to 2 keys" })).toBeDisabled();
            await dialog.getByRole("spinbutton", { name: "Budget per key (USD)" }).fill("25.5");
            const period = dialog.getByRole("combobox", { name: "Reset period" });
            await expect(period).toContainText(/monthly/i);
            await period.click();
            await page.getByRole("option", { name: /^weekly$/i }).click();
          }
          await dialog.getByRole("button", { name: "Apply to 2 keys" }).click();
          await expect(dialog).toBeHidden();
          const expected = before.map((key, index) => (index < 2 ? { ...key, ...changes } : key));
          await expect.poll(() => readKeyStates(page, aliases)).toEqual(expected);
          await expect(page.getByText("0 keys selected", { exact: true })).toBeVisible();
          await expect(page.getByRole("button", { name: "Actions", exact: true })).toHaveCount(0);
        });
      }
      await page.getByRole("button", { name: "Cancel selection" }).click();
      await expect(page.getByRole("checkbox")).toHaveCount(0);
      await page.reload();
      for (const alias of aliases) {
        await expect(page.getByRole("row").filter({ hasText: alias })).toBeVisible();
      }
    });
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
    const apiKey = (
      await page.getByRole("dialog", { name: "Save your Key" }).locator("pre").filter({ hasText: /^sk-/ }).innerText()
    ).trim();
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
