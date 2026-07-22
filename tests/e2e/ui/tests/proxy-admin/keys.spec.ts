import { test, expect } from "@playwright/test";
import {
  ADMIN_STORAGE_PATH,
  E2E_DELETE_KEY_ALIAS,
  E2E_REGENERATE_KEY_ALIAS,
  E2E_UPDATE_LIMITS_KEY_ALIAS,
  E2E_INTERNAL_USER_KEY_ALIAS,
  E2E_TEAM_CRUD_ALIAS,
  E2E_UI_OPENAI_MODEL,
} from "../../constants";
import { Page } from "../../fixtures/pages";
import { navigateToPage, dismissFeedbackPopup } from "../../helpers/navigation";

test.describe("Proxy Admin - Keys", () => {
  test.use({ storageState: ADMIN_STORAGE_PATH });

  test("Create a key in a team", async ({ page }) => {
    await navigateToPage(page, Page.ApiKeys);
    await dismissFeedbackPopup(page);

    // Click "+ Create New Key" button
    await page.getByRole("button", { name: /Create New Key/i }).click();

    // Wait for the key creation modal
    await expect(page.getByText("Key Ownership")).toBeVisible({ timeout: 10_000 });

    // Fill key name (has data-testid="base-input" in the built UI)
    const keyName = `e2e-admin-key-${Date.now()}`;
    await page.getByTestId("base-input").fill(keyName);

    // Select team — the team dropdown has placeholder "Search or select a team"
    const teamSelect = page.locator(".ant-select", { hasText: "Search or select a team" });
    await teamSelect.click();
    await page.keyboard.type(E2E_TEAM_CRUD_ALIAS);
    await page.locator(".ant-select-dropdown:visible").getByText(E2E_TEAM_CRUD_ALIAS).first().click();

    // Select models
    await page.locator(".ant-select-selection-overflow").click();
    await page.locator(".ant-select-dropdown:visible").getByText("All Team Models").click();
    await page.keyboard.press("Escape");

    // Submit
    await page.getByRole("button", { name: "Create Key", exact: true }).click();

    // Success shows "Save your Key" in a second dialog
    await expect(page.getByText("Save your Key")).toBeVisible({ timeout: 10_000 });
    await page.keyboard.press("Escape");

    // Verify the new key appears in the table
    await expect(page.getByText(keyName)).toBeVisible({ timeout: 10_000 });
  });

  test("Regenerate key", async ({ page }) => {
    await navigateToPage(page, Page.ApiKeys);
    await dismissFeedbackPopup(page);

    // Key IDs are rendered as buttons in the table
    const keyRow = page.locator("tr", { hasText: E2E_REGENERATE_KEY_ALIAS });
    await expect(keyRow).toBeVisible({ timeout: 10_000 });
    await keyRow.locator("button").first().click();

    await expect(page.getByText("Back to Keys")).toBeVisible({ timeout: 10_000 });

    await page.getByRole("button", { name: "Regenerate Key" }).click();

    // Scope to the modal — the Regenerate button has an icon whose aria-label
    // ("sync") is concatenated into the button's accessible name, and the
    // "Regenerate Key" button is still in the DOM behind the modal.
    const modal = page.locator(".ant-modal:visible");
    await modal.getByRole("button", { name: /Regenerate/ }).click();

    // Success view shows a Copy button in the footer (text varies between modal versions)
    await expect(modal.getByRole("button", { name: /Copy.*Key/ })).toBeVisible({ timeout: 20_000 });
  });

  test("Update key TPM and RPM limits", async ({ page }) => {
    await navigateToPage(page, Page.ApiKeys);
    await dismissFeedbackPopup(page);

    const keyRow = page.locator("tr", { hasText: E2E_UPDATE_LIMITS_KEY_ALIAS });
    await expect(keyRow).toBeVisible({ timeout: 10_000 });
    await keyRow.locator("button").first().click();

    await expect(page.getByText("Back to Keys")).toBeVisible({ timeout: 10_000 });

    await page.getByRole("tab", { name: "Settings" }).click();
    await page.getByRole("button", { name: "Edit Settings" }).click();

    await page.getByRole("spinbutton", { name: "TPM Limit" }).fill("123");
    await page.getByRole("spinbutton", { name: "RPM Limit" }).fill("456");
    await page.getByRole("button", { name: "Save Changes" }).click();

    await expect(page.getByRole("paragraph").filter({ hasText: "TPM: 123" })).toBeVisible({ timeout: 10_000 });
    await expect(page.getByRole("paragraph").filter({ hasText: "RPM: 456" })).toBeVisible({ timeout: 10_000 });
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

    const modal = page.locator(".ant-modal:visible");
    await expect(modal).toBeVisible({ timeout: 5_000 });
    await modal.locator("input").fill(E2E_DELETE_KEY_ALIAS);

    const deleteButton = modal.getByRole("button", { name: "Delete", exact: true });
    await expect(deleteButton).toBeEnabled();
    await deleteButton.click();

    await expect(page.getByText(/Key deleted/i).first()).toBeVisible({ timeout: 10_000 });
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
    await page.getByTestId("base-input").fill(keyName);

    // No team selection — leave team dropdown empty so the key is owned by the admin user

    // Select models — open the multi-select and pick the all-models meta-option.
    // With no team selected the modal offers "All Proxy Models"; the team-scoped
    // "All Team Models" option only appears once a team is picked.
    await page.locator(".ant-select-selection-overflow").click();
    await page.locator(".ant-select-dropdown:visible").getByText("All Proxy Models").click();
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
    await page.getByTestId("base-input").fill(keyName);

    // Open the model multi-select and pick a single specific model. Use
    // getByRole("option", ...) to avoid the strict-mode collision between
    // the option container and its inner text node.
    const modelName = E2E_UI_OPENAI_MODEL;
    await page.locator(".ant-select-selection-overflow").click();
    const option = page.locator(".ant-select-dropdown:visible").getByRole("option", { name: modelName, exact: true });
    await option.waitFor({ state: "attached" });
    // Dispatch the click via the DOM — antd's dropdown can render the option
    // off-viewport during the open animation, which trips Playwright's
    // visibility/stability checks. The click handler fires regardless.
    await option.evaluate((el: HTMLElement) => el.click());
    await page.keyboard.press("Escape");

    await page.getByRole("button", { name: "Create Key", exact: true }).click();

    await expect(page.getByText("Save your Key")).toBeVisible({ timeout: 10_000 });

    // Grab the new key from the success modal (rendered inside a <pre>) and
    // verify it can call /chat/completions for the model it was scoped to.
    // The deployment routes to a real provider, so this costs a fraction of
    // a cent and proves the whole key -> router -> provider path.
    const apiKey = (await page.locator(".ant-modal:visible pre").innerText()).trim();
    expect(apiKey).toMatch(/^sk-/);

    const response = await page.request.post("/chat/completions", {
      headers: { Authorization: `Bearer ${apiKey}` },
      data: {
        model: modelName,
        messages: [{ role: "user", content: "Reply with the single word: pong" }],
      },
    });
    expect(response.status()).toBe(200);
    const body = await response.json();
    expect(body.choices?.[0]?.message?.content?.length).toBeGreaterThan(0);

    await page.keyboard.press("Escape");

    await expect(page.getByText(keyName)).toBeVisible({ timeout: 10_000 });
  });
});
