import { test, expect } from "@playwright/test";
import { ADMIN_STORAGE_PATH } from "../../constants";
import { navigateToPage, dismissFeedbackPopup } from "../../helpers/navigation";
import { Page } from "../../fixtures/pages";

test.describe("AI Hub (internal admin view)", () => {
  test.use({ storageState: ADMIN_STORAGE_PATH });

  test("Make models public via the multi-step modal", async ({ page }) => {
    await navigateToPage(page, Page.ModelHubTable);

    // Open the "Select Models to Make Public" modal
    await page.getByRole("button", { name: /Select Models to Make Public/i }).click();

    const modal = page.locator(".ant-modal:visible").filter({ hasText: "Make Models Public" });
    await expect(modal).toBeVisible({ timeout: 5_000 });

    // Guard: the "Select All (N)" label only shows a count when filteredData
    // has at least one row. Asserting N>=1 here turns a missing-seed-data
    // failure into an immediate diagnostic rather than a downstream timeout
    // on the disabled-Next button or the success toast.
    await expect(modal.getByText(/Select All \(\d+\)/)).toBeVisible({ timeout: 5_000 });

    // Step 1: pick the seeded models via "Select All"
    await modal.getByText(/Select All/i).click();

    // Move to confirm step
    await modal.getByRole("button", { name: "Next" }).click();
    await expect(modal.getByText("Confirm Making Models Public")).toBeVisible({ timeout: 5_000 });

    // Submit
    await modal.getByRole("button", { name: "Make Public" }).click();

    await expect(page.getByText(/Successfully made .* model group\(s\) public/i).first()).toBeVisible({
      timeout: 15_000,
    });
  });

  test("AI Hub tab list renders Model Hub, Agent Hub, MCP Hub and Skill Hub", async ({ page }) => {
    await navigateToPage(page, Page.ModelHubTable);

    // The tab strip lives in the main view; check each tab is present and clickable.
    // (The "Claude Code Plugin Marketplace" tab from the manual-QA checklist was
    // renamed to "Skill Hub" — verify the current label here so the test stays
    // in sync with the UI.)
    //
    // Note: unlike the public /ui/model_hub_table view (test below), the admin
    // ModelHubTable renders all four tabs unconditionally — there are no `&&`
    // guards around <Tab>Agent Hub</Tab> or <Tab>MCP Hub</Tab> in the source
    // (ModelHubTable.tsx ~L436-439). Asserting all four here is intentional:
    // this pins the manual-QA contract that the AI Hub tab strip exposes
    // exactly these labels regardless of seeded agent/MCP data.
    for (const tabName of ["Model Hub", "Agent Hub", "MCP Hub", "Skill Hub"]) {
      const tab = page.getByRole("tab", { name: tabName });
      await expect(tab, `${tabName} tab should be present`).toBeVisible({ timeout: 5_000 });
      await tab.click();
    }
  });
});

test.describe("Public model hub (/ui/model_hub_table)", () => {
  // No storageState — the public page is reached anonymously with a `key` query param.

  test("Public model_hub_table loads and renders the Model Hub tab", async ({ page }) => {
    // The page expects the proxy key as the `key` query param. Use the master
    // key the e2e runner already exports — this matches what the AI Hub copy
    // button hands out.
    const masterKey = process.env.LITELLM_MASTER_KEY || "sk-1234";
    await page.goto(`/ui/model_hub_table?key=${masterKey}`);

    // Dismiss the feedback popup before asserting on the tab, so a popup
    // race can't briefly mask the tab while we're evaluating visibility.
    await dismissFeedbackPopup(page);

    // Page loads (no auth redirect) and the Model Hub tab is always present.
    // Agent Hub and MCP Hub tabs are conditionally rendered only when public
    // agents/MCP servers exist, so we don't assert on them in a fresh CI run.
    await expect(page.getByRole("tab", { name: "Model Hub" })).toBeVisible({ timeout: 10_000 });
  });
});
