import { test, expect } from "@playwright/test";
import { ADMIN_STORAGE_PATH } from "../../constants";
import { navigateToPage, dismissFeedbackPopup } from "../../helpers/navigation";
import { Page } from "../../fixtures/pages";
import { masterKey, uniqueSuffix } from "../../helpers/traffic";

test.describe("AI Hub (internal admin view)", () => {
  test.use({ storageState: ADMIN_STORAGE_PATH });

  test("Make models public via the multi-step modal", async ({ page }) => {
    await navigateToPage(page, Page.ModelHubTable);

    // Open the "Select Models to Make Public" modal
    await page.getByRole("button", { name: /Select Models to Make Public/i }).click();

    const modal = page.getByRole("dialog", { name: "Make Models Public" });
    await expect(modal).toBeVisible({ timeout: 5_000 });

    // Guard: the "Select All (N)" label only shows a count when filteredData
    // has at least one row. Asserting N>=1 here turns a missing-seed-data
    // failure into an immediate diagnostic rather than a downstream timeout
    // on the disabled-Next button or the success toast.
    await expect(modal.getByText(/Select All \(\d+\)/)).toBeVisible({ timeout: 5_000 });

    // Step 1: pick the seeded models via "Select All"
    await modal.getByRole("checkbox", { name: /Select All/ }).check();

    // Move to confirm step
    const next = modal.getByRole("button", { name: "Next" });
    await expect(next).toBeEnabled();
    await next.click();
    await expect(modal.getByText("Confirm Making Models Public")).toBeVisible({ timeout: 5_000 });

    // Submit
    await modal.getByRole("button", { name: "Make Public" }).click();

    await expect(page.getByText(/Successfully made .* model group\(s\) public/i).first()).toBeVisible({
      timeout: 15_000,
    });
  });

  test("AI Hub tab list renders Model Hub, Agent Hub, MCP Hub and Skills Hub", async ({ page }) => {
    await navigateToPage(page, Page.ModelHubTable);

    // Note: unlike the public /ui/model_hub_table view (test below), the admin
    // ModelHubTable renders all four tabs unconditionally — there are no `&&`
    // guards around <Tab>Agent Hub</Tab> or <Tab>MCP Hub</Tab> in the source
    // (ModelHubTable.tsx ~L436-439). Asserting all four here is intentional:
    // this pins the manual-QA contract that the AI Hub tab strip exposes
    // exactly these labels regardless of seeded agent/MCP data.
    for (const tabName of ["Model Hub", "Agent Hub", "MCP Hub", "Skills Hub"]) {
      const tab = page.getByRole("tab", { name: tabName });
      await expect(tab, `${tabName} tab should be present`).toBeVisible({ timeout: 5_000 });
      await tab.click();
    }
  });
});

test.describe("Public model hub (/ui/model_hub_table)", () => {
  test("shows the Model Hub and Skills Hub catalogs", async ({ page }) => {
    await page.goto("/ui/model_hub_table");
    await dismissFeedbackPopup(page);

    await expect(
      page.getByRole("tab", { name: "Model Hub", exact: true }),
    ).toBeVisible();
    await expect(
      page.getByRole("textbox", { name: "Search models", exact: true }),
    ).toBeVisible();
    await expect(page.getByRole("tab")).toHaveCount(2);
    await page.getByRole("tab", { name: "Skills Hub", exact: true }).click();
    await expect(
      page.getByRole("textbox", { name: "Search skills, tags, or people…", exact: true }),
    ).toBeVisible();
  });

  test("finds a published skill and opens its installation instructions", async ({
    page,
    request,
  }) => {
    const skillName = `e2e-catalog-${uniqueSuffix()}`;
    const domain = `catalog-${uniqueSuffix()}`;
    const auth = { Authorization: `Bearer ${masterKey()}` };
    const created = await request.post("/claude-code/plugins", {
      headers: auth,
      data: {
        name: skillName,
        source: { source: "github", repo: "example/catalog-skills" },
        description: "Catalog browser test skill",
        domain,
        version: "1.0.0",
      },
    });
    expect(
      created.ok(),
      `skill registration failed (${created.status()}): ${await created.text()}`,
    ).toBe(true);
    try {
      await page.goto("/ui/model_hub_table?tab=skills");
      await dismissFeedbackPopup(page);
      await page.getByRole("checkbox", { name: domain, exact: true }).check();
      await page
        .getByRole("textbox", { name: "Search skills, tags, or people…", exact: true })
        .fill(skillName);
      await page.getByRole("link").filter({ hasText: skillName }).click();

      const details = page.getByRole("dialog");
      await expect(
        details.getByRole("heading", { name: skillName, exact: true }),
      ).toBeVisible();
      await expect(
        details.getByText(`/plugin install ${skillName}@litellm`, {
          exact: true,
        }),
      ).toBeVisible();
      await expect(page).toHaveURL(new RegExp(`open=${skillName}`));
      await page.reload();
      await expect(
        page
          .getByRole("dialog")
          .getByRole("heading", { name: skillName, exact: true }),
      ).toBeVisible();
    } finally {
      const deleted = await request.delete(
        `/claude-code/plugins/${skillName}`,
        { headers: auth },
      );
      expect(deleted.ok(), `skill cleanup failed (${deleted.status()})`).toBe(
        true,
      );
    }
  });

  test("retired catalog links return to Models and clear the old selection", async ({
    page,
  }) => {
    for (const tab of ["agents", "mcp"]) {
      await page.goto(
        `/ui/model_hub_table?tab=${tab}&open=retired-entry&key=preserved`,
      );
      await dismissFeedbackPopup(page);
      await expect(
        page.getByRole("tab", { name: "Model Hub", exact: true }),
      ).toHaveAttribute("aria-selected", "true");
      await expect(page).toHaveURL(/model_hub_table\?key=preserved$/);
      await expect(page.getByRole("dialog")).toHaveCount(0);
    }
  });
});
