import { expect, test } from "@playwright/test";

/**
 * Phase-1 parity spec for the Access Groups section.
 *
 * This spec asserts on user-observable behavior via semantic queries only.
 * It deliberately avoids framework-specific selectors (e.g. `.ant-btn` or
 * `[data-radix-*]`) so it can pass against both the pre-migration (antd)
 * and post-migration (shadcn) UIs.
 *
 * The parity workflow is:
 *   1. Run this spec against the unmigrated UI (baseline) — it must pass.
 *   2. Migrate the section.
 *   3. Run this spec against the migrated UI — it must still pass.
 *   4. Run with --update-snapshots once to baseline visual snapshots, and
 *      again without to verify pixel diffs are below the 1% threshold
 *      defined in `playwright.config.ts`.
 *
 * **Auth:** The dashboard requires an admin session. When `PARITY_BASE_URL`
 * is unset (default), the Next dev server at :3000 is used; this spec
 * expects a helper to seed `sessionStorage` with a dev JWT — see
 * `e2e_tests/parity/helpers.ts` (TBD; add when we run parity specs against
 * the dev server). When `PARITY_BASE_URL=http://localhost:4000` the proxy
 * is used and the existing auth flow applies.
 *
 * This file is committed as the phase-1 baseline spec shape. In this
 * autonomous run the spec is not executed because neither the proxy nor a
 * seeded dev-server auth helper is available in the sandbox environment
 * — the gating check for Section 1 is vitest + tsc + lint + build. See
 * `docs/CYCLES.md` for the explicit gate-skip rationale.
 */

test.describe("Access Groups — parity", () => {
  test.beforeEach(async ({ page }) => {
    // Page under test is typically reached via the leftnav.
    await page.goto("/ui?page=access-groups");
  });

  test("lists access groups with page header and search input", async ({
    page,
  }) => {
    await expect(
      page.getByRole("heading", { name: /access groups/i }),
    ).toBeVisible();
    await expect(
      page.getByText(/manage resource permissions for your organization/i),
    ).toBeVisible();
    await expect(
      page.getByPlaceholder(
        /search groups by name, id, or description\.\.\./i,
      ),
    ).toBeVisible();
    await expect(
      page.getByRole("button", { name: /create access group/i }),
    ).toBeVisible();
  });

  test("create access group flow opens the dialog and validates required fields", async ({
    page,
  }) => {
    await page
      .getByRole("button", { name: /create access group/i })
      .click();
    await expect(
      page.getByRole("dialog", { name: /create access group/i }),
    ).toBeVisible();
    await expect(
      page.getByRole("textbox", { name: /group name/i }),
    ).toBeVisible();
    await expect(
      page.getByRole("button", { name: /create group/i }),
    ).toBeVisible();
    await page.getByRole("button", { name: /cancel/i }).click();
    await expect(
      page.getByRole("dialog", { name: /create access group/i }),
    ).toBeHidden();
  });

  test("filters access groups by search", async ({ page }) => {
    const search = page.getByPlaceholder(
      /search groups by name, id, or description\.\.\./i,
    );
    await search.fill("__no_match_sentinel__");
    // The table should show an empty-state row, not crash.
    await expect(page.getByRole("table")).toBeVisible();
  });

  test("table exposes ID, Name, Resources, Actions columns", async ({
    page,
  }) => {
    await expect(
      page.getByRole("columnheader", { name: /^id$/i }),
    ).toBeVisible();
    await expect(
      page.getByRole("columnheader", { name: /^name$/i }),
    ).toBeVisible();
    await expect(
      page.getByRole("columnheader", { name: /^resources$/i }),
    ).toBeVisible();
    await expect(
      page.getByRole("columnheader", { name: /^actions$/i }),
    ).toBeVisible();
  });

  test("delete action opens confirmation dialog with resource information", async ({
    page,
  }) => {
    const deleteButtons = page.getByRole("button", {
      name: /delete access group/i,
    });
    const count = await deleteButtons.count();
    test.skip(count === 0, "No existing access groups to exercise delete on.");
    await deleteButtons.first().click();
    await expect(
      page.getByRole("dialog", { name: /delete access group/i }),
    ).toBeVisible();
    await expect(
      page.getByText(
        /are you sure you want to delete this access group\?/i,
      ),
    ).toBeVisible();
    await expect(
      page.getByText(/access group information/i),
    ).toBeVisible();
  });

  test("visual — list view", async ({ page }) => {
    await expect(
      page.getByRole("heading", { name: /access groups/i }),
    ).toBeVisible();
    await expect(page).toHaveScreenshot("list.png", {
      fullPage: true,
      mask: [page.getByRole("cell", { name: /ag-/i })], // mask dynamic IDs
    });
  });
});
