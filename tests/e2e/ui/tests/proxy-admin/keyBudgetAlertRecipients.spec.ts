import { test, expect, type Page as PlaywrightPage } from "@playwright/test";
import {
  ADMIN_STORAGE_PATH,
  E2E_TEAM_CRUD_ALIAS,
  E2E_TEAM_CRUD_ID,
} from "../../constants";
import { Page } from "../../fixtures/pages";
import {
  dismissFeedbackPopup,
  navigateToPage,
  openKeyDetail,
} from "../../helpers/navigation";
import { readBack } from "../../helpers/roundTrip";
import {
  createVirtualKey,
  deleteVirtualKey,
  uniqueSuffix,
} from "../../helpers/traffic";

const SHARED_MAILBOX = "shared-mailbox@example.com";
const DISTRIBUTION_LIST = "finance-dl@example.com";
const RECIPIENTS_PLACEHOLDER =
  "Additional recipients, e.g. finance@example.com";

async function keyMetadata(
  page: PlaywrightPage,
  token: string,
): Promise<Record<string, unknown>> {
  const body = await readBack<{ info: { metadata: Record<string, unknown> } }>(
    page,
    `/key/info?key=${encodeURIComponent(token)}`,
  );
  return body.info.metadata;
}

async function tokenForAlias(
  page: PlaywrightPage,
  alias: string,
): Promise<string | undefined> {
  const body = await readBack<{ keys: { key_alias: string; token: string }[] }>(
    page,
    `/key/list?key_alias=${encodeURIComponent(alias)}&return_full_object=true&size=100`,
  );
  return body.keys.find((row) => row.key_alias === alias)?.token;
}

async function addAlertRow(
  page: PlaywrightPage,
  threshold: string,
  emails: string,
): Promise<void> {
  await page
    .getByRole("button", { name: "Add Budget Alert Threshold" })
    .click();
  await page.getByPlaceholder("% of budget").last().fill(threshold);
  await page.getByPlaceholder(RECIPIENTS_PLACEHOLDER).last().fill(emails);
}

test.describe("Proxy Admin - Key budget alert recipients", () => {
  test.use({ storageState: ADMIN_STORAGE_PATH });

  test("Create Key stores the alert recipients next to the metadata JSON", async ({
    page,
  }) => {
    const alias = `e2e-alert-create-${uniqueSuffix()}`;
    await navigateToPage(page, Page.ApiKeys);
    await dismissFeedbackPopup(page);
    await page.getByRole("button", { name: /Create New Key/i }).click();
    await page.getByLabel(/Key Name/).fill(alias);
    await page.getByTestId("team-dropdown").getByRole("combobox").click();
    await page.keyboard.type(E2E_TEAM_CRUD_ALIAS);
    await page
      .getByRole("option", { name: E2E_TEAM_CRUD_ALIAS })
      .first()
      .click();
    await page.getByRole("combobox", { name: "Select models" }).click();
    await page
      .getByRole("option", { name: "All Team Models", exact: true })
      .click();
    await page.keyboard.press("Escape");

    await page.getByRole("button", { name: /Optional Settings/i }).click();
    await page.getByLabel(/Max Budget \(USD\)/).fill("10");
    await addAlertRow(page, "80", `${SHARED_MAILBOX}, ${DISTRIBUTION_LIST}`);
    await page.getByLabel("Metadata").fill('{"cost_center": "ops"}');
    await page.getByRole("button", { name: "Create Key", exact: true }).click();
    await expect(page.getByText("Save your Key")).toBeVisible({
      timeout: 10_000,
    });

    const token = await tokenForAlias(page, alias);
    expect(token, `key ${alias} readable from /key/list`).toBeTruthy();
    try {
      expect(await keyMetadata(page, token!)).toMatchObject({
        cost_center: "ops",
        max_budget_alert_emails: { "80": [SHARED_MAILBOX, DISTRIBUTION_LIST] },
      });
    } finally {
      await deleteVirtualKey(page.request, token!);
    }
  });

  test("Edit Key round-trips the rows, keeps the rest of the metadata, and clears them when every row is removed", async ({
    page,
  }) => {
    const alias = `e2e-alert-edit-${uniqueSuffix()}`;
    const created = await createVirtualKey(page.request, {
      key_alias: alias,
      team_id: E2E_TEAM_CRUD_ID,
      max_budget: 10,
      metadata: {
        cost_center: "ops",
        max_budget_alert_emails: { "80": [SHARED_MAILBOX] },
      },
    });
    try {
      await navigateToPage(page, Page.ApiKeys);
      await dismissFeedbackPopup(page);
      await openKeyDetail(page, alias);
      await page.getByRole("tab", { name: "Settings" }).click();
      await page.getByRole("button", { name: "Edit Settings" }).click();

      await expect(page.getByPlaceholder("% of budget")).toHaveValue("80");
      await expect(page.getByPlaceholder(RECIPIENTS_PLACEHOLDER)).toHaveValue(
        SHARED_MAILBOX,
      );
      expect(
        JSON.parse(await page.getByLabel("Metadata").inputValue()),
      ).toEqual({ cost_center: "ops" });

      await page
        .getByRole("button", { name: "Remove budget alert threshold" })
        .click();
      await addAlertRow(page, "50", DISTRIBUTION_LIST);
      await page.getByRole("button", { name: "Save Changes" }).click();
      await expect
        .poll(() => keyMetadata(page, created.token), { timeout: 20_000 })
        .toMatchObject({
          cost_center: "ops",
          max_budget_alert_emails: { "50": [DISTRIBUTION_LIST] },
        });

      await page.getByRole("button", { name: "Edit Settings" }).click();
      await expect(page.getByPlaceholder("% of budget")).toHaveValue("50");
      await page
        .getByRole("button", { name: "Remove budget alert threshold" })
        .click();
      await page.getByRole("button", { name: "Save Changes" }).click();
      await expect
        .poll(
          async () =>
            "max_budget_alert_emails" in
            (await keyMetadata(page, created.token)),
          { timeout: 20_000 },
        )
        .toBe(false);
      expect(await keyMetadata(page, created.token)).toMatchObject({
        cost_center: "ops",
      });
    } finally {
      await deleteVirtualKey(page.request, created.token);
    }
  });
});
