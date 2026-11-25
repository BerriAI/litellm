import {
  expect,
  test,
  type APIRequestContext,
  type Locator,
  type Page,
} from "@playwright/test";
import { ADMIN_STORAGE_PATH } from "../../constants";
import { masterKey, rootPath, uniqueSuffix } from "../../helpers/traffic";

const DISCOUNTS = "/config/cost_discount_config";
const MARGINS = "/config/cost_margin_config";
type CostConfig = Record<
  string,
  number | { percentage?: number; fixed_amount?: number }
>;

const configResponse = (page: Page, method: "GET" | "PATCH", path: string) =>
  page.waitForResponse((response) => {
    return (
      response.request().method() === method &&
      new URL(response.url()).pathname === rootPath() + path
    );
  });

async function openCostSettings(
  page: Page,
  path: string,
  section: RegExp,
): Promise<CostConfig> {
  const loading = configResponse(page, "GET", path);
  await page.goto(`${rootPath()}/ui/cost-tracking/`);
  const response = await loading;
  expect(response.ok(), `Load ${path} (HTTP ${response.status()})`).toBe(true);
  await page.getByRole("button", { name: section }).click();
  return (await response.json()).values;
}

async function selectProvider(page: Page, dialog: Locator, label: string) {
  await dialog.getByRole("combobox", { name: "Provider" }).fill(label);
  await page
    .getByRole("option")
    .filter({ has: page.getByText(label, { exact: true }) })
    .click();
}

async function restoreConfig(
  request: APIRequestContext,
  path: string,
  config: CostConfig,
) {
  const response = await request.patch(rootPath() + path, {
    headers: { Authorization: `Bearer ${masterKey()}` },
    data: config,
  });
  expect(response.ok(), `Restore ${path} (HTTP ${response.status()})`).toBe(
    true,
  );
}

test.describe("Cost tracking settings", () => {
  test.describe.configure({ mode: "serial" });
  test.use({ storageState: ADMIN_STORAGE_PATH });

  test("lets an admin correct an invalid discount and save it for a model pattern", async ({
    page,
    request,
  }) => {
    const original = await openCostSettings(
      page,
      DISCOUNTS,
      /^Provider Discounts/,
    );
    await page
      .getByRole("button", { name: "+ Add Provider Discount", exact: true })
      .click();
    const dialog = page.getByRole("dialog", { name: "Add Provider Discount" });
    const pattern = `e2e-${uniqueSuffix()}-*`;
    await selectProvider(page, dialog, "OpenAI");
    await dialog
      .getByRole("textbox", { name: "Model pattern" })
      .fill(`  ${pattern} `);
    const percentage = dialog.getByLabel("Discount Percentage");
    const add = dialog.getByRole("button", {
      name: "Add Provider Discount",
      exact: true,
    });
    await percentage.fill("105");
    await add.click();
    await expect(
      page.getByText("Discount must be between 0% and 100%", { exact: true }),
    ).toBeVisible();
    await expect(
      dialog.getByRole("textbox", { name: "Model pattern" }),
    ).toHaveValue(`  ${pattern} `);

    await percentage.fill("12.5");
    const saving = configResponse(page, "PATCH", DISCOUNTS);
    await add.click();
    const response = await saving;
    try {
      expect(response.ok(), `Save discount (HTTP ${response.status()})`).toBe(
        true,
      );
      expect(response.request().postDataJSON()).toEqual({
        ...original,
        [`openai/${pattern}`]: 0.125,
      });
      await expect(
        page.getByText("Discount configuration updated successfully", {
          exact: true,
        }),
      ).toBeVisible();
      await expect(dialog).toBeHidden();
    } finally {
      await restoreConfig(request, DISCOUNTS, original);
    }
  });

  test("keeps both margin drafts when switching types and saves only the selected fixed amount", async ({
    page,
    request,
  }) => {
    const original = await openCostSettings(
      page,
      MARGINS,
      /^Fee\/Price Margin/,
    );
    await page
      .getByRole("button", { name: "+ Add Provider Margin", exact: true })
      .click();
    const dialog = page.getByRole("dialog", { name: "Add Provider Margin" });
    await selectProvider(page, dialog, "Global (All Providers)");
    const percentage = dialog.getByRole("textbox", {
      name: "Margin Percentage",
    });
    const fixed = dialog.getByRole("textbox", { name: "Fixed Margin Amount" });
    const percentageType = dialog.getByRole("radio", {
      name: "Percentage-based",
    });
    const fixedType = dialog.getByRole("radio", { name: "Fixed Amount" });
    await percentage.fill("25");
    await fixedType.click();
    await fixed.fill("0.008");
    await percentageType.click();
    await expect(percentage).toHaveValue("25");
    await fixedType.click();
    await expect(fixed).toHaveValue("0.008");

    const saving = configResponse(page, "PATCH", MARGINS);
    await dialog
      .getByRole("button", { name: "Add Provider Margin", exact: true })
      .click();
    const response = await saving;
    try {
      expect(response.ok(), `Save margin (HTTP ${response.status()})`).toBe(
        true,
      );
      expect(response.request().postDataJSON()).toEqual({
        ...original,
        global: { fixed_amount: 0.008 },
      });
      await expect(
        page.getByText("Margin configuration updated successfully", {
          exact: true,
        }),
      ).toBeVisible();
      await expect(dialog).toBeHidden();
    } finally {
      await restoreConfig(request, MARGINS, original);
    }
  });
});
