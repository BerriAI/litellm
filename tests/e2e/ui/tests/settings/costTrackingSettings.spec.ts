import {
  expect,
  test,
  type APIRequestContext,
  type Locator,
  type Page,
  type Request,
} from "@playwright/test";
import { ADMIN_STORAGE_PATH } from "../../constants";
import { masterKey, rootPath, uniqueSuffix } from "../../helpers/traffic";

const DISCOUNTS = "/config/cost_discount_config";
const MARGINS = "/config/cost_margin_config";
const DISCOUNT_PREFIX = "openai/e2e-";
const MARGIN_AMOUNT = 0.008;
const MARGIN_PROVIDER_POOLS = [
  [
    ["anthropic", "Anthropic"],
    ["bedrock", "Amazon Bedrock"],
    ["ai21", "Ai21"],
  ],
  [
    ["cohere", "Cohere"],
    ["cerebras", "Cerebras"],
    ["cloudflare", "Cloudflare"],
  ],
  [
    ["groq", "Groq"],
    ["fireworks_ai", "Fireworks AI"],
    ["nvidia_nim", "Nvidia Nim"],
  ],
  [
    ["mistral", "Mistral AI"],
    ["ollama", "Ollama"],
  ],
  [
    ["perplexity", "Perplexity"],
    ["replicate", "Replicate"],
  ],
] as const;
const MARGIN_PROVIDERS = MARGIN_PROVIDER_POOLS.flat();
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

async function openCostSettings(page: Page, path: string, section: RegExp) {
  const reads: Promise<CostConfig>[] = [];
  const recordConfigRead = (request: Request) => {
    if (
      request.method() !== "GET" ||
      new URL(request.url()).pathname !== rootPath() + path
    )
      return;
    reads.push(
      request
        .response()
        .then(async (response) =>
          response?.ok() ? (await response.json()).values : {},
        ),
    );
  };
  page.on("requestfinished", recordConfigRead);
  const loading = configResponse(page, "GET", path);
  await page.goto(`${rootPath()}/ui/cost-tracking/`);
  const response = await loading;
  expect(response.ok(), `Load ${path} (HTTP ${response.status()})`).toBe(true);
  await page.getByRole("button", { name: section }).click();
  await expect(
    page.getByText("Loading configuration...", { exact: true }),
  ).toBeHidden();

  return async (
    isTestEntry: (key: string, value: CostConfig[string]) => boolean,
  ) => {
    page.off("requestfinished", recordConfigRead);
    const [first, ...others] = await Promise.all(reads);
    return {
      occupied: new Set([first, ...others].flatMap(Object.keys)),
      existing: Object.fromEntries(
        Object.entries(first).filter(
          ([key, value]) =>
            !isTestEntry(key, value) &&
            others.every(
              (recent) => JSON.stringify(recent[key]) === JSON.stringify(value),
            ),
        ),
      ),
    };
  };
}

async function selectProvider(page: Page, dialog: Locator, label: string) {
  await dialog.getByRole("combobox", { name: "Provider" }).fill(label);
  await page
    .getByRole("option")
    .filter({ has: page.getByText(label, { exact: true }) })
    .click();
}

async function removeTestEntry(
  request: APIRequestContext,
  path: string,
  key: string,
) {
  try {
    const options = {
      headers: { Authorization: `Bearer ${masterKey()}`, Connection: "close" },
      timeout: 1500,
    };
    const response = await request.get(rootPath() + path, options);
    if (!response.ok()) return;
    const current: CostConfig = (await response.json()).values;
    if (!Object.hasOwn(current, key)) return;
    const data = Object.fromEntries(
      Object.entries(current).filter(([entry]) => entry !== key),
    );
    await request.patch(rootPath() + path, { ...options, data });
  } catch {
    return;
  }
}

test.describe("Cost tracking settings", () => {
  test.describe.configure({ mode: "serial" });
  test.use({ storageState: ADMIN_STORAGE_PATH });

  test("lets an admin correct an invalid discount and save it for a model pattern", async ({
    page,
    request,
  }) => {
    const retainedEntries = await openCostSettings(
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
    const key = `openai/${pattern}`;
    const { existing } = await retainedEntries((entry) =>
      entry.startsWith(DISCOUNT_PREFIX),
    );
    const saving = configResponse(page, "PATCH", DISCOUNTS);
    await add.click();
    const response = await saving;
    try {
      expect(response.ok(), `Save discount (HTTP ${response.status()})`).toBe(
        true,
      );
      const body: CostConfig = response.request().postDataJSON();
      expect(body[key]).toBe(0.125);
      for (const [entry, value] of Object.entries(existing)) {
        expect(body[entry], `Preserve discount ${entry}`).toEqual(value);
      }
      await expect(
        page.getByText("Discount configuration updated successfully", {
          exact: true,
        }),
      ).toBeVisible();
      await expect(dialog).toBeHidden();
    } finally {
      if (response.ok()) await removeTestEntry(request, DISCOUNTS, key);
    }
  });

  test("keeps both margin drafts when switching types and saves only the selected fixed amount", async ({
    page,
    request,
  }, testInfo) => {
    const retainedEntries = await openCostSettings(
      page,
      MARGINS,
      /^Fee\/Price Margin/,
    );
    await page
      .getByRole("button", { name: "+ Add Provider Margin", exact: true })
      .click();
    const dialog = page.getByRole("dialog", { name: "Add Provider Margin" });
    const { occupied, existing } = await retainedEntries(
      (key, value) =>
        MARGIN_PROVIDERS.some(([provider]) => provider === key) &&
        typeof value === "object" &&
        value.fixed_amount === MARGIN_AMOUNT &&
        value.percentage === undefined,
    );
    const candidates =
      MARGIN_PROVIDER_POOLS[
        testInfo.repeatEachIndex % MARGIN_PROVIDER_POOLS.length
      ];
    const provider = candidates.find(([key]) => !occupied.has(key));
    if (!provider)
      throw new Error("No unused provider is available for the margin test");
    const [key, label] = provider;
    await selectProvider(page, dialog, label);
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
    await fixed.fill(String(MARGIN_AMOUNT));
    await percentageType.click();
    await expect(percentage).toHaveValue("25");
    await fixedType.click();
    await expect(fixed).toHaveValue(String(MARGIN_AMOUNT));

    const saving = configResponse(page, "PATCH", MARGINS);
    await dialog
      .getByRole("button", { name: "Add Provider Margin", exact: true })
      .click();
    const response = await saving;
    try {
      expect(response.ok(), `Save margin (HTTP ${response.status()})`).toBe(
        true,
      );
      const body: CostConfig = response.request().postDataJSON();
      expect(body[key]).toEqual({ fixed_amount: MARGIN_AMOUNT });
      for (const [entry, value] of Object.entries(existing)) {
        expect(body[entry], `Preserve margin ${entry}`).toEqual(value);
      }
      await expect(
        page.getByText("Margin configuration updated successfully", {
          exact: true,
        }),
      ).toBeVisible();
      await expect(dialog).toBeHidden();
    } finally {
      if (response.ok()) await removeTestEntry(request, MARGINS, key);
    }
  });
});
