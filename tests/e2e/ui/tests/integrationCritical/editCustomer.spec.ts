import { test as base, expect, type APIRequestContext, type Locator, type Page, type Request } from "@playwright/test";
import { dismissFeedbackPopup } from "../../helpers/navigation";
import { masterKey, uniqueSuffix } from "../../helpers/traffic";

const ORIGINAL_ALIAS = "Original Alias";
const ORIGINAL_BUDGET = 25;

type Customer = Readonly<{ id: string; budgetId: string }>;

const headers = () => ({ Authorization: `Bearer ${masterKey()}` });

async function addCustomerToBudget(request: APIRequestContext, customer: Customer): Promise<void> {
  const created = await request.post("/customer/new", {
    headers: headers(),
    data: { user_id: customer.id, alias: ORIGINAL_ALIAS, budget_id: customer.budgetId },
  });
  expect(created.ok(), await created.text()).toBe(true);
}

const test = base.extend<{ customer: Customer }>({
  customer: async ({ request }, use) => {
    const id = `e2e-edit-customer-${uniqueSuffix()}`;
    const customer = { id, budgetId: `${id}-budget` };
    const budget = await request.post("/budget/new", {
      headers: headers(),
      data: { budget_id: customer.budgetId, max_budget: ORIGINAL_BUDGET },
    });
    expect(budget.ok(), await budget.text()).toBe(true);

    try {
      await addCustomerToBudget(request, customer);
      await use(customer);
    } finally {
      try {
        const removed = await request.post("/customer/delete", {
          headers: headers(),
          data: { user_ids: [customer.id] },
        });
        expect(removed.ok() || removed.status() === 404, await removed.text()).toBe(true);
      } finally {
        const removed = await request.post("/budget/delete", { headers: headers(), data: { id: customer.budgetId } });
        expect(removed.ok(), await removed.text()).toBe(true);
      }
    }
  },
});

const isCustomerUpdate = (req: Request) =>
  req.method() === "POST" && new URL(req.url()).pathname.replace(/\/+$/, "").endsWith("/customer/update");

const editButton = (page: Page) => page.getByRole("button", { name: "Edit", exact: true });
const editDialog = (page: Page) => page.getByRole("dialog", { name: "Edit customer" });
const aliasInput = (dialog: Locator) => dialog.getByLabel("Alias", { exact: true });
const budgetInput = (dialog: Locator) => dialog.getByLabel("Max Budget (USD)", { exact: true });

async function openCustomer(page: Page, customer: Customer, heading: string): Promise<void> {
  await expect(async () => {
    await page.goto(`/ui/customers?id=${encodeURIComponent(customer.id)}`);
    await expect(page.getByRole("heading", { level: 1, name: heading })).toBeVisible({ timeout: 10_000 });
  }).toPass({ timeout: 60_000 });
  await dismissFeedbackPopup(page);
}

async function openEditDialog(page: Page): Promise<Locator> {
  await editButton(page).click();
  const dialog = editDialog(page);
  await expect(aliasInput(dialog)).toBeVisible();
  return dialog;
}

async function saveAndCapture(
  page: Page,
  dialog: Locator,
  name = "Save",
): Promise<{ body: unknown; status: number | undefined }> {
  const sent = page.waitForRequest(isCustomerUpdate);
  await dialog.getByRole("button", { name, exact: true }).click();
  const req = await sent;
  return { body: req.postDataJSON(), status: (await req.response())?.status() };
}

test.describe("Edit customer dialog", () => {
  test.beforeEach(async ({ page }) => {
    await page.goto("/ui/login");
    await page.getByPlaceholder("Enter your username").fill("admin");
    await page.getByPlaceholder("Enter your password").fill(masterKey());
    await page.getByRole("button", { name: "Login", exact: true }).click();
    await expect(page).toHaveURL((url) => url.pathname.startsWith("/ui") && !url.pathname.includes("login"));
  });

  test("saving a new alias and budget updates the customer and reopens with the saved values", async ({
    page,
    customer,
  }) => {
    await openCustomer(page, customer, ORIGINAL_ALIAS);
    const dialog = await openEditDialog(page);
    await expect(aliasInput(dialog)).toHaveValue(ORIGINAL_ALIAS);
    await expect(budgetInput(dialog)).toHaveValue(String(ORIGINAL_BUDGET));

    await aliasInput(dialog).fill("Renamed Alias");
    await budgetInput(dialog).fill("40");
    const saved = await saveAndCapture(page, dialog);

    expect(saved.body).toEqual({ user_id: customer.id, alias: "Renamed Alias", blocked: false, max_budget: 40 });
    await expect(page.getByText("Customer updated")).toBeVisible();
    await expect(dialog).toBeHidden();
    await expect(page.getByRole("heading", { level: 1, name: "Renamed Alias" })).toBeVisible();

    const reopened = await openEditDialog(page);
    await expect(aliasInput(reopened)).toHaveValue("Renamed Alias");
    await expect(budgetInput(reopened)).toHaveValue("40");
  });

  test("the budget field keeps only the digits and decimal point the user types", async ({ page, customer }) => {
    await openCustomer(page, customer, ORIGINAL_ALIAS);
    const dialog = await openEditDialog(page);

    await budgetInput(dialog).fill("");
    await budgetInput(dialog).pressSequentially("12a.5$");
    await expect(budgetInput(dialog)).toHaveValue("12.5");
    const saved = await saveAndCapture(page, dialog);

    expect(saved.body).toEqual({ user_id: customer.id, alias: ORIGINAL_ALIAS, blocked: false, max_budget: 12.5 });
  });

  test("clearing the alias and budget sends null for both", async ({ page, customer }) => {
    await openCustomer(page, customer, ORIGINAL_ALIAS);
    const dialog = await openEditDialog(page);

    await aliasInput(dialog).fill("");
    await budgetInput(dialog).fill("");
    const saved = await saveAndCapture(page, dialog);

    expect(saved.body).toEqual({ user_id: customer.id, alias: null, blocked: false, max_budget: null });
  });

  test("a save the server rejects shows its error, keeps the typed values, and the retry goes through", async ({
    page,
    request,
    customer,
  }) => {
    await openCustomer(page, customer, ORIGINAL_ALIAS);
    const dialog = await openEditDialog(page);
    await aliasInput(dialog).fill("Retry Alias");
    const removed = await request.post("/customer/delete", { headers: headers(), data: { user_ids: [customer.id] } });
    expect(removed.ok(), await removed.text()).toBe(true);
    const expectedBody = { user_id: customer.id, alias: "Retry Alias", blocked: false, max_budget: ORIGINAL_BUDGET };

    const rejected = await saveAndCapture(page, dialog);
    expect(rejected).toEqual({ body: expectedBody, status: 404 });
    await expect(page.getByText(/^Failed to update customer: .*does not exist/)).toContainText(customer.id);
    await expect(dialog).toBeVisible();
    await expect(aliasInput(dialog)).toHaveValue("Retry Alias");

    await addCustomerToBudget(request, customer);
    const retried = await saveAndCapture(page, dialog);
    expect(retried).toEqual({ body: expectedBody, status: 200 });
    await expect(page.getByText("Customer updated")).toBeVisible();
    await expect(dialog).toBeHidden();
  });

  test("cancel discards the draft without sending anything", async ({ page, customer }) => {
    const updates: Request[] = [];
    page.on("request", (req) => {
      if (isCustomerUpdate(req)) updates.push(req);
    });
    await openCustomer(page, customer, ORIGINAL_ALIAS);
    const dialog = await openEditDialog(page);

    await aliasInput(dialog).fill("Discarded Draft");
    await budgetInput(dialog).fill("99");
    await dialog.getByRole("button", { name: "Cancel", exact: true }).click();
    await expect(dialog).toBeHidden();

    const reopened = await openEditDialog(page);
    await expect(aliasInput(reopened)).toHaveValue(ORIGINAL_ALIAS);
    await expect(budgetInput(reopened)).toHaveValue(String(ORIGINAL_BUDGET));
    expect(updates).toHaveLength(0);
  });

  test.describe("Mobile", () => {
    test.use({ viewport: { width: 375, height: 812 }, isMobile: true, hasTouch: true });

    test("the Chinese dialog saves a non-ASCII alias on mobile", async ({ page, customer }) => {
      await page.evaluate(() => localStorage.setItem("litellm_language", "zh-CN"));
      await openCustomer(page, customer, ORIGINAL_ALIAS);
      await page.getByRole("button", { name: "编辑", exact: true }).tap();

      const dialog = page.getByRole("dialog", { name: "编辑客户", exact: true });
      const alias = dialog.getByLabel("别名", { exact: true });
      await expect(alias).toHaveValue(ORIGINAL_ALIAS);
      await expect(dialog.getByLabel("预算上限（USD）", { exact: true })).toHaveValue(String(ORIGINAL_BUDGET));
      await alias.fill("测试客户");

      const saved = await saveAndCapture(page, dialog, "保存");
      expect(saved).toEqual({
        body: { user_id: customer.id, alias: "测试客户", blocked: false, max_budget: ORIGINAL_BUDGET },
        status: 200,
      });
      await expect(dialog).toBeHidden();
      await expect(page.getByRole("heading", { level: 1, name: "测试客户" })).toBeVisible();
    });
  });
});
