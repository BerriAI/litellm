import { test, expect, type APIRequestContext, type Locator, type Page as PlaywrightPage } from "@playwright/test";
import { ADMIN_STORAGE_PATH } from "../../constants";
import { Page } from "../../fixtures/pages";
import { navigateToPage } from "../../helpers/navigation";
import { proxyIsPremium } from "../../helpers/premium";
import { CHAT_MODEL_A, deleteVirtualKey, masterKey, sendChatCompletion, waitForSpendLog } from "../../helpers/traffic";

interface GeneratedKeyResponse {
  key: string;
  token?: string;
  token_id?: string;
}

interface GeneratedKey {
  key: string;
  token: string;
}

interface CreatedTeam {
  team_id: string;
}

interface SpendLog {
  request_id: string;
}

const uniqueSuffix = (): string => `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;

const adminHeaders = (): Record<string, string> => ({ Authorization: `Bearer ${masterKey()}` });

const visibleTestId = (page: PlaywrightPage, id: string): Locator => page.getByTestId(id).filter({ visible: true });

const visibleDataTable = (page: PlaywrightPage): Locator => visibleTestId(page, "data-table-root").first();

const visibleRows = (page: PlaywrightPage): Locator => visibleDataTable(page).locator("tbody tr");

const rowWithText = (page: PlaywrightPage, text: string): Locator => visibleRows(page).filter({ hasText: text }).first();

async function postOk<T>(request: APIRequestContext, path: string, data: Record<string, unknown>): Promise<T> {
  const response = await request.post(path, { headers: adminHeaders(), data });
  expect(response.ok(), `POST ${path} failed (${response.status()}): ${await response.text()}`).toBe(true);
  return (await response.json()) as T;
}

async function createKey(
  request: APIRequestContext,
  alias: string,
  teamId?: string,
): Promise<GeneratedKey> {
  const response = await postOk<GeneratedKeyResponse>(request, "/key/generate", {
    key_alias: alias,
    ...(teamId ? { team_id: teamId } : {}),
  });
  const token = response.token ?? response.token_id;
  if (!token) {
    throw new Error("POST /key/generate did not return a key token");
  }
  return { key: response.key, token };
}

async function cellForHeader(page: PlaywrightPage, row: Locator, title: string): Promise<Locator> {
  const headers = await visibleDataTable(page).getByRole("columnheader").allTextContents();
  const index = headers.findIndex((header) => header.includes(title));
  if (index < 0) {
    throw new Error(`Could not find the ${title} table column`);
  }
  return row.getByRole("cell").nth(index);
}

async function createAuditTeam(request: APIRequestContext): Promise<CreatedTeam> {
  return postOk<CreatedTeam>(request, "/team/new", { team_alias: `e2e-row-audit-${uniqueSuffix()}` });
}

async function locateAuditRow(page: PlaywrightPage, teamId: string): Promise<Locator> {
  await navigateToPage(page, Page.Logs);
  await page.getByRole("tab", { name: "Audit Logs" }).click();
  await page.getByPlaceholder("Search audit logs by ID…").fill(teamId);
  const row = rowWithText(page, teamId);
  await expect(row, `audit row for team ${teamId} should be visible before activation checks`).toBeVisible();
  return row;
}

async function locateRequestRow(page: PlaywrightPage, requestId: string): Promise<Locator> {
  await navigateToPage(page, Page.Logs);
  await page.getByPlaceholder("Search logs by ID…").fill(requestId);
  const row = rowWithText(page, requestId);
  await expect(row, `request row ${requestId} should be visible before activation checks`).toBeVisible();
  return row;
}

async function readSpendLogRequestId(request: APIRequestContext, requestId: string): Promise<string> {
  const response = await request.get(`/spend/logs?request_id=${encodeURIComponent(requestId)}`, {
    headers: adminHeaders(),
  });
  expect(response.ok(), `GET /spend/logs for ${requestId} failed (${response.status()})`).toBe(true);
  const body = (await response.json()) as SpendLog[] | { data: SpendLog[] };
  const rows = Array.isArray(body) ? body : body.data;
  const spendLog = rows.find((row) => row.request_id === requestId);
  if (!spendLog) {
    throw new Error(`Spend log for request ${requestId} was not returned after waitForSpendLog`);
  }
  return spendLog.request_id;
}

test.describe("Virtual Keys row activation", () => {
  test.use({ storageState: ADMIN_STORAGE_PATH });

  test("a non-interactive cell opens that key's details", async ({ page, request }) => {
    const alias = `e2e-row-key-cell-${uniqueSuffix()}`;
    const key = await createKey(request, alias);
    try {
      await navigateToPage(page, Page.ApiKeys);
      const row = rowWithText(page, alias);
      await expect(row, `key row ${alias} should be visible before clicking Created At`).toBeVisible();

      await (await cellForHeader(page, row, "Created At")).click();

      await expect(page).toHaveURL((url) => url.searchParams.get("key") === key.token);
      await expect(page.getByText("Back to Keys", { exact: true })).toBeVisible();
    } finally {
      await deleteVirtualKey(request, key.token);
    }
  });

  test("the row is focusable, named, and opens details with Enter and Space", async ({ page, request }) => {
    const alias = `e2e-row-key-keyboard-${uniqueSuffix()}`;
    const key = await createKey(request, alias);
    try {
      await navigateToPage(page, Page.ApiKeys);
      const row = rowWithText(page, alias);
      await expect(row, `key row ${alias} should be visible before keyboard checks`).toBeVisible();
      await expect.soft(row).toHaveAccessibleName(`Open key ${alias}`);

      await row.focus();
      await expect.soft(row).toBeFocused();
      await page.keyboard.press("Enter");
      await expect.soft(page).toHaveURL((url) => url.searchParams.get("key") === key.token);
      await expect.soft(page.getByText("Back to Keys", { exact: true })).toBeVisible();

      const backToKeys = page.getByText("Back to Keys", { exact: true });
      if (await backToKeys.isVisible()) {
        await backToKeys.click();
      }
      await expect(row).toBeVisible();
      await row.focus();
      await page.keyboard.press("Space");
      await expect.soft(page).toHaveURL((url) => url.searchParams.get("key") === key.token);
      await expect.soft(page.getByText("Back to Keys", { exact: true })).toBeVisible();
    } finally {
      await deleteVirtualKey(request, key.token);
    }
  });

  test("the team link does not open key details", async ({ page, request }) => {
    const alias = `e2e-row-key-link-${uniqueSuffix()}`;
    const key = await createKey(request, alias, "e2e-team-crud");
    try {
      await navigateToPage(page, Page.ApiKeys);
      const row = rowWithText(page, alias);
      await expect(row, `key row ${alias} should be visible before checking its team link`).toBeVisible();
      const teamLink = row.getByRole("link", { name: "E2E Team CRUD" });
      await expect(teamLink).toBeVisible();
      await teamLink.click();

      await expect(page).not.toHaveURL((url) => url.searchParams.has("key"));
      await expect(page.getByText("Back to Keys", { exact: true })).not.toBeVisible();
    } finally {
      await deleteVirtualKey(request, key.token);
    }
  });
});

test.describe("Request Logs row activation", () => {
  test.use({ storageState: ADMIN_STORAGE_PATH });

  test("clicking a non-interactive cell opens the single-request drawer", async ({ page, request }) => {
    const suffix = uniqueSuffix();
    const key = await createKey(request, `e2e-row-log-mouse-${suffix}`);
    try {
      const requestId = await sendChatCompletion(request, {
        model: CHAT_MODEL_A,
        prompt: `row activation mouse ${suffix}`,
        apiKey: key.key,
      });
      await waitForSpendLog(request, requestId);
      const spendLogRequestId = await readSpendLogRequestId(request, requestId);
      const row = await locateRequestRow(page, spendLogRequestId);
      await (await cellForHeader(page, row, "Time")).click();

      await expect(page.getByRole("dialog").filter({ hasText: spendLogRequestId })).toBeVisible();
    } finally {
      await deleteVirtualKey(request, key.token);
    }
  });

  test("the row is focusable, named, and opens the drawer with Enter", async ({ page, request }) => {
    const suffix = uniqueSuffix();
    const key = await createKey(request, `e2e-row-log-keyboard-${suffix}`);
    try {
      const requestId = await sendChatCompletion(request, {
        model: CHAT_MODEL_A,
        prompt: `row activation keyboard ${suffix}`,
        apiKey: key.key,
      });
      await waitForSpendLog(request, requestId);
      const spendLogRequestId = await readSpendLogRequestId(request, requestId);
      const row = await locateRequestRow(page, spendLogRequestId);
      const drawer = page.getByRole("dialog").filter({ hasText: spendLogRequestId });

      await expect.soft(row).toHaveAccessibleName(`Open request ${spendLogRequestId}`);
      await row.focus();
      await expect.soft(row).toBeFocused();
      await page.keyboard.press("Enter");
      await expect.soft(drawer).toBeVisible();
    } finally {
      await deleteVirtualKey(request, key.token);
    }
  });
});

test.describe("Audit Logs row activation", () => {
  test.use({ storageState: ADMIN_STORAGE_PATH });

  test("clicking Timestamp opens that audit log", async ({ page, request }) => {
    test.skip(!proxyIsPremium(), "Audit Logs require a premium proxy license");
    const team = await createAuditTeam(request);
    try {
      const row = await locateAuditRow(page, team.team_id);
      await (await cellForHeader(page, row, "Timestamp")).click();

      await expect(page.getByRole("dialog").filter({ hasText: team.team_id })).toBeVisible();
    } finally {
      await request.post("/team/delete", { headers: adminHeaders(), data: { team_ids: [team.team_id] } });
    }
  });

  test("the row is focusable, named, and opens the drawer with Enter", async ({ page, request }) => {
    test.skip(!proxyIsPremium(), "Audit Logs require a premium proxy license");
    const team = await createAuditTeam(request);
    try {
      const row = await locateAuditRow(page, team.team_id);
      const action = (await row.getByRole("cell").nth(1).innerText()).trim();
      const drawer = page.getByRole("dialog").filter({ hasText: team.team_id });

      await expect.soft(row).toHaveAccessibleName(`Open audit log ${action} ${team.team_id}`);
      await row.focus();
      await expect.soft(row).toBeFocused();
      await page.keyboard.press("Enter");
      await expect.soft(drawer).toBeVisible();
    } finally {
      await request.post("/team/delete", { headers: adminHeaders(), data: { team_ids: [team.team_id] } });
    }
  });
});
