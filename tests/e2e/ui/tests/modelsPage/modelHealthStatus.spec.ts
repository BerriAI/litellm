import {
  test as base,
  expect,
  type Locator,
  type Page as PlaywrightPage,
} from "@playwright/test";
import { ADMIN_STORAGE_PATH } from "../../constants";
import { Page } from "../../fixtures/pages";
import { navigateToPage } from "../../helpers/navigation";
import { readBack } from "../../helpers/roundTrip";
import { masterKey } from "../../helpers/traffic";

const MOCK_LLM_BASE = `http://127.0.0.1:${process.env.MOCK_LLM_PORT ?? "8090"}/v1`;
const UNREACHABLE_BASE = "http://127.0.0.1:9/v1";

async function isRegistered(
  page: PlaywrightPage,
  modelName: string,
): Promise<boolean> {
  const body = await readBack<{ data: { model_name?: string }[] }>(
    page,
    "/v2/model/info",
  );
  return body.data.some((row) => row.model_name === modelName);
}

function healthRow(page: PlaywrightPage, modelName: string): Locator {
  return page.getByRole("row").filter({ hasText: modelName });
}

function pageOf(label: string): { current: number; total: number } {
  const [current, total] = label
    .replace("Page ", "")
    .split(" of ")
    .map((part) => Number(part.trim()));
  return { current, total };
}

async function locateHealthRow(
  page: PlaywrightPage,
  modelName: string,
): Promise<Locator> {
  const pageLabel = page.getByTestId("pagination-page");
  await expect(
    pageLabel,
    "the health table reports which page it is showing",
  ).toBeVisible({ timeout: 20_000 });

  const deadline = Date.now() + 60_000;
  while (Date.now() < deadline) {
    const row = healthRow(page, modelName);
    const onThisPage = await row
      .first()
      .waitFor({ state: "visible", timeout: 3_000 })
      .then(() => true)
      .catch(() => false);
    if (onThisPage) return row;

    const { current, total } = pageOf(await pageLabel.innerText());
    const goTo = current < total ? current + 1 : 1;
    if (total === 1) continue;
    await page
      .getByRole("button", {
        name: current < total ? "Go to next page" : "Go to first page",
      })
      .click();
    await expect(pageLabel).toContainText(`Page ${goTo} of`, {
      timeout: 15_000,
    });
  }
  return healthRow(page, modelName);
}

async function openHealthTab(page: PlaywrightPage): Promise<void> {
  await page.getByRole("tab", { name: "Health Status" }).click();
  await expect(
    page.getByRole("heading", { name: "Model Health Status" }),
  ).toBeVisible({ timeout: 15_000 });
}

async function expectStatus(
  page: PlaywrightPage,
  modelName: string,
  status: string,
): Promise<void> {
  const row = await locateHealthRow(page, modelName);
  await expect(row, `${modelName} has one row in the health table`).toHaveCount(
    1,
    { timeout: 20_000 },
  );
  await expect(
    row.getByText(status, { exact: true }),
    `the Health Status cell for ${modelName} reads ${status}`,
  ).toHaveCount(1, { timeout: 60_000 });
}

async function deleteDeployment(
  page: PlaywrightPage,
  id: string,
): Promise<void> {
  const post = () =>
    page.request.post("/model/delete", {
      headers: { Authorization: `Bearer ${masterKey()}` },
      data: { id },
    });
  const deleted = await post().catch(() => post());
  expect(
    deleted.ok(),
    `cleanup: /model/delete ${id} returned ${deleted.status()}`,
  ).toBe(true);
}

const uniqueSuffix = (): string =>
  `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;

async function withDeployment(
  page: PlaywrightPage,
  prefix: string,
  apiBase: string,
  use: (name: string) => Promise<void>,
): Promise<void> {
  const name = `${prefix}-${uniqueSuffix()}`;
  const created = await page.request.post("/model/new", {
    headers: { Authorization: `Bearer ${masterKey()}` },
    data: {
      model_name: name,
      litellm_params: {
        model: `openai/${name}`,
        api_base: apiBase,
        api_key: "fake-key",
      },
      model_info: {},
    },
  });
  expect(
    created.ok(),
    `/model/new for ${name} failed: ${created.status()} ${await created.text()}`,
  ).toBe(true);
  const id = (await created.json()).model_info?.id;
  expect(id, `model id from /model/new for ${name}`).toBeTruthy();
  try {
    await expect
      .poll(() => isRegistered(page, name), {
        message: `deployment ${name} never appeared in /v2/model/info after create`,
        timeout: 60_000,
      })
      .toBe(true);
    await use(name);
  } finally {
    await deleteDeployment(page, id);
  }
}

const test = base.extend<{ reachableName: string; unreachableName: string }>({
  reachableName: async ({ page }, use) => {
    await withDeployment(page, "e2e-health-up", MOCK_LLM_BASE, use);
  },
  unreachableName: async ({ page }, use) => {
    await withDeployment(page, "e2e-health-down", UNREACHABLE_BASE, use);
  },
});

test.describe("Model health status", () => {
  test.use({ storageState: ADMIN_STORAGE_PATH });

  test("Run Health Check reports a reachable deployment healthy and an unreachable one unhealthy", async ({
    page,
    reachableName,
    unreachableName,
  }) => {
    await navigateToPage(page, Page.Models);
    await openHealthTab(page);

    for (const name of [reachableName, unreachableName]) {
      const row = await locateHealthRow(page, name);
      await expect(row, `${name} has one row in the health table`).toHaveCount(
        1,
        { timeout: 20_000 },
      );
      await row
        .getByRole("button", { name: "Run Health Check", exact: true })
        .click();
    }

    await expectStatus(page, reachableName, "healthy");
    await expect(
      healthRow(page, reachableName).getByText("unhealthy", { exact: true }),
      "a reachable deployment is never reported unhealthy",
    ).toHaveCount(0);
    await expectStatus(page, unreachableName, "unhealthy");

    const successDetail = (
      await locateHealthRow(page, reachableName)
    ).getByRole("button", {
      name: "View response details",
    });
    await expect(
      successDetail,
      `${reachableName} offers its health check response for inspection`,
    ).toBeVisible({ timeout: 60_000 });
    await successDetail.click();
    const successDialog = page.getByRole("dialog");
    await expect(
      successDialog.getByRole("heading", {
        name: `Health Check Response - ${reachableName}`,
      }),
      "the healthy deployment's detail opens its own response dialog",
    ).toBeVisible({ timeout: 10_000 });
    await successDialog.getByRole("button", { name: "Close" }).last().click();
    await expect(successDialog).toBeHidden({ timeout: 10_000 });

    const errorDetail = (
      await locateHealthRow(page, unreachableName)
    ).getByRole("button", {
      name: "View full error details",
    });
    await expect(
      errorDetail,
      `${unreachableName} offers its health check error for inspection`,
    ).toBeVisible({ timeout: 60_000 });
    await errorDetail.click();
    const errorDialog = page.getByRole("dialog");
    await expect(
      errorDialog.getByRole("heading", {
        name: `Health Check Error - ${unreachableName}`,
      }),
      "the unreachable deployment's detail opens its own error dialog",
    ).toBeVisible({ timeout: 10_000 });
    await expect(
      errorDialog,
      "the error dialog carries the upstream connection failure, not a generic message",
    ).toContainText(/connection error/i, { timeout: 10_000 });
    await expect(
      errorDialog,
      "the error dialog names the endpoint that could not be reached",
    ).toContainText(UNREACHABLE_BASE);
    await errorDialog.getByRole("button", { name: "Close" }).last().click();
    await expect(errorDialog).toBeHidden({ timeout: 10_000 });

    await page.reload();
    await openHealthTab(page);
    await expectStatus(page, reachableName, "healthy");
    await expectStatus(page, unreachableName, "unhealthy");
  });
});
