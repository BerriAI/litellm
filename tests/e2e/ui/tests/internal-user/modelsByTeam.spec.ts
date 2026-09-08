import {
  test as base,
  expect,
  type Locator,
  type Page as PlaywrightPage,
} from "@playwright/test";
import {
  E2E_TEAM_CRUD_ALIAS,
  E2E_TEAM_ORG_ALIAS,
  INTERNAL_USER_STORAGE_PATH,
} from "../../constants";
import { Page } from "../../fixtures/pages";
import { navigateToPage } from "../../helpers/navigation";
import { readBack } from "../../helpers/roundTrip";
import { CHAT_MODEL_A, CHAT_MODEL_B, masterKey } from "../../helpers/traffic";

const MOCK_LLM_BASE = `http://127.0.0.1:${process.env.MOCK_LLM_PORT ?? "8090"}/v1`;
const CURRENT_TEAM_VIEW = "Current Team Models";
const ALL_MODELS_VIEW = "All Available Models";
const PERSONAL_TEAM = "Personal";

const teamSelector = (page: PlaywrightPage): Locator =>
  page.getByRole("combobox", { name: "Current team", exact: true });
const viewSelector = (page: PlaywrightPage): Locator =>
  page.getByRole("combobox", { name: "View", exact: true });

async function chooseOption(
  page: PlaywrightPage,
  selector: Locator,
  optionName: string,
): Promise<void> {
  await selector.click();
  const option = page.getByRole("option", { name: optionName, exact: true });
  await expect(option, `option ${optionName} is offered`).toBeVisible({
    timeout: 10_000,
  });
  await option.click();
  await expect(
    selector,
    `${optionName} is the selection the control now reports`,
  ).toContainText(optionName, {
    timeout: 10_000,
  });
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

function modelRow(page: PlaywrightPage, modelName: string): Locator {
  return page.getByRole("row").filter({ hasText: modelName });
}

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

const uniqueSuffix = (): string =>
  `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;

const test = base.extend<{ ungrantedModelName: string }>({
  ungrantedModelName: async ({ page }, use) => {
    const ungrantedModelName = `e2e-ungranted-${uniqueSuffix()}`;
    const created = await page.request.post("/model/new", {
      headers: { Authorization: `Bearer ${masterKey()}` },
      data: {
        model_name: ungrantedModelName,
        litellm_params: {
          model: `openai/${ungrantedModelName}`,
          api_base: MOCK_LLM_BASE,
          api_key: "fake-key",
        },
        model_info: {},
      },
    });
    expect(
      created.ok(),
      `/model/new failed: ${created.status()} ${await created.text()}`,
    ).toBe(true);
    const ungrantedModelId = (await created.json()).model_info?.id;
    expect(ungrantedModelId, "model id from /model/new").toBeTruthy();

    try {
      await expect
        .poll(async () => await isRegistered(page, ungrantedModelName), {
          message: `deployment ${ungrantedModelName} never appeared in /v2/model/info after create`,
          timeout: 60_000,
        })
        .toBe(true);
      await use(ungrantedModelName);
    } finally {
      await deleteDeployment(page, ungrantedModelId);
    }
  },
});

test.describe("Models and Endpoints for an internal user", () => {
  test.use({ storageState: INTERNAL_USER_STORAGE_PATH });

  test("shows an internal user exactly the models of the team they select", async ({
    page,
    ungrantedModelName,
  }) => {
    await navigateToPage(page, Page.Models);

    await expect(
      page.getByRole("tab", { name: "Your Models" }),
      "an internal user lands on their own models tab, not an admin-only view",
    ).toBeVisible({ timeout: 15_000 });
    await expect(
      viewSelector(page),
      "the models table opens scoped to the selected team",
    ).toContainText(CURRENT_TEAM_VIEW, { timeout: 15_000 });
    await expect(
      modelRow(page, ungrantedModelName),
      `the personal view lists ${ungrantedModelName}, so it is on the proxy and reachable from this page`,
    ).toHaveCount(1, { timeout: 30_000 });

    await chooseOption(page, teamSelector(page), E2E_TEAM_CRUD_ALIAS);
    await expect(
      modelRow(page, CHAT_MODEL_A),
      `${E2E_TEAM_CRUD_ALIAS} lists ${CHAT_MODEL_A}`,
    ).toHaveCount(1, {
      timeout: 15_000,
    });
    await expect(
      modelRow(page, CHAT_MODEL_B),
      `${E2E_TEAM_CRUD_ALIAS} lists ${CHAT_MODEL_B}`,
    ).toHaveCount(1, {
      timeout: 15_000,
    });
    await expect(
      modelRow(page, ungrantedModelName),
      `${ungrantedModelName} is on the proxy but not granted to ${E2E_TEAM_CRUD_ALIAS}, so it must not be listed`,
    ).toHaveCount(0);

    await chooseOption(page, teamSelector(page), E2E_TEAM_ORG_ALIAS);
    await expect(
      modelRow(page, CHAT_MODEL_A),
      `${E2E_TEAM_ORG_ALIAS} lists ${CHAT_MODEL_A}`,
    ).toHaveCount(1, {
      timeout: 15_000,
    });
    await expect(
      page.getByTestId("pagination-range"),
      `${E2E_TEAM_ORG_ALIAS} lists the one model it grants and nothing else`,
    ).toHaveText("Showing 1-1 of 1", { timeout: 15_000 });
    await expect(
      modelRow(page, CHAT_MODEL_B),
      `${CHAT_MODEL_B} belongs to another team and must not leak into ${E2E_TEAM_ORG_ALIAS}`,
    ).toHaveCount(0);
    await expect(
      modelRow(page, ungrantedModelName),
      `${ungrantedModelName} is granted to no team and must not leak into ${E2E_TEAM_ORG_ALIAS}`,
    ).toHaveCount(0);

    await chooseOption(page, viewSelector(page), ALL_MODELS_VIEW);
    await expect(
      modelRow(page, CHAT_MODEL_A),
      `switching to ${ALL_MODELS_VIEW} leaves the table populated rather than blanking it`,
    ).toHaveCount(1, { timeout: 15_000 });

    await page.reload();
    await expect(
      teamSelector(page),
      "the team selection is not persisted across a reload, so the table returns to the personal view",
    ).toContainText(PERSONAL_TEAM, { timeout: 15_000 });
    await expect(
      viewSelector(page),
      "the view selection is not persisted across a reload either",
    ).toContainText(CURRENT_TEAM_VIEW, { timeout: 15_000 });
    await expect(
      modelRow(page, ungrantedModelName),
      "the personal view still renders models after a reload rather than coming back empty",
    ).toHaveCount(1, { timeout: 30_000 });
  });
});
