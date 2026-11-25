import {
  test as base,
  expect,
  type APIRequestContext,
  type Locator,
  type Page as PlaywrightPage,
} from "@playwright/test";
import { ADMIN_STORAGE_PATH } from "../../constants";
import { Page } from "../../fixtures/pages";
import { navigateToPage } from "../../helpers/navigation";
import { masterKey, rootPath, uniqueSuffix } from "../../helpers/traffic";

const MOCK_LLM_BASE = `http://127.0.0.1:${process.env.MOCK_LLM_PORT ?? "8090"}/v1`;

const adminHeaders = () => ({
  Authorization: `Bearer ${masterKey()}`,
  "Content-Type": "application/json",
});

const isListedAsDecisionsModel = async (
  request: APIRequestContext,
  name: string,
): Promise<boolean> => {
  const listed = await request.get("/model_group/info", {
    headers: adminHeaders(),
  });
  if (!listed.ok()) return false;
  const body: { data?: { model_group?: string; mode?: string }[] } =
    await listed.json();
  return (body.data ?? []).some(
    (group) => group.model_group === name && group.mode === "evaluation",
  );
};

const isRoutable = async (
  request: APIRequestContext,
  name: string,
): Promise<boolean> => {
  const ping = await request.post(`${rootPath()}/v1/systemone`, {
    headers: adminHeaders(),
    data: {
      model: name,
      state: "readiness ping",
      questions: {
        ready: {
          type: "noul",
          instructions: "Is this a ping?",
          criteria: { true: "yes", false: "no" },
        },
      },
    },
  });
  return ping.ok();
};

const test = base.extend<{ decisionsModel: string }>({
  decisionsModel: async ({ request }, use) => {
    const name = `e2e-decisions-header-${uniqueSuffix()}`;
    const created = await request.post("/model/new", {
      headers: adminHeaders(),
      data: {
        model_name: name,
        litellm_params: {
          model: "typesafe/jev-latest",
          api_base: MOCK_LLM_BASE,
          api_key: "fake-key",
        },
      },
    });
    expect(
      created.ok(),
      `POST /model/new failed (${created.status()}): ${await created.text()}`,
    ).toBe(true);
    const modelId: string = (await created.json()).model_id;
    try {
      await expect
        .poll(
          async () =>
            (await isListedAsDecisionsModel(request, name)) &&
            (await isRoutable(request, name)),
          {
            message: `${name} never became listed and routable`,
            timeout: 60_000,
          },
        )
        .toBe(true);
      await use(name);
    } finally {
      await request.post("/model/delete", {
        headers: adminHeaders(),
        data: { id: modelId },
      });
    }
  },
});

const centerY = async (locator: Locator): Promise<number> => {
  const box = await locator.boundingBox();
  expect(box, "element has no layout box").not.toBeNull();
  return box!.y + box!.height / 2;
};

const offCenterBy = async (inner: Locator, outer: Locator): Promise<number> =>
  Math.abs((await centerY(inner)) - (await centerY(outer)));

const dismissSearchHint = async (page: PlaywrightPage): Promise<void> => {
  const dismiss = page.getByRole("button", { name: "Dismiss search hint" });
  if (await dismiss.isVisible()) {
    await dismiss.click();
  }
};

test.describe("Decisions playground Response header", () => {
  test.use({
    storageState: ADMIN_STORAGE_PATH,
    viewport: { width: 1440, height: 900 },
  });

  test("keeps the title and its badges on one row centered in the header", async ({
    page,
    decisionsModel,
  }) => {
    await navigateToPage(page, Page.LlmPlayground);
    await page.getByRole("tab", { name: /Decisions/ }).click();

    const header = page
      .getByRole("region", { name: "Decisions results" })
      .locator('[data-slot="card-header"]');
    const title = header.getByText("Response", { exact: true });
    await expect(title).toBeVisible({ timeout: 20_000 });
    expect(
      await offCenterBy(title, header),
      "empty-state title is off center",
    ).toBeLessThanOrEqual(1);

    await page.getByRole("button", { name: "Load example" }).click();
    await page
      .getByRole("menuitem", { name: "Moderate a forum comment" })
      .click();
    const modelPicker = page.getByRole("combobox", { name: "Decision model" });
    await modelPicker.click();
    await modelPicker.fill(decisionsModel);
    await page
      .getByRole("option", { name: decisionsModel, exact: true })
      .click();
    await dismissSearchHint(page);
    await page.getByRole("button", { name: "Send", exact: true }).click();

    const usage = header.getByText("12 in / 3 out");
    await expect(usage).toBeVisible({ timeout: 30_000 });
    expect(
      await offCenterBy(title, header),
      "title is off center after Send",
    ).toBeLessThanOrEqual(1);
    for (const badge of [header.getByText(/^\d+ ms$/), usage]) {
      expect(
        await offCenterBy(badge, title),
        "badge wrapped off the title's row",
      ).toBeLessThanOrEqual(1);
    }
  });
});
