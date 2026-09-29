import {
  test as base,
  expect,
  type Page as PlaywrightPage,
} from "@playwright/test";
import { ADMIN_STORAGE_PATH } from "../../constants";
import { Page } from "../../fixtures/pages";
import { navigateToPage } from "../../helpers/navigation";
import { captureRequestBody, readBack } from "../../helpers/roundTrip";
import { masterKey, sendChatCompletion } from "../../helpers/traffic";

const MOCK_LLM_BASE = `http://127.0.0.1:${process.env.MOCK_LLM_PORT ?? "8090"}/v1`;
const CUSTOM_PARAM = "extra_headers";
const CUSTOM_PARAM_VALUE = { "X-E2E-Edit-Probe": "one" };

type StoredParams = Record<string, unknown>;

async function readStoredParams(
  page: PlaywrightPage,
  modelId: string,
): Promise<StoredParams> {
  const body = await readBack<{ data: { litellm_params: StoredParams }[] }>(
    page,
    `/model/info?litellm_model_id=${modelId}`,
  );
  return body.data[0]?.litellm_params ?? {};
}

function paramsEditor(page: PlaywrightPage) {
  return page.getByPlaceholder('"rpm": 100');
}

async function editParams(
  page: PlaywrightPage,
  mutate: (params: StoredParams) => StoredParams,
): Promise<void> {
  await page.getByRole("button", { name: "Edit Settings" }).click();
  const editor = paramsEditor(page);
  await expect(
    editor,
    "the LiteLLM Params editor is reachable on every visit to the edit form",
  ).toBeVisible({
    timeout: 15_000,
  });
  const shown = JSON.parse(await editor.inputValue()) as StoredParams;
  await editor.fill(JSON.stringify(mutate(shown), null, 2));
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

const test = base.extend<{
  deployment: { readonly modelName: string; readonly createdModelId: string };
}>({
  deployment: async ({ page, request }, use) => {
    const modelName = `e2e-edit-params-${uniqueSuffix()}`;
    const created = await page.request.post("/model/new", {
      headers: { Authorization: `Bearer ${masterKey()}` },
      data: {
        model_name: modelName,
        litellm_params: {
          model: `openai/${modelName}`,
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
    const createdModelId = (await created.json()).model_info?.id;
    expect(createdModelId, "model id from /model/new").toBeTruthy();

    try {
      await expect
        .poll(
          async () => {
            try {
              await sendChatCompletion(request, {
                model: modelName,
                prompt: `warmup ${modelName}`,
              });
              return true;
            } catch {
              return false;
            }
          },
          {
            message: `deployment ${modelName} never became routable after /model/new`,
            timeout: 60_000,
          },
        )
        .toBe(true);
      await use({ modelName, createdModelId });
    } finally {
      await deleteDeployment(page, createdModelId);
    }
  },
});

test.describe("Edit LiteLLM Params on a deployment", () => {
  test.use({ storageState: ADMIN_STORAGE_PATH });

  test("params added on a deployment can be re-edited, and the deployment keeps serving", async ({
    page,
    request,
    deployment: { modelName, createdModelId },
  }) => {
    await navigateToPage(page, Page.Models);
    const modelIdCell = page.getByTestId(`model-id-${createdModelId}`);
    await expect(
      modelIdCell,
      `the Models table lists ${modelName}`,
    ).toBeVisible({ timeout: 15_000 });
    await modelIdCell.click();
    await expect(page.getByText("Back to Models").first()).toBeVisible({
      timeout: 15_000,
    });

    await editParams(page, (params) => ({
      ...params,
      temperature: 0.2,
      [CUSTOM_PARAM]: CUSTOM_PARAM_VALUE,
    }));
    const firstSave = await captureRequestBody(
      page,
      { method: "PATCH", urlIncludes: `/model/${createdModelId}/update` },
      async () => {
        await page.getByRole("button", { name: "Save Changes" }).click();
      },
    );
    expect(
      firstSave.litellm_params?.temperature,
      "the added temperature goes on the wire",
    ).toBe(0.2);
    expect(
      firstSave.litellm_params?.[CUSTOM_PARAM],
      `the added ${CUSTOM_PARAM} goes on the wire`,
    ).toEqual(CUSTOM_PARAM_VALUE);
    expect(
      firstSave.litellm_params?.model,
      "a params edit does not rewrite the upstream model",
    ).toBe(`openai/${modelName}`);
    expect(
      firstSave.litellm_params?.api_base,
      "a params edit does not rewrite the api base",
    ).toBe(MOCK_LLM_BASE);
    expect(
      firstSave.litellm_params,
      "the credential is never re-sent, so a masked placeholder cannot overwrite the stored key",
    ).not.toHaveProperty("api_key");

    await expect
      .poll(
        async () => (await readStoredParams(page, createdModelId)).temperature,
        {
          message: "the added temperature never reached the stored deployment",
          timeout: 20_000,
        },
      )
      .toBe(0.2);
    const afterFirstSave = await readStoredParams(page, createdModelId);
    expect(
      afterFirstSave[CUSTOM_PARAM],
      `the added ${CUSTOM_PARAM} reached the stored deployment`,
    ).toEqual(CUSTOM_PARAM_VALUE);
    expect(
      afterFirstSave.model,
      "the stored upstream model survived the edit",
    ).toBe(`openai/${modelName}`);
    expect(
      afterFirstSave.api_base,
      "the stored api base survived the edit",
    ).toBe(MOCK_LLM_BASE);

    await editParams(page, (params) => ({
      ...Object.fromEntries(
        Object.entries(params).filter(([key]) => key !== CUSTOM_PARAM),
      ),
      temperature: 0.7,
    }));
    const secondSave = await captureRequestBody(
      page,
      { method: "PATCH", urlIncludes: `/model/${createdModelId}/update` },
      async () => {
        await page.getByRole("button", { name: "Save Changes" }).click();
      },
    );
    expect(
      secondSave.litellm_params?.temperature,
      "a param set by an earlier save can be edited again",
    ).toBe(0.7);
    expect(
      secondSave.litellm_params,
      `dropping ${CUSTOM_PARAM} from the editor drops it from the request the UI sends`,
    ).not.toHaveProperty(CUSTOM_PARAM);
    expect(
      secondSave.litellm_params?.model,
      "a second params edit still leaves the upstream model alone",
    ).toBe(`openai/${modelName}`);
    expect(
      secondSave.litellm_params?.api_base,
      "a second params edit still leaves the api base alone",
    ).toBe(MOCK_LLM_BASE);
    expect(
      secondSave.litellm_params,
      "the credential is still never re-sent",
    ).not.toHaveProperty("api_key");

    await expect
      .poll(
        async () => (await readStoredParams(page, createdModelId)).temperature,
        {
          message:
            "the re-edited temperature never reached the stored deployment",
          timeout: 20_000,
        },
      )
      .toBe(0.7);

    await page.reload();
    await expect(
      page
        .getByRole("tabpanel", { name: "Overview" })
        .getByText('"temperature": 0.7'),
      "reopening the deployment renders the re-edited value, not the one from the first save",
    ).toBeVisible({ timeout: 20_000 });

    await sendChatCompletion(request, {
      model: modelName,
      prompt: `still serving ${modelName}`,
    });
  });
});
