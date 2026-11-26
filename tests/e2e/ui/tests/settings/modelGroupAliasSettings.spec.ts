import {
  expect,
  test,
  type APIRequestContext,
  type Page,
} from "@playwright/test";
import { ADMIN_STORAGE_PATH } from "../../constants";
import { masterKey, rootPath, uniqueSuffix } from "../../helpers/traffic";

const CALLBACKS = "/get/config/callbacks";
const UPDATE = "/config/update";
const ALIAS_PREFIX = "e2e-alias-";
type AliasMap = Record<string, string>;
type StoredAliasMap = Record<string, string | { model: string }>;

const configResponse = (page: Page, method: string, path: string) =>
  page.waitForResponse(
    (response) =>
      response.request().method() === method &&
      new URL(response.url()).pathname === rootPath() + path,
  );

async function removeTestAliases(request: APIRequestContext, names: string[]) {
  const warn = (detail: string) =>
    test.info().annotations.push({
      type: "cleanup-warning",
      description: `Remove test aliases ${names.join(", ")}: ${detail}`,
    });
  try {
    const options = {
      headers: { Authorization: `Bearer ${masterKey()}`, Connection: "close" },
      timeout: 1500,
    };
    const response = await request.get(rootPath() + CALLBACKS, options);
    if (!response.ok()) {
      warn(`GET returned HTTP ${response.status()}`);
      return;
    }
    const aliases: StoredAliasMap =
      (await response.json()).router_settings?.model_group_alias ?? {};
    if (!names.some((name) => Object.hasOwn(aliases, name))) {
      warn("Test aliases were absent in the cleanup read");
      return;
    }
    const remaining = Object.fromEntries(
      Object.entries(aliases).filter(([name]) => !names.includes(name)),
    );
    const cleanup = await request.post(rootPath() + UPDATE, {
      ...options,
      data: { router_settings: { model_group_alias: remaining } },
    });
    if (!cleanup.ok()) warn(`POST returned HTTP ${cleanup.status()}`);
  } catch (error) {
    warn(
      String(error)
        .split("\n", 1)[0]
        .replace(/Bearer \S+/gi, "[REDACTED]"),
    );
  }
}

function assertAliasMap(
  body: { router_settings: { model_group_alias: AliasMap } },
  existing: AliasMap,
  name: string,
  target: string,
) {
  const aliasMap = body.router_settings.model_group_alias;
  expect(
    Object.fromEntries(
      Object.entries(aliasMap).filter(
        ([key]) => !key.startsWith(ALIAS_PREFIX) || key === name,
      ),
    ),
  ).toEqual({ ...existing, [name]: target });
}

test.describe.configure({ mode: "serial" });
test.use({ storageState: ADMIN_STORAGE_PATH });

test("adds an alias and renames it without changing the target model group", async ({
  page,
  request,
}) => {
  const models = await request.get(`${rootPath()}/v1/models`, {
    headers: { Authorization: `Bearer ${masterKey()}` },
  });
  expect(models.ok(), `Load available models (HTTP ${models.status()})`).toBe(
    true,
  );
  const modelList: { id: string }[] = (await models.json()).data;
  const target = modelList.find(({ id }) => !id.startsWith(ALIAS_PREFIX))?.id;
  if (!target) throw new Error("No model is available for the alias test");
  const name = `${ALIAS_PREFIX}${uniqueSuffix()}`;
  const renamed = `${name}-renamed`;
  const loading = configResponse(page, "GET", CALLBACKS);
  await page.goto(
    `${rootPath()}/ui/models-and-endpoints/?tab=model-group-alias`,
  );
  const initial = await loading;
  expect(initial.ok(), `Load aliases (HTTP ${initial.status()})`).toBe(true);
  const aliases: StoredAliasMap =
    (await initial.json()).router_settings?.model_group_alias ?? {};
  const existing = Object.fromEntries(
    Object.entries(aliases)
      .filter(([key]) => !key.startsWith(ALIAS_PREFIX))
      .map(([key, value]) => [
        key,
        typeof value === "string" ? value : value.model,
      ]),
  );
  const panel = page.getByRole("tabpanel", { name: "Model Group Alias" });
  await panel.getByPlaceholder("e.g., gpt-4o", { exact: true }).fill(name);
  await panel
    .getByPlaceholder("e.g., gpt-4o-mini-openai", { exact: true })
    .fill(target);

  const adding = configResponse(page, "POST", UPDATE);
  await panel.getByRole("button", { name: "Add Alias", exact: true }).click();
  const added = await adding;
  try {
    expect(added.ok(), `Add alias (HTTP ${added.status()})`).toBe(true);
    assertAliasMap(added.request().postDataJSON(), existing, name, target);
    const row = panel
      .getByRole("row")
      .filter({ has: page.getByRole("cell", { name, exact: true }) });
    await expect(
      row.getByRole("cell", { name: target, exact: true }),
    ).toBeVisible();
    await row.getByRole("button", { name: `Edit alias ${name}` }).click();
    const editing = panel
      .getByRole("row")
      .filter({ has: page.getByRole("button", { name: "Save", exact: true }) });
    await editing.getByRole("textbox").first().fill(renamed);

    const saving = configResponse(page, "POST", UPDATE);
    await editing.getByRole("button", { name: "Save", exact: true }).click();
    const saved = await saving;
    expect(saved.ok(), `Rename alias (HTTP ${saved.status()})`).toBe(true);
    const savedBody = saved.request().postDataJSON();
    assertAliasMap(savedBody, existing, renamed, target);
    expect(savedBody.router_settings.model_group_alias).not.toHaveProperty(
      name,
    );
    const renamedRow = panel
      .getByRole("row")
      .filter({ has: page.getByRole("cell", { name: renamed, exact: true }) });
    await expect(
      renamedRow.getByRole("cell", { name: target, exact: true }),
    ).toBeVisible();
    await expect(row).toBeHidden();
  } finally {
    if (added.ok()) await removeTestAliases(request, [name, renamed]);
  }
});
