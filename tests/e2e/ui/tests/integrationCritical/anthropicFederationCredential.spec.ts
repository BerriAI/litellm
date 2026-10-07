import {
  test,
  expect,
  type APIRequestContext,
  type Locator,
  type Page as PlaywrightPage,
} from "@playwright/test";
import { Page } from "../../fixtures/pages";
import { navigateToPage } from "../../helpers/navigation";
import { captureRequestBody, readBack } from "../../helpers/roundTrip";
import { uniqueSuffix } from "../../helpers/traffic";
import {
  logInThroughLoginPage,
  setInvitedUserPassword,
} from "../../helpers/userOnboarding";

const master = process.env.LITELLM_MASTER_KEY ?? "sk-integration-master";
const headers = { Authorization: `Bearer ${master}` };

const ANTHROPIC_LABEL = "Anthropic";
const OPENAI_LABEL = "OpenAI";
const FEDERATION_BADGE = "Workload identity federation";
const FEDERATION_BUTTON = "Use workload identity federation";
const FEDERATION_HELP =
  "Workload identity federation is saved as a credential, then attached to this model.";
const ADDED_TOAST = "Credential added successfully";
const UPDATED_TOAST = "Credential updated successfully";
const ALLOWLIST_VARIABLE = "LITELLM_OIDC_ALLOWED_CREDENTIAL_DIRS";
const NO_IDS_MESSAGE =
  "Enter at least one of these ids, or pick an identity source that stores a token. The proxy rejects a credential with no values";
const RAW_TOKEN_MESSAGE =
  "Enter an oidc/ secret reference such as oidc/env/VAR_NAME. Raw tokens and oidc/env_path/ references are not accepted";
const TTL_MESSAGE = "Enter a whole number of seconds from 1 to 3600";

type CredentialValues = Record<string, unknown>;

interface StoredCredential {
  credential_name: string;
  credential_values: CredentialValues;
  credential_info: { custom_llm_provider: string };
}

interface Deployment {
  model_name: string;
  litellm_params: Record<string, unknown>;
  model_info: { id: string };
}

interface CredentialWrite {
  credential_name: string;
  credential_values: CredentialValues;
  credential_info: { custom_llm_provider: string };
  credential_values_to_delete?: string[];
}

interface ModelWrite {
  model_name: string;
  litellm_params: Record<string, unknown>;
}

interface ModelCreated {
  model_id: string;
}

const masked = (value: string): string =>
  value.length <= 4 ? "*****" : `${value.slice(0, 2)}****${value.slice(-2)}`;

const exactText = (text: string): RegExp =>
  new RegExp(`^${text.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}$`);

async function logInAsAdmin(page: PlaywrightPage): Promise<void> {
  await page.goto("/ui/login");
  await page.getByPlaceholder("Enter your username").fill("admin");
  await page.getByPlaceholder("Enter your password").fill(master);
  await page.getByRole("button", { name: "Login", exact: true }).click();
  await expect(page).toHaveURL(
    (url) => url.pathname.startsWith("/ui") && !url.pathname.includes("login"),
  );
}

async function openModelsTab(page: PlaywrightPage, tab: string): Promise<void> {
  await navigateToPage(page, Page.Models);
  await page.getByRole("tab", { name: tab, exact: true }).click();
}

async function pickOption(
  page: PlaywrightPage,
  trigger: Locator,
  label: string,
): Promise<void> {
  await trigger.click();
  await page
    .getByRole("option")
    .filter({ hasText: exactText(label) })
    .first()
    .click();
}

async function pickProvider(
  page: PlaywrightPage,
  scope: Locator,
  provider: string,
): Promise<void> {
  await pickOption(page, scope.getByPlaceholder("Select a provider"), provider);
}

async function pickAuthMethod(
  page: PlaywrightPage,
  dialog: Locator,
  label: string,
): Promise<void> {
  await pickOption(page, dialog.locator("#anthropic_auth_method"), label);
}

async function pickIdentitySource(
  page: PlaywrightPage,
  dialog: Locator,
  label: string,
): Promise<void> {
  await pickOption(
    page,
    dialog.locator("#anthropic_federation_identity_source"),
    label,
  );
}

async function fillFields(
  dialog: Locator,
  values: Record<string, string>,
): Promise<void> {
  for (const [key, value] of Object.entries(values)) {
    await dialog.locator(`#${key}`).fill(value);
  }
}

async function openAddCredentialDialog(
  page: PlaywrightPage,
  name: string,
): Promise<Locator> {
  await page
    .getByRole("button", { name: "Add Credential", exact: true })
    .click();
  const dialog = page.getByRole("dialog", { name: "Add New Credential" });
  await expect(dialog).toBeVisible();
  await dialog
    .getByPlaceholder("Enter a friendly name for these credentials")
    .fill(name);
  await pickProvider(page, dialog, ANTHROPIC_LABEL);
  await pickAuthMethod(page, dialog, FEDERATION_BADGE);
  return dialog;
}

async function openEditDialog(
  page: PlaywrightPage,
  name: string,
): Promise<Locator> {
  await expect(page.getByText(UPDATED_TOAST)).toHaveCount(0);
  const row = page.locator("tr", { hasText: name });
  await expect(row).toBeVisible({ timeout: 15_000 });
  await row.getByTestId(`credential-actions-${name}`).click();
  await page.getByTestId("credential-action-edit").click();
  const dialog = page.getByRole("dialog", { name: "Edit Credential" });
  await expect(dialog).toBeVisible();
  return dialog;
}

const submitCredential = (
  page: PlaywrightPage,
  dialog: Locator,
  method: "POST" | "PATCH",
  urlIncludes: string,
  button: string,
): Promise<CredentialWrite> =>
  captureRequestBody<CredentialWrite>(page, { method, urlIncludes }, () =>
    dialog.getByRole("button", { name: button, exact: true }).click(),
  );

function countRequests(
  page: PlaywrightPage,
  method: string,
  pathSuffix: string,
): () => number {
  let seen = 0;
  page.on("request", (request) => {
    if (
      request.method() === method &&
      new URL(request.url()).pathname.endsWith(pathSuffix)
    ) {
      seen += 1;
    }
  });
  return () => seen;
}

async function expectBlocked(
  page: PlaywrightPage,
  dialog: Locator,
  button: string,
  message: string,
  postsBefore: number,
  posts: () => number,
): Promise<void> {
  await dialog.getByRole("button", { name: button, exact: true }).click();
  await expect(dialog.getByText(message)).toBeVisible();
  await expect(dialog).toBeVisible();
  expect(posts(), "a refused form must send nothing").toBe(postsBefore);
}

async function createCredential(
  request: APIRequestContext,
  name: string,
  values: CredentialValues,
  provider = "anthropic",
): Promise<void> {
  const response = await request.post("/credentials", {
    headers,
    data: {
      credential_name: name,
      credential_values: values,
      credential_info: { custom_llm_provider: provider },
    },
  });
  expect(response.status(), await response.text()).toBe(200);
}

async function deleteCredentials(
  request: APIRequestContext,
  names: readonly string[],
): Promise<void> {
  for (const name of names) {
    await request.delete(`/credentials/${name}`, { headers });
  }
}

const sameValues = (left: CredentialValues, right: CredentialValues): boolean =>
  JSON.stringify(Object.entries(left).sort()) ===
  JSON.stringify(Object.entries(right).sort());

async function readStoredCredential(
  request: APIRequestContext,
  name: string,
): Promise<StoredCredential | null> {
  const response = await request.get(`/credentials/by_name/${name}`, {
    headers,
  });
  return response.status() === 200
    ? ((await response.json()) as StoredCredential)
    : null;
}

async function expectStored(
  request: APIRequestContext,
  name: string,
  expected: CredentialValues,
): Promise<StoredCredential> {
  let last: StoredCredential | null = null;
  await expect
    .poll(
      async () => {
        last = await readStoredCredential(request, name);
        return last !== null && sameValues(last.credential_values, expected);
      },
      {
        timeout: 70_000,
        message: `the proxy must serve ${name} as ${JSON.stringify(expected)}; last read ${JSON.stringify(last)}`,
      },
    )
    .toBe(true);
  if (last === null) {
    throw new Error(`${name} was never read back`);
  }
  return last;
}

async function postAsMaster<T = Record<string, unknown>>(
  request: APIRequestContext,
  route: string,
  data: Record<string, unknown>,
): Promise<T> {
  const response = await request.post(route, { headers, data });
  expect(response.status(), `POST ${route}: ${await response.text()}`).toBe(
    200,
  );
  return (await response.json()) as T;
}

test("the Add Credential dialog stores each federation identity source with only the values it needs and refuses the invalid ones", async ({
  page,
  request,
}) => {
  const suffix = uniqueSuffix();
  const names = {
    tokenFile: `int-wif-file-${suffix}`,
    secretReference: `int-wif-ref-${suffix}`,
    internalIssuer: `int-wif-issuer-${suffix}`,
    environment: `int-wif-env-${suffix}`,
  };
  const ids = {
    anthropic_federation_rule_id: `fdrl_${suffix}`,
    anthropic_organization_id: `org-${suffix}`,
    anthropic_service_account_id: `svac_${suffix}`,
    anthropic_federation_workspace_id: `wrkspc_${suffix}`,
  };
  const tokenFile = `/var/run/secrets/anthropic/${suffix}/token`;
  const posts = countRequests(page, "POST", "/credentials");
  try {
    await logInAsAdmin(page);
    await openModelsTab(page, "LLM Credentials");

    const fileDialog = await openAddCredentialDialog(page, names.tokenFile);
    await pickIdentitySource(page, fileDialog, "Identity token file");
    await fillFields(fileDialog, {
      ...ids,
      anthropic_identity_token_file: tokenFile,
    });
    const created = await submitCredential(
      page,
      fileDialog,
      "POST",
      "/credentials",
      "Add Credential",
    );
    expect(created).toEqual({
      credential_name: names.tokenFile,
      credential_values: { ...ids, anthropic_identity_token_file: tokenFile },
      credential_info: { custom_llm_provider: ANTHROPIC_LABEL },
    });
    await expect(page.getByText(ADDED_TOAST)).toBeVisible();
    const row = page.locator("tr", { hasText: names.tokenFile });
    await expect(row).toBeVisible({ timeout: 15_000 });
    await expect(row).toContainText(FEDERATION_BADGE);
    await expectStored(request, names.tokenFile, {
      ...ids,
      anthropic_identity_token_file: masked(tokenFile),
    });

    const referenceDialog = await openAddCredentialDialog(
      page,
      names.secretReference,
    );
    await pickIdentitySource(
      page,
      referenceDialog,
      "Identity token secret reference",
    );
    await fillFields(referenceDialog, {
      anthropic_federation_rule_id: ids.anthropic_federation_rule_id,
      anthropic_identity_token: `eyJhbGciOiJSUzI1NiJ9.${suffix}.signature`,
    });
    await expectBlocked(
      page,
      referenceDialog,
      "Add Credential",
      RAW_TOKEN_MESSAGE,
      1,
      posts,
    );
    const reference = `oidc/env/ANTHROPIC_IDENTITY_${suffix.replace(/\W/g, "_")}`;
    await fillFields(referenceDialog, { anthropic_identity_token: reference });
    const referenced = await submitCredential(
      page,
      referenceDialog,
      "POST",
      "/credentials",
      "Add Credential",
    );
    expect(referenced.credential_values).toEqual({
      anthropic_federation_rule_id: ids.anthropic_federation_rule_id,
      anthropic_identity_token: reference,
    });
    await expect(
      page.locator("tr", { hasText: names.secretReference }),
    ).toBeVisible({ timeout: 15_000 });

    const issuerDialog = await openAddCredentialDialog(
      page,
      names.internalIssuer,
    );
    await pickIdentitySource(
      page,
      issuerDialog,
      "Token signed by LiteLLM (internal issuer)",
    );
    const issuer = {
      anthropic_issuer_url: `https://issuer-${suffix}.example`,
      anthropic_issuer_subject: `proxy-${suffix}`,
      anthropic_issuer_audience: `anthropic-${suffix}`,
      anthropic_issuer_signing_key_ref:
        "os.environ/ANTHROPIC_ISSUER_SIGNING_KEY",
    };
    await fillFields(issuerDialog, {
      anthropic_federation_rule_id: ids.anthropic_federation_rule_id,
      ...issuer,
      anthropic_issuer_ttl_seconds: "7200",
    });
    await expectBlocked(
      page,
      issuerDialog,
      "Add Credential",
      TTL_MESSAGE,
      2,
      posts,
    );
    await fillFields(issuerDialog, { anthropic_issuer_ttl_seconds: "900" });
    const issued = await submitCredential(
      page,
      issuerDialog,
      "POST",
      "/credentials",
      "Add Credential",
    );
    expect(issued.credential_values).toEqual({
      anthropic_federation_rule_id: ids.anthropic_federation_rule_id,
      ...issuer,
      anthropic_issuer_ttl_seconds: 900,
      anthropic_identity_source: "internal_issuer",
    });
    await expect(
      page.locator("tr", { hasText: names.internalIssuer }),
    ).toBeVisible({ timeout: 15_000 });

    const environmentDialog = await openAddCredentialDialog(
      page,
      names.environment,
    );
    await pickIdentitySource(
      page,
      environmentDialog,
      "Proxy environment variables",
    );
    await expectBlocked(
      page,
      environmentDialog,
      "Add Credential",
      NO_IDS_MESSAGE,
      3,
      posts,
    );
    await fillFields(environmentDialog, {
      anthropic_organization_id: ids.anthropic_organization_id,
    });
    const fromEnvironment = await submitCredential(
      page,
      environmentDialog,
      "POST",
      "/credentials",
      "Add Credential",
    );
    expect(fromEnvironment.credential_values).toEqual({
      anthropic_organization_id: ids.anthropic_organization_id,
    });
    await expect(
      page.locator("tr", { hasText: names.environment }),
    ).toBeVisible({ timeout: 15_000 });
    expect(posts()).toBe(4);
  } finally {
    await deleteCredentials(request, Object.values(names));
  }
});

test("the Edit Credential dialog sends only the changed values and deletes the keys the new choice leaves behind", async ({
  page,
  request,
}) => {
  const suffix = uniqueSuffix();
  const names = {
    fromApiKey: `int-wif-edit-key-${suffix}`,
    clearWorkspace: `int-wif-edit-clear-${suffix}`,
    switchSource: `int-wif-edit-source-${suffix}`,
    switchProvider: `int-wif-edit-provider-${suffix}`,
  };
  const ruleId = `fdrl_${suffix}`;
  const tokenFile = `/var/run/secrets/anthropic/${suffix}/token`;
  try {
    await createCredential(request, names.fromApiKey, {
      api_key: `sk-ant-${suffix}`,
    });
    await createCredential(request, names.clearWorkspace, {
      anthropic_federation_rule_id: ruleId,
      anthropic_federation_workspace_id: `wrkspc_${suffix}`,
      anthropic_identity_token_file: tokenFile,
    });
    await createCredential(request, names.switchSource, {
      anthropic_federation_rule_id: ruleId,
      anthropic_identity_token_file: tokenFile,
    });
    await createCredential(request, names.switchProvider, {
      anthropic_federation_rule_id: ruleId,
      anthropic_organization_id: `org-${suffix}`,
      anthropic_identity_token_file: tokenFile,
    });

    await logInAsAdmin(page);
    await openModelsTab(page, "LLM Credentials");

    const keyDialog = await openEditDialog(page, names.fromApiKey);
    await pickAuthMethod(page, keyDialog, FEDERATION_BADGE);
    await fillFields(keyDialog, {
      anthropic_federation_rule_id: ruleId,
      anthropic_identity_token_file: tokenFile,
    });
    const federated = await submitCredential(
      page,
      keyDialog,
      "PATCH",
      `/credentials/${names.fromApiKey}`,
      "Update Credential",
    );
    expect(federated).toEqual({
      credential_name: names.fromApiKey,
      credential_values: {
        anthropic_federation_rule_id: ruleId,
        anthropic_identity_token_file: tokenFile,
      },
      credential_info: { custom_llm_provider: "anthropic" },
      credential_values_to_delete: ["api_key"],
    });
    await expect(page.getByText(UPDATED_TOAST)).toBeVisible();
    await expect(
      page.locator("tr", { hasText: names.fromApiKey }),
    ).toContainText(FEDERATION_BADGE);
    await expectStored(request, names.fromApiKey, {
      anthropic_federation_rule_id: ruleId,
      anthropic_identity_token_file: masked(tokenFile),
    });

    const clearDialog = await openEditDialog(page, names.clearWorkspace);
    await fillFields(clearDialog, { anthropic_federation_workspace_id: "" });
    const cleared = await submitCredential(
      page,
      clearDialog,
      "PATCH",
      `/credentials/${names.clearWorkspace}`,
      "Update Credential",
    );
    expect(cleared).toEqual({
      credential_name: names.clearWorkspace,
      credential_values: {},
      credential_info: { custom_llm_provider: "anthropic" },
      credential_values_to_delete: ["anthropic_federation_workspace_id"],
    });
    await expect(page.getByText(UPDATED_TOAST)).toBeVisible();
    await expectStored(request, names.clearWorkspace, {
      anthropic_federation_rule_id: ruleId,
      anthropic_identity_token_file: masked(tokenFile),
    });

    const sourceDialog = await openEditDialog(page, names.switchSource);
    await pickIdentitySource(
      page,
      sourceDialog,
      "Identity token secret reference",
    );
    const reference = `oidc/env/ANTHROPIC_IDENTITY_${suffix.replace(/\W/g, "_")}`;
    await fillFields(sourceDialog, { anthropic_identity_token: reference });
    const switched = await submitCredential(
      page,
      sourceDialog,
      "PATCH",
      `/credentials/${names.switchSource}`,
      "Update Credential",
    );
    expect(switched).toEqual({
      credential_name: names.switchSource,
      credential_values: { anthropic_identity_token: reference },
      credential_info: { custom_llm_provider: "anthropic" },
      credential_values_to_delete: ["anthropic_identity_token_file"],
    });
    await expect(page.getByText(UPDATED_TOAST)).toBeVisible();
    await expectStored(request, names.switchSource, {
      anthropic_federation_rule_id: ruleId,
      anthropic_identity_token: masked(reference),
    });

    const providerDialog = await openEditDialog(page, names.switchProvider);
    await pickProvider(page, providerDialog, OPENAI_LABEL);
    const openai = {
      api_key: `sk-openai-${suffix}`,
      api_base: `https://openai-${suffix}.example/v1`,
    };
    await fillFields(providerDialog, openai);
    const moved = await submitCredential(
      page,
      providerDialog,
      "PATCH",
      `/credentials/${names.switchProvider}`,
      "Update Credential",
    );
    expect(moved.credential_values).toEqual(openai);
    expect(moved.credential_info).toEqual({
      custom_llm_provider: OPENAI_LABEL,
    });
    expect([...(moved.credential_values_to_delete ?? [])].sort()).toEqual([
      "anthropic_federation_rule_id",
      "anthropic_identity_token_file",
      "anthropic_organization_id",
    ]);
    await expect(page.getByText(UPDATED_TOAST)).toBeVisible();
    const onOpenai = await expectStored(request, names.switchProvider, {
      api_key: masked(openai.api_key),
      api_base: openai.api_base,
    });
    expect(onOpenai.credential_info.custom_llm_provider).toBe(OPENAI_LABEL);
    await expect(
      page.locator("tr", { hasText: names.switchProvider }),
    ).not.toContainText(FEDERATION_BADGE);
  } finally {
    await deleteCredentials(request, Object.values(names));
  }
});

test("a proxy admin saves a federation credential from the Add Model tab and the model is created against it", async ({
  page,
  request,
}) => {
  test.slow();
  const suffix = uniqueSuffix();
  const credentialName = `int-wif-model-${suffix}`;
  const modelName = `int-wif-model-${suffix}`;
  const ids = {
    anthropic_federation_rule_id: `fdrl_${suffix}`,
    anthropic_organization_id: `org-${suffix}`,
  };
  const tokenFile = `/opt/int-wif-${suffix}/token`;
  let deploymentId = "";
  try {
    await logInAsAdmin(page);
    await openModelsTab(page, "Add Model");
    await pickProvider(page, page.locator("body"), ANTHROPIC_LABEL);
    await expect(page.getByText(FEDERATION_HELP)).toBeVisible();
    await page
      .getByRole("button", { name: FEDERATION_BUTTON, exact: true })
      .click();

    const dialog = page.getByRole("dialog", { name: "Add New Credential" });
    await expect(dialog).toBeVisible();
    const provider = dialog.getByPlaceholder("Select a provider");
    await expect(provider).toHaveValue(ANTHROPIC_LABEL);
    await expect(provider).toBeDisabled();
    await expect(dialog.locator("#anthropic_auth_method")).toContainText(
      FEDERATION_BADGE,
    );
    await dialog
      .getByPlaceholder("Enter a friendly name for these credentials")
      .fill(credentialName);
    await fillFields(dialog, {
      ...ids,
      anthropic_identity_token_file: tokenFile,
    });
    const created = await submitCredential(
      page,
      dialog,
      "POST",
      "/credentials",
      "Add Credential",
    );
    expect(created).toEqual({
      credential_name: credentialName,
      credential_values: { ...ids, anthropic_identity_token_file: tokenFile },
      credential_info: { custom_llm_provider: ANTHROPIC_LABEL },
    });
    await expect(page.getByText(ADDED_TOAST)).toBeVisible();
    await expect(dialog).toBeHidden();
    await expect(page.locator("#litellm_credential_name")).toHaveValue(
      credentialName,
    );
    await expect(page.locator("#api_key")).toHaveCount(0);

    await pickOption(
      page,
      page.getByRole("combobox", { name: "Select models" }),
      "Custom Model Name (Enter below)",
    );
    await page.keyboard.press("Escape");
    await page.getByPlaceholder("Enter custom model name").fill(modelName);

    await expectStored(request, credentialName, {
      ...ids,
      anthropic_identity_token_file: masked(tokenFile),
    });

    const probe = await captureRequestBody<ModelWrite>(
      page,
      { method: "POST", urlIncludes: "/health/test_connection" },
      () => page.getByTestId("test-connect-btn").click(),
    );
    expect(probe.litellm_params.litellm_credential_name).toBe(credentialName);
    expect(probe.litellm_params.custom_llm_provider).toBe("anthropic");
    expect("api_key" in probe.litellm_params).toBe(false);
    const results = page.getByRole("dialog", {
      name: "Connection Test Results",
    });
    await expect(results.getByTestId("connection-failure-msg")).toBeVisible({
      timeout: 60_000,
    });
    await expect(results.getByText(ALLOWLIST_VARIABLE)).toBeVisible();
    await expect(results.getByText(tokenFile)).toBeVisible();
    await page.keyboard.press("Escape");
    await expect(results).toBeHidden();

    const creation = page.waitForResponse(
      (response) =>
        response.request().method() === "POST" &&
        response.url().includes("/model/new"),
    );
    const added = await captureRequestBody<ModelWrite>(
      page,
      { method: "POST", urlIncludes: "/model/new" },
      () => page.getByTestId("add-model-btn").click(),
    );
    const creationResponse = await creation;
    expect(
      creationResponse.status(),
      `POST /model/new: ${await creationResponse.text()}`,
    ).toBe(200);
    deploymentId = ((await creationResponse.json()) as ModelCreated).model_id;
    expect(deploymentId).not.toBe("");
    expect(added.model_name).toBe(modelName);
    expect(added.litellm_params.litellm_credential_name).toBe(credentialName);
    expect(added.litellm_params.custom_llm_provider).toBe("anthropic");
    expect("api_key" in added.litellm_params).toBe(false);

    const findDeployment = async (): Promise<Deployment | undefined> => {
      const info = await readBack<{ data: Deployment[] }>(page, "/model/info");
      return info.data.find(
        (deployment) => deployment.model_name === modelName,
      );
    };
    await expect
      .poll(async () => (await findDeployment())?.model_info.id ?? "", {
        timeout: 70_000,
        message: "the deployment never appeared in /model/info",
      })
      .not.toBe("");
    const deployment = await findDeployment();
    expect(deployment?.model_info.id).toBe(deploymentId);
    expect(deployment?.litellm_params.litellm_credential_name).toBe(
      credentialName,
    );
    expect(deployment?.litellm_params.custom_llm_provider).toBe("anthropic");
    expect("api_key" in (deployment?.litellm_params ?? {})).toBe(false);
  } finally {
    if (deploymentId) {
      await postAsMaster(request, "/model/delete", { id: deploymentId });
    }
    await deleteCredentials(request, [credentialName]);
  }
});

test("a team admin who is not a proxy admin gets no federation shortcut on the Add Model tab", async ({
  page,
  request,
}) => {
  const suffix = uniqueSuffix();
  const userId = `int-wif-team-admin-${suffix}`;
  const email = `${userId}@integration.example`;
  const password = `Int-Wif-${suffix}!`;
  const teamAlias = `int-wif-team-${suffix}`;
  let teamId = "";
  try {
    await postAsMaster(request, "/user/new", {
      user_id: userId,
      user_email: email,
      user_role: "internal_user",
      auto_create_key: false,
    });
    await setInvitedUserPassword(request, userId, password);
    teamId = (
      await postAsMaster<{ team_id: string }>(request, "/team/new", {
        team_alias: teamAlias,
        members_with_roles: [{ role: "admin", user_id: userId }],
      })
    ).team_id;

    await logInThroughLoginPage(page, email, password);
    await openModelsTab(page, "Add Model");
    await expect(page.getByText("Team Selection Required")).toBeVisible();
    await page.getByPlaceholder("Search or select a team").click();
    await page
      .getByRole("option")
      .filter({ hasText: teamAlias })
      .first()
      .click();
    await pickProvider(page, page.locator("body"), ANTHROPIC_LABEL);
    await expect(page.getByPlaceholder("Select a provider")).toHaveValue(
      ANTHROPIC_LABEL,
    );
    await expect(page.locator("#api_key")).toBeVisible();
    await expect(
      page.getByRole("button", { name: FEDERATION_BUTTON, exact: true }),
    ).toHaveCount(0);
    await expect(page.getByText(FEDERATION_HELP)).toHaveCount(0);
  } finally {
    if (teamId) {
      await postAsMaster(request, "/team/delete", { team_ids: [teamId] });
    }
    await postAsMaster(request, "/user/delete", { user_ids: [userId] });
  }
});
