import { expect, test, type Page } from "@playwright/test";
import path from "node:path";
import { randomUUID } from "node:crypto";
import { credentials } from "./setup";

const sessionKey = () => `litellm_control_plane_session:${process.env.CONTROL_PLANE_E2E_API_PREFIX}`;

async function signIn(page: Page) {
  const { username, password } = credentials();
  await page.getByLabel("Username").fill(username);
  await page.getByLabel("Password").fill(password);
  await page.getByRole("button", { name: "Sign in" }).click();
  await expect(page.getByRole("combobox", { name: "Dataplane" })).toBeVisible();
}

async function proof(page: Page, name: string) {
  const directory = process.env.CONTROL_PLANE_E2E_PROOF_DIR;
  if (directory) await page.screenshot({ path: path.join(directory, `${name}.png`), fullPage: true });
}

test("one session switches live dataplanes through the control plane", async ({ page, baseURL }) => {
  const prefix = encodeURI(process.env.CONTROL_PLANE_E2E_API_PREFIX!);
  const requests: URL[] = [];
  const denials: string[] = [];
  page.on("request", (request) => {
    if (["fetch", "xhr"].includes(request.resourceType())) requests.push(new URL(request.url()));
  });
  page.on("response", (response) => {
    if (response.status() === 403) denials.push(new URL(response.url()).pathname);
  });
  await page.goto("ui/login/");
  await expect(page.getByRole("heading", { name: "Control Plane Login" })).toBeVisible();
  await proof(page, "control-plane-login");
  const [first] = await Promise.all([
    page.waitForResponse((response) => {
      const pathname = new URL(response.url()).pathname;
      return pathname.startsWith(`${prefix}/dataplanes/`) && pathname.endsWith("/proxy/key/list");
    }),
    signIn(page),
  ]);
  expect(first.ok()).toBe(true);
  const brand = page.getByRole("img", { name: "LiteLLM", exact: true }).first();
  await expect.poll(() => brand.evaluate((img: HTMLImageElement) => img.naturalWidth)).toBeGreaterThan(0);
  const mount = new URL(baseURL!);
  const discoveryPath = mount.pathname === "/" ? "litellm/.well-known/litellm-ui-config" : ".well-known/litellm-ui-config";
  const discovery = await page.request.get(new URL(discoveryPath, mount).toString());
  const { litellm_version: version } = await discovery.json();
  if (version) await expect(page.getByRole("link", { name: `v${version}` })).toBeVisible();

  const picker = page.getByRole("combobox", { name: "Dataplane" });
  const options = picker.getByRole("option");
  await expect(options).toHaveCount(2);
  await expect(options.first()).not.toHaveText(/checking health/i);
  await expect(options.last()).not.toHaveText(/checking health/i);
  await proof(page, "control-plane-picker");
  const ids = await options.evaluateAll((elements) => elements.map((element) => (element as HTMLOptionElement).value));
  const waitForKeys = (id: string) => page.waitForResponse((response) => new URL(response.url()).pathname === `${prefix}/dataplanes/${id}/proxy/key/list`);
  expect(new URL(first.url()).pathname).toBe(`${prefix}/dataplanes/${ids[0]}/proxy/key/list`);
  await expect(page.getByRole("heading", { name: "Virtual Keys", exact: true }).last()).toBeVisible();
  await proof(page, "control-plane-keys");
  const [second] = await Promise.all([waitForKeys(ids[1]), picker.selectOption(ids[1])]);
  expect(second.ok()).toBe(true);
  await expect(picker).toHaveValue(ids[1]);
  await proof(page, "control-plane-keys-switched");
  const [models] = await Promise.all([
    page.waitForResponse((response) => new URL(response.url()).pathname === `${prefix}/dataplanes/${ids[1]}/proxy/v2/model/info`),
    page.getByRole("link", { name: "Model Management", exact: true }).click(),
  ]);
  expect(models.ok()).toBe(true);
  await expect(page.getByRole("heading", { name: "Model Management", exact: true }).last()).toBeVisible();
  await proof(page, "control-plane-models");
  expect(requests.filter((url) => url.pathname === `${prefix}/auth/login`)).toHaveLength(1);

  for (const [label, route, file] of [
    ["Fleet Models", "/fleet/models", "fleet-models"],
    ["Fleet Teams", "/fleet/teams", "fleet-teams"],
    ["Fleet Usage", "/usage/summary", "fleet-usage"],
    ["Dataplanes", "/dataplanes", "dataplanes"],
    ["Roles", "/roles", "roles"],
    ["Audit Log", "/audit", "audit-log"],
  ]) {
    const click = () => page.getByRole("link", { name: label, exact: true }).click();
    if (route === "/dataplanes") await click();
    else {
      const [response] = await Promise.all([
        page.waitForResponse((response) => new URL(response.url()).pathname === `${prefix}${route}`),
        click(),
      ]);
      expect(response.ok(), `${route} returned ${response.status()}`).toBe(true);
    }
    await expect(page.getByRole("heading", { name: label, exact: true }).last()).toBeVisible();
    await expect(page.getByRole("link", { name: label, exact: true })).toHaveAttribute("data-active", "");
    await expect(page.getByRole("link", { name: "Home", exact: true })).not.toHaveAttribute("data-active", "");
    await expect(page.locator("main").getByText(/loading/i)).toHaveCount(0);
    await proof(page, file);
  }
  expect(requests.every((url) => url.origin === new URL(baseURL!).origin)).toBe(true);
  expect(denials).toEqual([]);
  expect(requests.map((url) => url.pathname)).not.toEqual(expect.arrayContaining([
    expect.stringMatching(/\/proxy\/(?:health\/readiness\/details|get\/latest_release_info|(?:sso\/)?get\/ui_settings)$/),
  ]));
});

test("creates, updates, and deletes a runtime role", async ({ page, request }) => {
  const prefix = process.env.CONTROL_PLANE_E2E_API_PREFIX;
  const name = `e2e-${randomUUID()}`;
  await page.goto("ui/");
  await signIn(page);
  await page.getByRole("link", { name: "Roles", exact: true }).click();
  await expect(page.getByText("Config (read-only)").first()).toBeVisible();
  await page.getByRole("button", { name: "Create role" }).click();
  await page.getByLabel("Role name").fill(name);
  await page.getByLabel("Direct scopes (comma separated)").fill("fleet:read");
  const row = page.getByRole("row").filter({ has: page.getByRole("cell", { name, exact: true }) });
  try {
    await page.getByRole("button", { name: "Save", exact: true }).click();
    await expect(row).toContainText("fleet:read");
    await row.getByRole("button", { name: "Edit", exact: true }).click();
    await page.getByLabel("Direct scopes (comma separated)").fill("usage:read");
    await page.getByRole("button", { name: "Save", exact: true }).click();
    await expect(row).toContainText("usage:read");
    await row.getByRole("button", { name: "Delete", exact: true }).click();
    await page.getByRole("button", { name: "Confirm delete", exact: true }).click();
    await expect(row).toHaveCount(0);
  } finally {
    const token = await page.evaluate((key) => sessionStorage.getItem(key), sessionKey());
    if (token) {
      const response = await request.delete(`${prefix}/roles/${name}`, {
        headers: { Authorization: `Bearer ${token}` },
        maxRedirects: 0,
      });
      expect([204, 404]).toContain(response.status());
    }
  }
});

test("shows partial fleet results during a live dataplane outage", async ({ page }) => {
  const unavailable = process.env.CONTROL_PLANE_E2E_UNREACHABLE;
  test.skip(!unavailable, "Set CONTROL_PLANE_E2E_UNREACHABLE to a stopped dataplane's ID.");
  await page.goto("ui/");
  await signIn(page);
  for (const [name, route, filename] of [
    ["Fleet Models", "/fleet/models", "fleet-models-partial"],
    ["Fleet Teams", "/fleet/teams", "fleet-teams-partial"],
  ]) {
    const [response] = await Promise.all([
      page.waitForResponse((candidate) => new URL(candidate.url()).pathname ===
        `${process.env.CONTROL_PLANE_E2E_API_PREFIX}${route}`, { timeout: 45_000 }),
      page.getByRole("link", { name, exact: true }).click(),
    ]);
    expect(response.ok()).toBe(true);
    await expect(page.getByText("Partial results. Totals do not cover every dataplane.")).toBeVisible({
      timeout: 45_000,
    });
    await expect(page.getByText(`Unreachable: ${unavailable}`, { exact: false })).toBeVisible();
    await proof(page, filename);
  }
});

test("creates, edits, and deletes a runtime dataplane", async ({ page, request }) => {
  const prefix = process.env.CONTROL_PLANE_E2E_API_PREFIX;
  const id = `e2e-${randomUUID()}`;
  const name = "QA runtime dataplane";
  await page.goto("ui/");
  await signIn(page);
  await page.getByRole("link", { name: "Dataplanes", exact: true }).click();
  await expect(page.getByRole("table")).toBeVisible();
  const urlCell = page.getByRole("table").getByRole("cell", { name: /^https?:\/\// }).first();
  await expect(urlCell).toBeVisible();
  const url = await urlCell.textContent();
  const row = page.getByRole("row").filter({ has: page.getByRole("cell", { name: id, exact: true }) });
  try {
    await page.getByRole("button", { name: "Create dataplane" }).click();
    await page.getByLabel("Dataplane ID").fill(id);
    await page.getByLabel("Name", { exact: true }).fill(name);
    await page.getByLabel("URL", { exact: true }).fill(url!);
    if (url?.startsWith("http:")) await page.getByLabel("Allow insecure HTTP").check();
    await page.getByLabel("Service token value").fill("qa-registry-only-token");
    await page.getByRole("button", { name: "Save", exact: true }).click();
    await expect(row).toContainText(name);
    await expect(row).toContainText("Runtime");
    await proof(page, "dataplane-created");
    await row.getByRole("button", { name: "Edit" }).click();
    await page.getByLabel("Name", { exact: true }).fill(`${name} updated`);
    await page.getByLabel("Service token value").fill("qa-registry-only-token");
    await page.getByRole("button", { name: "Save", exact: true }).click();
    await expect(row).toContainText(`${name} updated`);
    await proof(page, "dataplane-edited");
    await row.getByRole("button", { name: "Delete" }).click();
    await page.getByRole("button", { name: "Confirm delete" }).click();
    await expect(row).toHaveCount(0);
    await proof(page, "dataplane-deleted");
  } finally {
    const token = await page.evaluate((key) => sessionStorage.getItem(key), sessionKey());
    if (token) {
      const response = await request.delete(`${prefix}/dataplanes/${id}`, {
        headers: { Authorization: `Bearer ${token}` },
        maxRedirects: 0,
      });
      expect([204, 404]).toContain(response.status());
    }
  }
});

test("explains a disabled write to a scoped session", async ({ page }) => {
  const token = process.env.CONTROL_PLANE_E2E_SCOPED_TOKEN;
  test.skip(!token, "Provide a read-only control plane token bound to one test dataplane.");
  await page.addInitScript(({ key, value }) => sessionStorage.setItem(key, value), { key: sessionKey(), value: token! });
  await page.goto("ui/");
  await expect(page.getByRole("combobox", { name: "Dataplane" })).toBeVisible();
  await expect(page.getByRole("combobox", { name: "Dataplane" }).getByRole("option")).toHaveCount(1);
  await page.getByRole("link", { name: "Virtual Keys" }).click();
  const create = page.getByRole("button", { name: "Create key" });
  await expect(create).toBeDisabled();
  const trigger = page.locator('[data-slot="tooltip-trigger"]').filter({ has: create });
  await trigger.hover();
  await expect(page.getByText("Requires proxy:write.", { exact: true })).toBeVisible();
  await proof(page, "control-plane-scoped-action");
});

test("keeps an unsupported key action visible even for a writer", async ({ page }) => {
  await page.goto("ui/login/");
  await signIn(page);
  await expect(page.getByRole("combobox", { name: "Dataplane" }).getByRole("option")).toHaveCount(2);
  await page.getByRole("link", { name: "Virtual Keys" }).click();
  await expect(page.getByRole("button", { name: "Create New Key" })).toBeEnabled();
  await page.getByRole("table").getByRole("row").nth(1).getByRole("button").first().click();
  const regenerate = page.getByRole("button", { name: "Regenerate key", exact: true });
  await expect(regenerate.first()).toBeDisabled();
  await page.locator('[data-slot="tooltip-trigger"]').filter({ has: regenerate }).first().hover();
  const message = "This action isn't available through the control plane yet.";
  await expect(page.getByText(message, { exact: true })).toBeVisible();
  await proof(page, "control-plane-unavailable-regenerate-key");
});
