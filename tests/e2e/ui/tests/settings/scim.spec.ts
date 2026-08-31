import { test, expect, Page as PlaywrightPage } from "@playwright/test";
import { ADMIN_STORAGE_PATH } from "../../constants";
import { Page } from "../../fixtures/pages";
import { navigateToPage } from "../../helpers/navigation";
import { masterKey } from "../../helpers/traffic";

const rootPath = (): string => process.env.SERVER_ROOT_PATH ?? "";

async function createScimTokenViaUi(page: PlaywrightPage, alias: string): Promise<string> {
  await navigateToPage(page, Page.AdminPanel);
  await page.getByRole("tab", { name: "SCIM" }).click();

  await expect(page.getByText("SCIM Tenant URL")).toBeVisible();
  await expect(page.locator("input[disabled]").first()).toHaveValue(/\/scim\/v2$/);

  await page.getByLabel("Token Name").fill(alias);
  await page.getByRole("button", { name: "Create SCIM Token" }).click();

  await expect(page.getByText(/copy this token now/i)).toBeVisible({ timeout: 15_000 });
  const token = await page.locator('input[type="password"]').inputValue();
  expect(token, "the one-time token panel shows a usable virtual key").toMatch(/^sk-/);
  return token;
}

test.describe("Admin Settings - SCIM", () => {
  test.use({ storageState: ADMIN_STORAGE_PATH });

  test("Create SCIM Token shows the token once and offers to create another", async ({ page }) => {
    const token = await createScimTokenViaUi(page, `e2e-scim-ui-${Date.now()}`);

    await page.getByRole("button", { name: "Create Another Token" }).click();
    await expect(page.getByRole("button", { name: "Create SCIM Token" })).toBeVisible();
    await expect(page.getByText(/copy this token now/i)).toBeHidden();

    await deleteKey(page, token);
  });

  test("a UI-minted SCIM token authorizes the SCIM API", async ({ page, request }) => {
    test.skip(!process.env.LITELLM_LICENSE, "LITELLM_LICENSE not set in test env — /scim/v2 is premium-gated");

    const token = await createScimTokenViaUi(page, `e2e-scim-api-${Date.now()}`);

    const denied = await request.get(`${rootPath()}/scim/v2/Groups`, {
      headers: { Authorization: "Bearer sk-not-a-real-key" },
    });
    expect(denied.status(), "an unknown key must not reach SCIM").toBe(401);

    const res = await request.get(`${rootPath()}/scim/v2/Groups`, {
      headers: { Authorization: `Bearer ${token}` },
    });
    expect(res.status(), `SCIM Groups listing failed: ${await res.text()}`).toBe(200);
    const body = await res.json();
    expect(body.schemas, "SCIM answers with a ListResponse").toContain("urn:ietf:params:scim:api:messages:2.0:ListResponse");
    expect(Array.isArray(body.Resources), "SCIM ListResponse carries a Resources array").toBe(true);

    await deleteKey(page, token);
  });
});

async function deleteKey(page: PlaywrightPage, key: string): Promise<void> {
  const res = await page.request.post(`${rootPath()}/key/delete`, {
    headers: { Authorization: `Bearer ${masterKey()}`, "Content-Type": "application/json" },
    data: { keys: [key] },
  });
  expect(res.ok(), `cleanup of the SCIM token failed (${res.status()})`).toBe(true);
}
