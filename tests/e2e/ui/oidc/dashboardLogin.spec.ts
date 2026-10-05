import { expect, test, type Page as PlaywrightPage, type Response } from "@playwright/test";
import { Page } from "../fixtures/pages";
import { navigateToPage } from "../helpers/navigation";

function sessionKey(tokenCookie: string): string {
  const claims: unknown = JSON.parse(Buffer.from(tokenCookie.split(".")[1] ?? "", "base64url").toString("utf8"));
  const key = claims !== null && typeof claims === "object" && "key" in claims ? claims.key : undefined;
  if (typeof key !== "string") throw new Error("The dashboard token cookie carries no key claim");
  return key;
}

async function openPageAndCapture(page: PlaywrightPage, target: Page, apiPath: string): Promise<Response> {
  const response = page.waitForResponse((r) => new URL(r.url()).pathname === apiPath);
  await navigateToPage(page, target);
  return response;
}

test("SSO login issues a session that authorizes dashboard data requests", async ({ page, context, baseURL }) => {
  const tokenCookie = (await context.cookies(baseURL)).find((cookie) => cookie.name === "token");
  expect(tokenCookie, "SSO login sets the dashboard token cookie").toBeDefined();
  const key = sessionKey(tokenCookie?.value ?? "");
  expect(key, "SSO login issues a session token, not a virtual key").not.toMatch(/^sk-/);

  const keyList = await openPageAndCapture(page, Page.ApiKeys, "/key/list");
  expect(keyList.request().headers()["authorization"]).toBe(`Bearer ${key}`);
  expect(keyList.status()).toBe(200);
  expect(Array.isArray((await keyList.json()).keys)).toBe(true);

  const modelInfo = await openPageAndCapture(page, Page.Models, "/v2/model/info");
  expect(modelInfo.request().headers()["authorization"]).toBe(`Bearer ${key}`);
  expect(modelInfo.status()).toBe(200);
  const models: { model_name: string }[] = (await modelInfo.json()).data;
  expect(models.length, "the stack serves at least one model").toBeGreaterThan(0);
  await expect(page.getByText(models[0].model_name, { exact: true }).first()).toBeVisible();
});
