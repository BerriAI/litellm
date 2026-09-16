import { expect, type APIRequestContext, type Page } from "@playwright/test";
import { UI_BASE_URL } from "../constants";
import { masterKey, rootPath } from "./traffic";

const endpoint = (route: string): string => `${UI_BASE_URL}${rootPath()}${route}`;

export async function setInvitedUserPassword(
  request: APIRequestContext,
  userId: string,
  password: string,
): Promise<void> {
  const invitation = await request.post(endpoint("/invitation/new"), {
    headers: { Authorization: `Bearer ${masterKey()}` },
    data: { user_id: userId },
  });
  expect(invitation.ok(), `Create invitation for ${userId}: HTTP ${invitation.status()}`).toBe(true);
  const { id } = await invitation.json();
  expect(typeof id, "invitation ID").toBe("string");

  const onboarding = await request.get(endpoint("/onboarding/get_token"), {
    params: { invite_link: id },
  });
  expect(onboarding.ok(), `Get onboarding session for ${userId}: HTTP ${onboarding.status()}`).toBe(true);
  const { token } = await onboarding.json();
  const payload = JSON.parse(Buffer.from(token.split(".")[1], "base64url").toString("utf-8"));
  expect(typeof payload.key, "onboarding credential").toBe("string");
  const claimed = await request.post(endpoint("/onboarding/claim_token"), {
    headers: { Authorization: `Bearer ${payload.key}` },
    data: { invitation_link: id, user_id: userId, password },
  });
  expect(claimed.ok(), `Claim invitation for ${userId}: HTTP ${claimed.status()}`).toBe(true);
}

export async function readDashboardSession(page: Page): Promise<{
  key: string;
  user_id: string;
  password_reset_required?: boolean;
}> {
  await expect.poll(async () => (await page.context().cookies()).some((cookie) => cookie.name === "token")).toBe(true);
  const cookie = (await page.context().cookies()).find((candidate) => candidate.name === "token")!;
  return JSON.parse(Buffer.from(cookie.value.split(".")[1], "base64url").toString("utf-8"));
}

export async function expectUnrestrictedDashboard(page: Page): Promise<void> {
  const virtualKeys = page.getByRole("complementary").getByRole("link", { name: "Virtual Keys", exact: true });
  await expect(virtualKeys).toBeVisible({ timeout: 30_000 });
  const session = await readDashboardSession(page);
  expect(session.password_reset_required === true, "login must not require a password reset").toBe(false);
  await virtualKeys.click();
  await expect(page.getByRole("main").getByRole("heading", { name: "Virtual Keys", exact: true })).toBeVisible({
    timeout: 30_000,
  });
  const info = await page.request.get(endpoint("/user/info"), {
    headers: { Authorization: `Bearer ${session.key}` },
    params: { user_id: session.user_id },
  });
  expect(info.ok(), `Read own user with dashboard session: HTTP ${info.status()}`).toBe(true);
  expect((await info.json()).user_id).toBe(session.user_id);
}
