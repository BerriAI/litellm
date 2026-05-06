/**
 * Shared helpers for the cloud-agents Playwright suite.
 *
 * The dashboard's app router routes (/agents/...) only render under
 * `next dev`, not under the proxy's static export. These tests therefore
 * target http://localhost:3000 directly and run against the mock provider
 * (NEXT_PUBLIC_USE_MOCK_AGENTS=true) so they're deterministic without
 * Epic A merged.
 *
 * useAuthorized() requires a valid (unexpired) JWT in the `token` cookie
 * before it'll render the dashboard. We mint a 1-hour HS-style fake here —
 * client-side jwt-decode never verifies the signature, so any structurally
 * valid base64 payload works.
 */
import type { Page } from "@playwright/test";

export const AGENTS_DEV_URL = process.env.AGENTS_DEV_URL ?? "http://localhost:3000";

function base64url(input: string): string {
  return Buffer.from(input).toString("base64").replace(/=+$/g, "").replace(/\+/g, "-").replace(/\//g, "_");
}

/**
 * Build a fake JWT with `key` (used as accessToken) and an exp 1 hour out.
 * Signature is the literal string "fake" — jwt-decode ignores it.
 */
export function buildFakeToken(): string {
  const header = base64url(JSON.stringify({ alg: "none", typ: "JWT" }));
  const payload = base64url(
    JSON.stringify({
      key: "sk-mock",
      user_id: "e2e-mock-user",
      user_email: "e2e@test.local",
      user_role: "proxy_admin",
      exp: Math.floor(Date.now() / 1000) + 3600,
    }),
  );
  return `${header}.${payload}.fake`;
}

/**
 * Inject the fake token cookie for localhost:3000 so useAuthorized() doesn't
 * redirect to /ui/login. Call before any page.goto() in the agents suite.
 */
export async function authenticateAgentsPage(page: Page): Promise<void> {
  const url = new URL(AGENTS_DEV_URL);
  await page.context().addCookies([
    {
      name: "token",
      value: buildFakeToken(),
      domain: url.hostname,
      path: "/",
      sameSite: "Lax",
    },
  ]);
}

/**
 * Wait for the mock client to settle and the agents table to render.
 * The mock provider resolves immediately, but React still has to flush.
 */
export async function gotoAgentsList(page: Page): Promise<void> {
  await authenticateAgentsPage(page);
  await page.goto(`${AGENTS_DEV_URL}/agents`);
  await page.getByTestId("agents-table").waitFor({ state: "visible" });
}
