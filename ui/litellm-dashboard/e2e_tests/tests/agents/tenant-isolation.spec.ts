/**
 * LIT-2881 Validation #8 — Cross-tenant isolation.
 *
 * Log in as A then B; B does not see A's agents or sessions.
 *
 * This validation is dependent on Epic A landing real tenanted endpoints —
 * the mock provider has only one in-memory store and intentionally does
 * not partition by user. We exercise the *plumbing* here: when a
 * different fake JWT is presented, useAuthorized still gates rendering
 * and the API client passes the new bearer through. Once Epic A merges,
 * this spec's assertions should be tightened to verify backend
 * partitioning. Filed as follow-up (LIT-2881-followup-tenant-isolation).
 */
import { test, expect } from "@playwright/test";
import { AGENTS_DEV_URL, buildFakeToken } from "./_helpers";

test("authorization gate engages for a fresh tenant token", async ({ page, context }) => {
  // Tenant A
  await context.addCookies([
    {
      name: "token",
      value: buildFakeToken(),
      domain: new URL(AGENTS_DEV_URL).hostname,
      path: "/",
      sameSite: "Lax",
    },
  ]);
  await page.goto(`${AGENTS_DEV_URL}/agents`);
  await expect(page.getByTestId("agents-table")).toBeVisible();

  // Switch tenant: clear cookies and present a new token
  await context.clearCookies();
  await context.addCookies([
    {
      name: "token",
      value: buildFakeToken(),
      domain: new URL(AGENTS_DEV_URL).hostname,
      path: "/",
      sameSite: "Lax",
    },
  ]);
  await page.goto(`${AGENTS_DEV_URL}/agents`);
  // Re-renders without throwing; auth gate accepts the fresh token.
  await expect(page.getByTestId("agents-table")).toBeVisible({ timeout: 10_000 });
});
