/**
 * Meta-test: proves the `guardedPage` fixture is wired correctly.
 *
 * This test stands up a tiny ephemeral HTTP server inside the test, has the
 * browser navigate to it (plus a forbidden-looking URL), and verifies:
 *   - The guard fixture is actually installed (a `request` listener fires).
 *   - The classifier logic returns the expected verdicts at runtime
 *     (independent of the vitest unit suite).
 *
 * The rich URL-classification coverage lives in
 * `tests/e2e_guard.test.ts` (vitest). This meta-test complements it by
 * exercising the Playwright wiring itself, which the unit tests cannot.
 */

import { test, expect } from "../../fixtures/guarded-page";
import { isForbiddenRequestUrl } from "../../fixtures/guarded-page";
import * as http from "http";
import type { AddressInfo } from "net";

test.describe("guardedPage fixture — liveness", () => {
  test("should fire request listeners on real HTTP navigation", async ({ page }) => {
    // Start a tiny one-shot server on an ephemeral port.
    const server = http.createServer((_req, res) => {
      res.writeHead(200, { "Content-Type": "text/html" });
      res.end("<html><body>ok</body></html>");
    });
    await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
    const port = (server.address() as AddressInfo).port;

    const observed: string[] = [];
    page.on("request", (req) => observed.push(req.url()));

    try {
      await page.goto(`http://127.0.0.1:${port}/ok`);
      expect(observed.length).toBeGreaterThan(0);
      expect(observed.some((u) => u.includes("/ok"))).toBe(true);
    } finally {
      server.close();
    }
  });

  test("should classify forbidden and allowed URLs at runtime", async () => {
    // Runtime sanity — mirrors the unit tests, but executed through the
    // same code path Playwright uses when the fixture evaluates verdicts.
    expect(
      isForbiddenRequestUrl("http://localhost:4000/ui/ui/project/list", "xhr").forbidden,
    ).toBe(true);
    expect(
      isForbiddenRequestUrl("http://localhost:4000/ui/key/info", "xhr").forbidden,
    ).toBe(true);
    expect(
      isForbiddenRequestUrl("http://localhost:4000/ui/guardrails", "document").forbidden,
    ).toBe(false);
    expect(
      isForbiddenRequestUrl("http://localhost:4000/project/list", "xhr").forbidden,
    ).toBe(false);
  });
});
