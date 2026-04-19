/**
 * Meta-test — verifies the `guardedPage` fixture actually *fails* tests when
 * the browser issues a forbidden request.
 *
 * Playwright fixtures teardown after the test body, so we cannot observe the
 * fixture's `expect()` failure from inside the test. Instead, we structure
 * this spec so the meta-test harness flips Playwright's `test.fail()` state:
 *
 *   - The test body intentionally fires a `/ui/ui/...` fetch.
 *   - Because the fixture's teardown-time assertion will raise, the test is
 *     expected to fail.
 *   - We use Playwright's `test.fail()` annotation to invert that expectation
 *     — the spec *passes* when the body-+-teardown combined result is a
 *     failure.
 *
 * If someone breaks the fixture (e.g. removes the expect() call, drops the
 * request listener) this test will flip to "passed when it should have
 * failed" and the spec will go red. That is the signal we want.
 */

import { test } from "../../fixtures/guarded-page";
import * as http from "http";
import type { AddressInfo } from "net";

test.describe("guardedPage fixture — regression trigger", () => {
  // Stand up a tiny server shared across the triggers. Kept minimal so the
  // trace is easy to read if a regression fires.
  let server: http.Server;
  let port: number;

  test.beforeAll(async () => {
    server = http.createServer((_req, res) => {
      res.writeHead(200, { "Content-Type": "application/json" });
      res.end("{}");
    });
    await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
    port = (server.address() as AddressInfo).port;
  });

  test.afterAll(() => {
    server.close();
  });

  test("fires when a /ui/ui/ XHR is made", async ({ page }) => {
    test.fail(true, "intentionally triggers the fixture's URL guard");
    await page.goto(`http://127.0.0.1:${port}/`);
    await page.evaluate(async (p) => {
      await fetch(`http://127.0.0.1:${p}/ui/ui/project/list`);
    }, port);
  });

  test("fires when an /ui/<api-verb> XHR is made", async ({ page }) => {
    test.fail(true, "intentionally triggers the fixture's URL guard");
    await page.goto(`http://127.0.0.1:${port}/`);
    await page.evaluate(async (p) => {
      await fetch(`http://127.0.0.1:${p}/ui/key/info`);
    }, port);
  });
});
