import { expect, Page } from "@playwright/test";
import { masterKey } from "./traffic";

/**
 * Assert-the-round-trip helpers.
 *
 * WHY THESE EXIST: most mutation specs in this suite stop at a success toast,
 * and a success toast is exactly the thing that lies. The recurring customer
 * report is a form that says "Saved!" and then either no-ops or clobbers an
 * unrelated field -- a toast-only assertion passes in both cases.
 *
 * A mutation is verified when two things hold:
 *   1. the UI put the intended change on the wire, and
 *   2. the change is still there when read back from the API.
 *
 * (2) alone misses "the UI sent the right thing but the backend dropped it";
 * (1) alone misses "the request was right but nothing persisted". Assert both.
 *
 * The idiom is lifted from the three specs that already do this correctly --
 * modelsPage/clearCustomPricing.spec.ts, modelsPage/credentials.spec.ts and
 * settings/routerSettings.spec.ts -- and is only factored out here so the rest
 * of the suite stops reinventing it.
 */

/**
 * Runs `action` and returns the parsed body of the first matching request it
 * causes.
 *
 * `action` is a callback rather than something you await beforehand on purpose:
 * the listener has to be armed BEFORE the click, and doing that by hand is the
 * easy mistake -- await the click first and the request has already gone by the
 * time you start waiting, so the test hangs until timeout for a reason that
 * looks nothing like the real cause.
 */
export async function captureRequestBody(
  page: Page,
  match: { method: string; urlIncludes: string },
  action: () => Promise<void>,
): Promise<Record<string, any>> {
  const pending = page.waitForRequest((req) => req.method() === match.method && req.url().includes(match.urlIncludes));
  await action();
  const request = await pending;
  return JSON.parse(request.postData() ?? "{}") as Record<string, any>;
}

/**
 * Reads an endpoint back through the management API, authenticated as the
 * master key rather than the browser session -- so a read-back failure is
 * unambiguously "the data is wrong", never "the UI's token expired".
 */
export async function readBack<T = any>(page: Page, endpoint: string): Promise<T> {
  const res = await page.request.get(endpoint, {
    headers: { Authorization: `Bearer ${masterKey()}` },
  });
  expect(res.ok(), `GET ${endpoint}`).toBe(true);
  return (await res.json()) as T;
}
