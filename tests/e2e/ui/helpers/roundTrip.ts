import { expect, Page } from "@playwright/test";
import { masterKey } from "./traffic";

/**
 * Runs `action` and returns the parsed body of the first matching request.
 *
 * `action` is a callback so the listener is armed before the click; awaiting the
 * click first lets the request go by, and the test then hangs until timeout.
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

/** Reads an endpoint as the master key, so a failure is bad data and not an expired UI token. */
export async function readBack<T = any>(page: Page, endpoint: string): Promise<T> {
  const res = await page.request.get(endpoint, {
    headers: { Authorization: `Bearer ${masterKey()}` },
  });
  expect(res.ok(), `GET ${endpoint}`).toBe(true);
  return (await res.json()) as T;
}
