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
  const pending = page.waitForRequest(
    (req) =>
      req.method() === match.method && req.url().includes(match.urlIncludes),
  );
  await action();
  const request = await pending;
  return JSON.parse(request.postData() ?? "{}") as Record<string, any>;
}

/** Reads an endpoint as the master key, so a failure is bad data and not an expired UI token. */
export async function readBack<T = any>(
  page: Page,
  endpoint: string,
): Promise<T> {
  const res = await page.request.get(endpoint, {
    headers: { Authorization: `Bearer ${masterKey()}` },
  });
  expect(res.ok(), `GET ${endpoint}`).toBe(true);
  return (await res.json()) as T;
}

type OperationOutcome =
  | { readonly status: "success" }
  | { readonly status: "failure"; readonly error: unknown };

type RunFailure =
  | { readonly status: "action_failure"; readonly error: unknown }
  | { readonly status: "cleanup_failure"; readonly error: unknown }
  | {
      readonly status: "action_and_cleanup_failure";
      readonly actionError: unknown;
      readonly cleanupError: unknown;
    };

function toRunFailure(
  actionOutcome: OperationOutcome,
  cleanupOutcome: OperationOutcome,
): RunFailure | null {
  if (
    actionOutcome.status === "failure" &&
    cleanupOutcome.status === "failure"
  ) {
    return {
      status: "action_and_cleanup_failure",
      actionError: actionOutcome.error,
      cleanupError: cleanupOutcome.error,
    };
  }
  if (actionOutcome.status === "failure") {
    return { status: "action_failure", error: actionOutcome.error };
  }
  if (cleanupOutcome.status === "failure") {
    return { status: "cleanup_failure", error: cleanupOutcome.error };
  }
  return null;
}

function raiseRunFailure(failure: RunFailure): never {
  switch (failure.status) {
    case "action_failure":
      throw failure.error;
    case "cleanup_failure":
      throw failure.error;
    case "action_and_cleanup_failure":
      throw new AggregateError(
        [failure.actionError, failure.cleanupError],
        "Action and cleanup failed",
      );
  }
}

async function runAction(
  action: () => void | Promise<void>,
): Promise<OperationOutcome> {
  return Promise.resolve()
    .then(action)
    .then(
      () => ({ status: "success" as const }),
      (error: unknown) => ({ status: "failure" as const, error }),
    );
}

async function runCleanup(
  cleanup: () => boolean | Promise<boolean>,
): Promise<OperationOutcome> {
  return Promise.resolve()
    .then(cleanup)
    .then(
      (succeeded) =>
        succeeded
          ? { status: "success" as const }
          : {
              status: "failure" as const,
              error: new Error("Failed to clean up UI E2E resource"),
            },
      (error: unknown) => ({ status: "failure" as const, error }),
    );
}

export async function runWithCleanup(
  action: () => void | Promise<void>,
  cleanup: () => boolean | Promise<boolean>,
): Promise<void> {
  const actionOutcome = await runAction(action);
  const cleanupOutcome = await runCleanup(cleanup);
  const failure = toRunFailure(actionOutcome, cleanupOutcome);
  if (failure !== null) raiseRunFailure(failure);
}
