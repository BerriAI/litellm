import { ApiError } from "@/lib/http/client";

/**
 * Why a trace read failed, as the proxy reports it in the `code` of a failed response. Each kind has
 * one recovery: `invalid` and `changed` need a new traversal, `too_large` cannot be paged through at
 * all, `unavailable` is worth retrying after the server's `Retry-After`, and `unknown` is treated like
 * `unavailable` because the loaded pages are still good.
 */
export type TraceReadFailureKind = "invalid" | "changed" | "too_large" | "unavailable" | "unknown";

export interface TraceReadFailure {
  readonly kind: TraceReadFailureKind;
  readonly message: string;
  readonly retryAfterMs: number | null;
}

const KIND_BY_CODE: Readonly<Record<string, TraceReadFailureKind>> = {
  invalid_request: "invalid",
  trace_changed: "changed",
  too_large: "too_large",
  unavailable: "unavailable",
};

const KIND_BY_STATUS: Readonly<Record<number, TraceReadFailureKind>> = {
  400: "invalid",
  409: "changed",
  413: "too_large",
  503: "unavailable",
};

const failureCode = (body: unknown): string | undefined => {
  if (typeof body !== "object" || body === null || !("detail" in body)) return undefined;
  const detail = body.detail;
  if (typeof detail !== "object" || detail === null || !("code" in detail)) return undefined;
  return typeof detail.code === "string" ? detail.code : undefined;
};

export function classifyTraceReadFailure(error: unknown): TraceReadFailure {
  if (!(error instanceof ApiError)) {
    return { kind: "unknown", message: error instanceof Error ? error.message : String(error), retryAfterMs: null };
  }
  const code = failureCode(error.body);
  const kind = (code && KIND_BY_CODE[code]) || KIND_BY_STATUS[error.status] || "unknown";
  return { kind, message: error.message, retryAfterMs: error.retryAfterMs };
}

/** Only an outage is worth retrying without the user; a changed or invalid cursor fails the same way every time. */
export const isRetryableTraceRead = (failure: TraceReadFailure): boolean =>
  failure.kind === "unavailable" || failure.kind === "unknown";

export const TRACE_READ_AUTO_RETRIES = 2;
const DEFAULT_RETRY_DELAY_MS = 1_000;

export const traceReadRetry = (failureCount: number, error: unknown): boolean =>
  failureCount < TRACE_READ_AUTO_RETRIES && classifyTraceReadFailure(error).kind === "unavailable";

export const traceReadRetryDelay = (_attempt: number, error: unknown): number =>
  classifyTraceReadFailure(error).retryAfterMs ?? DEFAULT_RETRY_DELAY_MS;
