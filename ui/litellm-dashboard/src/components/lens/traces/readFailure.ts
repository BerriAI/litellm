import { ApiError } from "@/lib/http/client";

import type { FailureCode } from "./types";

const FAILURE_CODES: ReadonlySet<string> = new Set<FailureCode>([
  "invalid_request",
  "invalid_cursor",
  "traversal_expired",
  "traversal_changed",
  "resource_too_large",
  "budget_exceeded",
  "view_not_ready",
  "unavailable",
  "busy",
]);

const RESTART_CODES: ReadonlySet<FailureCode> = new Set<FailureCode>([
  "invalid_cursor",
  "traversal_expired",
  "traversal_changed",
]);

const isFailureCode = (value: unknown): value is FailureCode => typeof value === "string" && FAILURE_CODES.has(value);

/** The stable `detail.code` the proxy attaches to a failed trace read, if the error carries one. */
export const readFailureCode = (error: unknown): FailureCode | null => {
  if (!(error instanceof ApiError) || typeof error.body !== "object" || error.body === null) return null;
  const detail: unknown = (error.body as { detail?: unknown }).detail;
  if (typeof detail !== "object" || detail === null) return null;
  const code: unknown = (detail as { code?: unknown }).code;
  return isFailureCode(code) ? code : null;
};

/** Continuing from the current cursor cannot succeed; the traversal has to start over from a fresh first page. */
export const restartsTraversal = (error: unknown): boolean => {
  const code = readFailureCode(error);
  return code !== null && RESTART_CODES.has(code);
};
