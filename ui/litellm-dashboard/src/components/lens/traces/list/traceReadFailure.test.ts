import { describe, expect, it } from "vitest";

import { ApiError } from "@/lib/http/client";

import {
  classifyTraceReadFailure,
  isRetryableTraceRead,
  TRACE_READ_AUTO_RETRIES,
  traceReadRetry,
  traceReadRetryDelay,
} from "./traceReadFailure";

const failure = (status: number, code?: string, retryAfterMs: number | null = null) =>
  new ApiError("boom", status, { detail: code ? { code, message: "boom" } : "boom" }, retryAfterMs);

describe("classifyTraceReadFailure", () => {
  it.each([
    ["invalid_request", "invalid"],
    ["trace_changed", "changed"],
    ["too_large", "too_large"],
    ["unavailable", "unavailable"],
  ] as const)("reads the %s code ahead of the status", (code, kind) => {
    expect(classifyTraceReadFailure(failure(500, code)).kind).toBe(kind);
  });

  it.each([
    [400, "invalid"],
    [409, "changed"],
    [413, "too_large"],
    [503, "unavailable"],
    [500, "unknown"],
  ] as const)("falls back to status %i without a code", (status, kind) => {
    expect(classifyTraceReadFailure(failure(status)).kind).toBe(kind);
  });

  it("treats a non-HTTP error as unknown and keeps its message", () => {
    expect(classifyTraceReadFailure(new Error("offline"))).toEqual({
      kind: "unknown",
      message: "offline",
      retryAfterMs: null,
    });
  });
});

describe("retry policy", () => {
  it("retries an outage automatically after the server's Retry-After and stops at the cap", () => {
    const outage = failure(503, "unavailable", 2_000);
    expect(traceReadRetry(0, outage)).toBe(true);
    expect(traceReadRetry(TRACE_READ_AUTO_RETRIES, outage)).toBe(false);
    expect(traceReadRetryDelay(0, outage)).toBe(2_000);
    expect(traceReadRetryDelay(0, failure(503, "unavailable"))).toBe(1_000);
  });

  it("never retries a changed, invalid, or oversized read", () => {
    for (const code of ["trace_changed", "invalid_request", "too_large"]) {
      expect(traceReadRetry(0, failure(500, code))).toBe(false);
      expect(isRetryableTraceRead(classifyTraceReadFailure(failure(500, code)))).toBe(false);
    }
    expect(isRetryableTraceRead(classifyTraceReadFailure(new Error("offline")))).toBe(true);
  });
});
