import { describe, expect, it } from "vitest";

import { getApiKeyTruncation, getExportBlockedReason, type UsageFetchState } from "./exportBlockedReason";

const state = (overrides: Partial<UsageFetchState> = {}): UsageFetchState => ({
  coversRange: true,
  cancelled: false,
  failed: false,
  apiKeyTruncation: undefined,
  ...overrides,
});

describe("getExportBlockedReason", () => {
  it("lets the export through once the data on screen covers the range", () => {
    expect(getExportBlockedReason(state())).toBeUndefined();
  });

  it("blocks whenever the data on screen does not cover the range, which is when a CSV silently under-reports", () => {
    expect(getExportBlockedReason(state({ coversRange: false }))).toMatch(/still loading/i);
  });

  it("blocks after a stopped fetch and says a reload is what fixes it", () => {
    const reason = getExportBlockedReason(state({ coversRange: false, cancelled: true }));

    expect(reason).toMatch(/stopped/i);
    expect(reason).toMatch(/reload/i);
  });

  it("blocks after a failed page and names the failure rather than the stop", () => {
    const reason = getExportBlockedReason(state({ coversRange: false, failed: true, cancelled: true }));

    expect(reason).toMatch(/failed to load/i);
    expect(reason).not.toMatch(/stopped/i);
  });

  it("blocks when the aggregated endpoint dropped keys, since a per-team CSV would miss them", () => {
    const reason = getExportBlockedReason(state({ apiKeyTruncation: { limit: 100, total: 3000 } }));

    expect(reason).toMatch(/100 highest-spend keys of 3000/);
    expect(reason).toMatch(/a per-team export/);
    expect(reason).toMatch(/USAGE_TOP_API_KEYS_LIMIT/);
  });

  it("names whatever export the caller passes instead of assuming a team", () => {
    const reason = getExportBlockedReason(state({ apiKeyTruncation: { limit: 100, total: 3000 } }), "the export");

    expect(reason).toMatch(/so the export would under-report/);
    expect(reason).not.toMatch(/team/);
  });

  it("does not block on truncation when a server export will cover every key", () => {
    expect(getExportBlockedReason(state({ apiKeyTruncation: null }))).toBeUndefined();
  });
});

describe("getApiKeyTruncation", () => {
  it("reports truncation once the proxy saw more keys than it returned", () => {
    expect(getApiKeyTruncation(100, 101)).toEqual({ limit: 100, total: 101 });
  });

  it("stays quiet when exactly the cap exists, since every key is on screen", () => {
    expect(getApiKeyTruncation(100, 100)).toBeUndefined();
    expect(getApiKeyTruncation(100, 7)).toBeUndefined();
  });

  it("stays quiet when the response carries no cap, as the paginated fallback does", () => {
    expect(getApiKeyTruncation(undefined, undefined)).toBeUndefined();
    expect(getApiKeyTruncation(100, null)).toBeUndefined();
  });
});
