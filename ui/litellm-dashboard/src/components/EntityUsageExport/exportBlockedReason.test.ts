import { describe, expect, it } from "vitest";

import { getApiKeyTruncation, getExportBlockedReason, type UsageFetchState } from "./exportBlockedReason";

const state = (overrides: Partial<UsageFetchState> = {}): UsageFetchState => ({
  coversRange: true,
  cancelled: false,
  failed: false,
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
