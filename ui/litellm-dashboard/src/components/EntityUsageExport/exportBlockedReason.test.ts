import { describe, expect, it } from "vitest";

import { getExportBlockedReason, type UsageFetchState } from "./exportBlockedReason";

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
