import { describe, expect, it } from "vitest";

import { getExportBlockedReason, type UsageFetchState } from "./exportBlockedReason";

const state = (overrides: Partial<UsageFetchState> = {}): UsageFetchState => ({
  loading: false,
  isFetchingMore: false,
  cancelled: false,
  failed: false,
  ...overrides,
});

describe("getExportBlockedReason", () => {
  it("lets the export through once the range has fully loaded", () => {
    expect(getExportBlockedReason(state())).toBeUndefined();
  });

  it("blocks the first load, before any page has arrived", () => {
    expect(getExportBlockedReason(state({ loading: true }))).toMatch(/still loading/i);
  });

  it("blocks while later pages are still arriving, which is when a CSV silently under-reports", () => {
    expect(getExportBlockedReason(state({ isFetchingMore: true }))).toMatch(/still loading/i);
  });

  it("blocks after a stopped fetch and says a reload is what fixes it", () => {
    const reason = getExportBlockedReason(state({ cancelled: true }));

    expect(reason).toMatch(/stopped/i);
    expect(reason).toMatch(/reload/i);
  });

  it("blocks after a failed page and names the failure rather than the stop", () => {
    const reason = getExportBlockedReason(state({ failed: true, cancelled: true }));

    expect(reason).toMatch(/failed to load/i);
    expect(reason).not.toMatch(/stopped/i);
  });
});
