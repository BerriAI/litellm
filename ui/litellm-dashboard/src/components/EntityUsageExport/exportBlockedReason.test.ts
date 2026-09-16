import { describe, expect, it } from "vitest";

import type { DailyData } from "@/components/UsagePage/types";

import { getApiKeyLimitReached, getExportBlockedReason, type UsageFetchState } from "./exportBlockedReason";

const state = (overrides: Partial<UsageFetchState> = {}): UsageFetchState => ({
  coversRange: true,
  cancelled: false,
  failed: false,
  apiKeyLimitReached: undefined,
  ...overrides,
});

const dayWithKeys = (date: string, ...keys: string[]): DailyData =>
  ({
    date,
    metrics: {},
    breakdown: { api_keys: Object.fromEntries(keys.map((k) => [k, { metrics: {}, metadata: {} }])) },
  }) as unknown as DailyData;

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

  it("blocks when the aggregated endpoint hit its key cap, since a per-team CSV would miss keys", () => {
    const reason = getExportBlockedReason(state({ apiKeyLimitReached: 100 }));

    expect(reason).toMatch(/100 highest-spend keys/);
    expect(reason).toMatch(/USAGE_TOP_API_KEYS_LIMIT/);
  });
});

describe("getApiKeyLimitReached", () => {
  it("reports the cap once the distinct keys across every day reach it", () => {
    const results = [dayWithKeys("2026-06-01", "key-1", "key-2"), dayWithKeys("2026-06-02", "key-2", "key-3")];

    expect(getApiKeyLimitReached(results, 3)).toBe(3);
  });

  it("stays quiet while fewer keys than the cap came back, which means every key is on screen", () => {
    const results = [dayWithKeys("2026-06-01", "key-1", "key-2"), dayWithKeys("2026-06-02", "key-2")];

    expect(getApiKeyLimitReached(results, 3)).toBeUndefined();
  });

  it("stays quiet when the response carries no cap, as the paginated fallback does", () => {
    expect(getApiKeyLimitReached([dayWithKeys("2026-06-01", "key-1")], undefined)).toBeUndefined();
  });
});
