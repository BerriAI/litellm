import { act, renderHook } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

const mockUsePaginatedDailyActivity = vi.fn();

const mockCancel = vi.fn();
let mockReportingTimezone: string | undefined;
let mockMetadata: Record<string, number> = {};

vi.mock("@/app/(dashboard)/usage/_components/hooks/usePaginatedDailyActivity", () => ({
  usePaginatedDailyActivity: (args: unknown) => {
    mockUsePaginatedDailyActivity(args);
    return {
      data: { results: [], metadata: mockMetadata },
      loading: false,
      isFetchingMore: false,
      progress: { currentPage: 4, totalPages: 9 },
      cancelled: false,
      failed: false,
      coversRange: true,
      cancel: mockCancel,
    };
  },
}));

vi.mock("@/components/networking", () => ({
  userDailyActivityCall: vi.fn(),
  userDailyActivityAggregatedCall: vi.fn(),
}));

import { userDailyActivityAggregatedCall } from "@/components/networking";
import { useActivityDateRange, useDailyActivityRange } from "./useDailyActivityRange";

const argsOfLastCall = () => mockUsePaginatedDailyActivity.mock.calls.at(-1)?.[0].args as unknown[];

describe("useDailyActivityRange", () => {
  it("offers date-range state without starting a daily-activity query", () => {
    const { result } = renderHook(() => useActivityDateRange());

    expect(result.current.dateValue.from).toBeInstanceOf(Date);
    expect(result.current.dateValue.to).toBeInstanceOf(Date);
    expect(mockUsePaginatedDailyActivity).not.toHaveBeenCalled();
  });

  it("queries every user's activity for an admin", () => {
    renderHook(() => useDailyActivityRange("test-token", "u1", "proxy_admin"));

    expect(argsOfLastCall()).toEqual(["test-token", expect.any(Date), expect.any(Date), null, true, null]);
  });

  it("scopes the query to the caller for a non-admin", () => {
    renderHook(() => useDailyActivityRange("test-token", "u1", "internal_user"));

    expect(argsOfLastCall()).toEqual(["test-token", expect.any(Date), expect.any(Date), "u1", true, null]);
  });

  it.each(["org_admin", "Org Admin"])(
    "scopes the query to the caller for %s, who has no admin view on this endpoint",
    (role) => {
      renderHook(() => useDailyActivityRange("test-token", "u1", role));

      expect(argsOfLastCall()).toEqual(["test-token", expect.any(Date), expect.any(Date), "u1", true, null]);
    },
  );

  it("fetches through the single-shot aggregated endpoint first so days never fragment across pages", () => {
    renderHook(() => useDailyActivityRange("test-token", "u1", "proxy_admin"));

    expect(mockUsePaginatedDailyActivity).toHaveBeenLastCalledWith(
      expect.objectContaining({ aggregatedFetchFn: userDailyActivityAggregatedCall }),
    );
  });

  it("forwards the pagination progress and cancel affordances instead of dropping them", () => {
    const { result } = renderHook(() => useDailyActivityRange("test-token", "u1", "proxy_admin"));

    expect(result.current.progress).toEqual({ currentPage: 4, totalPages: 9 });
    expect(result.current.cancelled).toBe(false);
    expect(result.current.cancel).toBe(mockCancel);
  });

  it("stays disabled until an access token is available", () => {
    renderHook(() => useDailyActivityRange(null, "u1", "proxy_admin"));

    expect(mockUsePaginatedDailyActivity).toHaveBeenLastCalledWith(expect.objectContaining({ enabled: false }));
  });

  it("reports how many keys the proxy left out of the per-key lists", () => {
    mockMetadata = { api_key_limit: 100, total_api_keys: 3000 };
    const { result } = renderHook(() => useDailyActivityRange("test-token", "u1", "proxy_admin"));

    expect(result.current.apiKeyTruncation).toEqual({ limit: 100, total: 3000 });
  });

  it("reports no key truncation when every key fit under the proxy limit", () => {
    mockMetadata = { api_key_limit: 100, total_api_keys: 100 };
    const { result } = renderHook(() => useDailyActivityRange("test-token", "u1", "proxy_admin"));

    expect(result.current.apiKeyTruncation).toBeUndefined();
  });
});

vi.mock("@/app/(dashboard)/hooks/uiSettings/useUISettings", () => ({
  useDailyUsageTimezone: () => mockReportingTimezone,
}));

it("loads reporting defaults without overwriting a manually chosen range", () => {
  vi.useFakeTimers();
  vi.setSystemTime(new Date("2026-09-25T17:00:00Z"));
  try {
    const { result, rerender } = renderHook(() => useActivityDateRange());
    mockReportingTimezone = "Asia/Singapore";
    rerender();
    expect(result.current.dateValue.to).toEqual(new Date(2026, 8, 26));
    const selected = { from: new Date(2026, 7, 1), to: new Date(2026, 7, 31) };
    act(() => result.current.onDateChange(selected));
    mockReportingTimezone = "America/Los_Angeles";
    rerender();
    expect(result.current.dateValue).toEqual(selected);
  } finally {
    mockReportingTimezone = undefined;
    vi.useRealTimers();
  }
});
