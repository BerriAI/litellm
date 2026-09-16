import { act, renderHook, waitFor } from "@testing-library/react";
import { NuqsTestingAdapter, type OnUrlUpdateFunction } from "nuqs/adapters/testing";
import type { ReactNode } from "react";
import { describe, expect, it, vi } from "vitest";

const mockUsePaginatedDailyActivity = vi.fn();

const mockCancel = vi.fn();

vi.mock("@/app/(dashboard)/usage/_components/hooks/usePaginatedDailyActivity", () => ({
  usePaginatedDailyActivity: (args: unknown) => {
    mockUsePaginatedDailyActivity(args);
    return {
      data: { results: [] },
      loading: false,
      isFetchingMore: false,
      progress: { currentPage: 4, totalPages: 9 },
      cancelled: false,
      cancel: mockCancel,
    };
  },
}));

vi.mock("@/components/networking", () => ({
  userDailyActivityCall: vi.fn(),
  userDailyActivityAggregatedCall: vi.fn(),
}));

import { userDailyActivityAggregatedCall } from "@/components/networking";
import { useActivityDateRange, useDailyActivityRange, useUrlActivityDateRange } from "./useDailyActivityRange";

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
});

const THIRTY_DAYS_MS = 30 * 24 * 60 * 60 * 1000;
const URL_KEYS = { start: "start_date", end: "end_date" };

const renderUrlRange = (searchParams = "", onUrlUpdate?: OnUrlUpdateFunction) =>
  renderHook(() => useUrlActivityDateRange(URL_KEYS), {
    wrapper: ({ children }: { children: ReactNode }) => (
      <NuqsTestingAdapter searchParams={searchParams} onUrlUpdate={onUrlUpdate} hasMemory>
        {children}
      </NuqsTestingAdapter>
    ),
  });

describe("useUrlActivityDateRange", () => {
  it("reads the range from the URL as whole local days", () => {
    const { result } = renderUrlRange("?start_date=2026-01-05&end_date=2026-01-07");

    expect(result.current.dateValue.from).toEqual(new Date(2026, 0, 5));
    expect(result.current.dateValue.to).toEqual(new Date(2026, 0, 7, 23, 59, 59, 999));
  });

  it.each([
    ["an impossible calendar day", "?start_date=2026-02-30&end_date=2026-03-02"],
    ["a start after the end", "?start_date=2026-03-02&end_date=2026-03-01"],
    ["a missing end", "?start_date=2026-03-01"],
    ["a non-date value", "?start_date=yesterday&end_date=2026-03-01"],
  ])("falls back to the trailing 30 days for %s", (_label, searchParams) => {
    const before = Date.now();
    const { result } = renderUrlRange(searchParams);
    const { from, to } = result.current.dateValue;

    expect(to!.getTime()).toBeGreaterThanOrEqual(before);
    expect(to!.getTime()).toBeLessThanOrEqual(Date.now());
    expect(to!.getTime() - from!.getTime()).toBe(THIRTY_DAYS_MS);
  });

  it("keeps the same range object across rerenders so date-keyed effects do not refire", () => {
    const fromUrl = renderUrlRange("?start_date=2026-01-05&end_date=2026-01-07");
    const firstUrlValue = fromUrl.result.current.dateValue;
    fromUrl.rerender();
    expect(fromUrl.result.current.dateValue).toBe(firstUrlValue);

    const fallback = renderUrlRange();
    const firstFallback = fallback.result.current.dateValue;
    fallback.rerender();
    expect(fallback.result.current.dateValue).toBe(firstFallback);
  });

  it("writes the picked local days to the URL keys and reflects them back", async () => {
    const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
    const { result } = renderUrlRange("", onUrlUpdate);

    act(() => result.current.onDateChange({ from: new Date(2026, 3, 1), to: new Date(2026, 3, 9, 23, 59, 59) }));

    await waitFor(() => expect(onUrlUpdate).toHaveBeenCalled());
    const { searchParams } = onUrlUpdate.mock.calls.at(-1)![0];
    expect(searchParams.get("start_date")).toBe("2026-04-01");
    expect(searchParams.get("end_date")).toBe("2026-04-09");
    expect(result.current.dateValue.from).toEqual(new Date(2026, 3, 1));
    expect(result.current.dateValue.to).toEqual(new Date(2026, 3, 9, 23, 59, 59, 999));
  });

  it("removes both URL keys when the range is cleared", async () => {
    const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
    const { result } = renderUrlRange("?start_date=2026-01-05&end_date=2026-01-07", onUrlUpdate);

    act(() => result.current.onDateChange({ from: undefined, to: undefined }));

    await waitFor(() => expect(onUrlUpdate).toHaveBeenCalled());
    const { searchParams } = onUrlUpdate.mock.calls.at(-1)![0];
    expect(searchParams.has("start_date")).toBe(false);
    expect(searchParams.has("end_date")).toBe(false);
  });

  it("drives the daily-activity query when handed to useDailyActivityRange", () => {
    renderHook(() => useDailyActivityRange("test-token", "u1", "proxy_admin", useUrlActivityDateRange(URL_KEYS)), {
      wrapper: ({ children }: { children: ReactNode }) => (
        <NuqsTestingAdapter searchParams="?start_date=2026-01-05&end_date=2026-01-07">{children}</NuqsTestingAdapter>
      ),
    });

    expect(argsOfLastCall()).toEqual([
      "test-token",
      new Date(2026, 0, 5),
      new Date(2026, 0, 7, 23, 59, 59, 999),
      null,
      true,
      null,
    ]);
  });
});
