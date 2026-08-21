import { renderHook } from "@testing-library/react";
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
import { useDailyActivityRange } from "./useDailyActivityRange";

const argsOfLastCall = () => mockUsePaginatedDailyActivity.mock.calls.at(-1)?.[0].args as unknown[];

describe("useDailyActivityRange", () => {
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
