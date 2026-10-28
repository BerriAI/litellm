import { renderHook } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { EMPTY_DAILY_ACTIVITY_METADATA } from "@/components/UsagePage/dailyActivityApi";

const mockUseAggregatedDailyActivity = vi.fn();

vi.mock("@/app/(dashboard)/usage/_components/hooks/useAggregatedDailyActivity", () => ({
  useAggregatedDailyActivity: (options: unknown) => {
    mockUseAggregatedDailyActivity(options);
    return {
      data: { results: [], metadata: EMPTY_DAILY_ACTIVITY_METADATA },
      loading: false,
      failed: false,
    };
  },
}));

vi.mock("@/components/networking", () => ({
  dailyActivityAggregatedCall: vi.fn().mockResolvedValue({ results: [], metadata: {} }),
}));

import { dailyActivityAggregatedCall } from "@/components/networking";
import { useActivityDateRange, useDailyActivityRange } from "./useDailyActivityRange";

interface CapturedOptions {
  fetch: () => Promise<unknown>;
  enabled: boolean;
  deps: unknown[];
}

const lastOptions = () => mockUseAggregatedDailyActivity.mock.calls.at(-1)?.[0] as CapturedOptions;

describe("useDailyActivityRange", () => {
  it("offers date-range state without starting a daily-activity query", () => {
    const { result } = renderHook(() => useActivityDateRange());

    expect(result.current.dateValue.from).toBeInstanceOf(Date);
    expect(result.current.dateValue.to).toBeInstanceOf(Date);
    expect(mockUseAggregatedDailyActivity).not.toHaveBeenCalled();
  });

  it("fetches every user's activity for an admin through the aggregated endpoint", async () => {
    renderHook(() => useDailyActivityRange("test-token", "u1", "proxy_admin"));

    await lastOptions().fetch();
    expect(dailyActivityAggregatedCall).toHaveBeenCalledWith(
      "user",
      expect.objectContaining({
        accessToken: "test-token",
        entityIds: null,
        includeCurrentUtcDay: true,
      }),
    );
  });

  it("scopes the query to the caller for a non-admin", async () => {
    renderHook(() => useDailyActivityRange("test-token", "u1", "internal_user"));

    await lastOptions().fetch();
    expect(dailyActivityAggregatedCall).toHaveBeenCalledWith("user", expect.objectContaining({ entityIds: ["u1"] }));
  });

  it.each(["org_admin", "Org Admin"])(
    "scopes the query to the caller for %s, who has no admin view on this endpoint",
    async (role) => {
      renderHook(() => useDailyActivityRange("test-token", "u1", role));

      await lastOptions().fetch();
      expect(dailyActivityAggregatedCall).toHaveBeenCalledWith("user", expect.objectContaining({ entityIds: ["u1"] }));
    },
  );

  it("stays disabled until an access token is available", () => {
    renderHook(() => useDailyActivityRange(null, "u1", "proxy_admin"));

    expect(lastOptions().enabled).toBe(false);
  });

  it("exposes the request scope so sibling hooks fetch under the same filters", () => {
    const { result } = renderHook(() => useDailyActivityRange("test-token", "u1", "internal_user"));

    expect(result.current.scope).toMatchObject({
      accessToken: "test-token",
      userId: "u1",
      apiKey: null,
    });
  });
});
