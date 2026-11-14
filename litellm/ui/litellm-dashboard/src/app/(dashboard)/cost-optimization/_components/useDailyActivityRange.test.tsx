import { NuqsTestingAdapter } from "nuqs/adapters/testing";
import { act, renderHook as testingRenderHook, waitFor } from "@testing-library/react";
import { type PropsWithChildren } from "react";
import { describe, expect, it, vi } from "vitest";

import { EMPTY_DAILY_ACTIVITY_METADATA } from "@/components/UsagePage/dailyActivityApi";

const mockUseAggregatedDailyActivity = vi.fn();
const renderHook: typeof testingRenderHook = (render, options) =>
  testingRenderHook(render, { ...options, wrapper: NuqsTestingAdapter });

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

it("opens the alert's complete UTC date window", () => {
  const { result } = testingRenderHook(() => useActivityDateRange(), {
    wrapper: ({ children }) => (
      <NuqsTestingAdapter searchParams="activity_from=2026-01-01&activity_to=2026-01-08">{children}</NuqsTestingAdapter>
    ),
  });
  expect(result.current.dateValue.from?.getDate()).toBe(1);
  expect(result.current.dateValue.to?.getDate()).toBe(8);
  expect(result.current.dateValue.from?.getFullYear()).toBe(2026);
});

it("updates an already mounted Activity view when another alert changes the URL dates", async () => {
  let searchParams = "activity_from=2026-01-01&activity_to=2026-01-08";
  const wrapper = ({ children }: PropsWithChildren) => (
    <NuqsTestingAdapter searchParams={searchParams} hasMemory>
      {children}
    </NuqsTestingAdapter>
  );
  const { result, rerender } = testingRenderHook(() => useActivityDateRange(), { wrapper });
  searchParams = "activity_from=2026-02-01&activity_to=2026-02-08";
  rerender();
  await waitFor(() => expect(result.current.dateValue.from).toEqual(new Date(2026, 1, 1, 12)));
  expect(result.current.dateValue.to).toEqual(new Date(2026, 1, 8, 12));
});

it("updates the dates when the user changes the date picker", async () => {
  const wrapper = ({ children }: PropsWithChildren) => (
    <NuqsTestingAdapter searchParams="activity_from=2026-01-01&activity_to=2026-01-08" hasMemory>
      {children}
    </NuqsTestingAdapter>
  );
  const { result } = testingRenderHook(() => useActivityDateRange(), { wrapper });
  await act(async () => result.current.onDateChange({ from: new Date(2026, 2, 1, 12), to: new Date(2026, 2, 8, 12) }));
  expect(result.current.dateValue.from).toEqual(new Date(2026, 2, 1, 12));
});
