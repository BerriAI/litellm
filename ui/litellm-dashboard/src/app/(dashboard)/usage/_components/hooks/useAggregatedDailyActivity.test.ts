import { act, renderHook, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import {
  EMPTY_DAILY_ACTIVITY_METADATA,
  type DailyActivityAggregatedResponse,
} from "@/components/UsagePage/dailyActivityApi";
import { useAggregatedDailyActivity } from "./useAggregatedDailyActivity";

const response = (spend: number): DailyActivityAggregatedResponse => ({
  results: [],
  metadata: { ...EMPTY_DAILY_ACTIVITY_METADATA, total_spend: spend },
});

describe("useAggregatedDailyActivity", () => {
  it("calls fetch once per deps change and exposes the data", async () => {
    const fetch = vi.fn().mockResolvedValue(response(12));
    const { result, rerender } = renderHook(
      ({ dep }: { dep: string }) => useAggregatedDailyActivity({ fetch, enabled: true, deps: [dep] }),
      { initialProps: { dep: "a" } },
    );

    await waitFor(() => expect(result.current.loading).toBe(false));
    expect(fetch).toHaveBeenCalledTimes(1);
    expect(result.current.data.metadata?.total_spend).toBe(12);

    rerender({ dep: "b" });
    await waitFor(() => expect(fetch).toHaveBeenCalledTimes(2));
  });

  it("does not fetch while disabled and returns empty data", () => {
    const fetch = vi.fn();
    const { result } = renderHook(() => useAggregatedDailyActivity({ fetch, enabled: false, deps: ["a"] }));

    expect(fetch).not.toHaveBeenCalled();
    expect(result.current.data.results).toEqual([]);
    expect(result.current.loading).toBe(false);
    expect(result.current.failed).toBe(false);
  });

  it("exposes failed on rejection", async () => {
    const fetch = vi.fn().mockRejectedValue(new Error("boom"));
    const { result } = renderHook(() => useAggregatedDailyActivity({ fetch, enabled: true, deps: ["a"] }));

    await waitFor(() => expect(result.current.failed).toBe(true));
    expect(result.current.loading).toBe(false);
  });

  it("ignores an out-of-order resolution from the previous deps", async () => {
    let resolveFirst: ((value: DailyActivityAggregatedResponse) => void) | undefined;
    const first = new Promise<DailyActivityAggregatedResponse>((resolve) => {
      resolveFirst = resolve;
    });
    const fetch = vi
      .fn()
      .mockImplementationOnce(() => first)
      .mockResolvedValueOnce(response(99));

    const { result, rerender } = renderHook(
      ({ dep }: { dep: string }) => useAggregatedDailyActivity({ fetch, enabled: true, deps: [dep] }),
      { initialProps: { dep: "a" } },
    );

    rerender({ dep: "b" });
    await act(async () => {
      resolveFirst?.(response(1));
    });
    await waitFor(() => expect(result.current.loading).toBe(false));
    expect(result.current.data.metadata?.total_spend).toBe(99);
  });
});
