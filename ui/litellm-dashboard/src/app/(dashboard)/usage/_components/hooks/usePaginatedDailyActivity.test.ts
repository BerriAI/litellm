import { renderHook, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { usePaginatedDailyActivity } from "./usePaginatedDailyActivity";

const page = (date: string, spend: number, totalPages: number, pageNumber: number) => ({
  results: [
    {
      date,
      metrics: {
        spend,
        prompt_tokens: 0,
        completion_tokens: 0,
        total_tokens: spend * 10,
        api_requests: 1,
        successful_requests: 1,
        failed_requests: 0,
        cache_read_input_tokens: 0,
        cache_creation_input_tokens: 0,
      },
      breakdown: {
        models: {},
        model_groups: {},
        mcp_servers: {},
        providers: {},
        api_keys: {},
        entities: {
          "team-1": {
            metrics: {
              spend,
              prompt_tokens: 0,
              completion_tokens: 0,
              total_tokens: spend * 10,
              api_requests: 1,
              successful_requests: 1,
              failed_requests: 0,
              cache_read_input_tokens: 0,
              cache_creation_input_tokens: 0,
            },
            metadata: {},
            api_key_breakdown: {},
          },
        },
      },
    },
  ],
  metadata: { total_spend: spend, total_tokens: spend * 10, total_pages: totalPages, page: pageNumber },
});

const args = ["token", new Date("2026-02-05"), new Date("2026-08-05"), null];

describe("usePaginatedDailyActivity", () => {
  beforeEach(() => {
    vi.spyOn(console, "error").mockImplementation(() => {});
  });

  it("collapses a date split across pages into a single day entry", async () => {
    const fetchFn = vi
      .fn()
      .mockResolvedValueOnce(page("2026-06-25", 22.38, 2, 1))
      .mockResolvedValueOnce(page("2026-06-25", 14.52, 2, 2));

    const { result } = renderHook(() => usePaginatedDailyActivity({ fetchFn, args, enabled: true }));

    await waitFor(() => expect(fetchFn).toHaveBeenCalledTimes(2), { timeout: 3000 });
    await waitFor(() => expect(result.current.data.metadata.total_spend).toBeCloseTo(36.9, 10), { timeout: 3000 });

    expect(result.current.data.results).toHaveLength(1);
    expect(result.current.data.results[0].metrics.spend).toBeCloseTo(36.9, 10);
    expect(result.current.data.results[0].breakdown.entities["team-1"].metrics.spend).toBeCloseTo(36.9, 10);
    expect(result.current.incomplete).toBe(false);
  });

  it("flags the range as incomplete when a page fetch fails instead of looking complete", async () => {
    const fetchFn = vi
      .fn()
      .mockResolvedValueOnce(page("2026-06-25", 22.38, 3, 1))
      .mockRejectedValueOnce(new Error("boom"));

    const { result } = renderHook(() => usePaginatedDailyActivity({ fetchFn, args, enabled: true }));

    await waitFor(() => expect(result.current.failed).toBe(true), { timeout: 3000 });

    expect(result.current.incomplete).toBe(true);
    expect(result.current.isFetchingMore).toBe(false);
    expect(result.current.data.metadata.total_spend).toBeCloseTo(22.38, 10);
  });
});
