import { renderHook, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { sumMetadata, usePaginatedDailyActivity } from "./usePaginatedDailyActivity";

describe("sumMetadata", () => {
  it("sums flat cost across pages instead of keeping the first page's value", () => {
    // A team whose activity spans more than one page accrues flat cost on each of them.
    // Keeping page 1's value under-reports the Flat Cost and Total Cost tiles.
    const merged = sumMetadata({ total_spend: 1, total_flat_cost: 174.5 }, { total_spend: 2, total_flat_cost: 777 });

    expect(merged.total_flat_cost).toBe(951.5);
    expect(merged.total_spend).toBe(3);
  });

  it("treats a page missing the field as zero rather than dropping the running total", () => {
    expect(sumMetadata({ total_flat_cost: 480 }, {}).total_flat_cost).toBe(480);
    expect(sumMetadata({}, { total_flat_cost: 480 }).total_flat_cost).toBe(480);
  });

  it("carries non-summable keys through from the first page", () => {
    const merged = sumMetadata(
      { page: 1, total_pages: 3, total_spend: 1 },
      { page: 2, total_pages: 3, total_spend: 2 },
    );

    expect(merged.page).toBe(1);
    expect(merged.total_pages).toBe(3);
  });

  it("sums every total_* metric the daily activity metadata exposes", () => {
    // Guards the class of bug rather than one field: a new backend total that nobody adds
    // to SUMMABLE_METADATA_KEYS freezes at page 1, and spend still looks right so it reads
    // as trustworthy.
    const page = {
      total_spend: 1,
      total_prompt_tokens: 1,
      total_completion_tokens: 1,
      total_tokens: 1,
      total_api_requests: 1,
      total_successful_requests: 1,
      total_failed_requests: 1,
      total_cache_read_input_tokens: 1,
      total_cache_creation_input_tokens: 1,
      total_flat_cost: 1,
    };
    const merged = sumMetadata(page, page);

    for (const key of Object.keys(page)) {
      expect(merged[key], `${key} must be summed across pages`).toBe(2);
    }
  });
});

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
