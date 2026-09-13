import { renderHook, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { DailyData, SpendMetrics } from "@/components/UsagePage/types";
import { mergeDailyResults, sumMetadata, usePaginatedDailyActivity } from "./usePaginatedDailyActivity";

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

const metricsOf = (spend: number): SpendMetrics => ({
  spend,
  prompt_tokens: 0,
  completion_tokens: 0,
  total_tokens: 0,
  api_requests: 1,
  successful_requests: 1,
  failed_requests: 0,
  cache_read_input_tokens: 0,
  cache_creation_input_tokens: 0,
  compression_savings_spend: spend,
});

const dayOf = (date: string, spend: number, apiKey: string = "sk-1"): DailyData => ({
  date,
  metrics: metricsOf(spend),
  breakdown: {
    models: {
      "gpt-4o": {
        metrics: metricsOf(spend),
        metadata: {},
        api_key_breakdown: {
          [apiKey]: { metrics: metricsOf(spend), metadata: { key_alias: "alias-1", team_id: null } },
        },
      },
    },
    model_groups: {},
    mcp_servers: {},
    providers: {},
    api_keys: { [apiKey]: { metrics: metricsOf(spend), metadata: { key_alias: "alias-1", team_id: null } } },
    entities: {},
  },
});

describe("mergeDailyResults", () => {
  it("collapses repeated dates into one entry with summed metrics (the LIT-5818 $2/$2/$1 case)", () => {
    const merged = mergeDailyResults(mergeDailyResults([dayOf("2026-08-16", 2)], [dayOf("2026-08-16", 2)]), [
      dayOf("2026-08-16", 1),
    ]);

    expect(merged).toHaveLength(1);
    expect(merged[0].metrics.spend).toBe(5);
    expect(merged[0].metrics.compression_savings_spend).toBe(5);
  });

  it("appends unseen dates in arrival order", () => {
    const merged = mergeDailyResults([dayOf("2026-08-16", 2)], [dayOf("2026-08-15", 0.5)]);

    expect(merged.map((d) => d.date)).toEqual(["2026-08-16", "2026-08-15"]);
    expect(merged[1].metrics.spend).toBe(0.5);
  });

  it("merges every breakdown level including the nested per-key breakdown", () => {
    const merged = mergeDailyResults([dayOf("2026-08-16", 2, "sk-1")], [dayOf("2026-08-16", 3, "sk-1")]);

    expect(merged[0].breakdown.models["gpt-4o"].metrics.spend).toBe(5);
    expect(merged[0].breakdown.models["gpt-4o"].api_key_breakdown["sk-1"].metrics.spend).toBe(5);
    expect(merged[0].breakdown.api_keys["sk-1"].metrics.spend).toBe(5);
    expect(merged[0].breakdown.api_keys["sk-1"].metadata.key_alias).toBe("alias-1");
  });

  it("unions breakdown keys that appear on different pages", () => {
    const merged = mergeDailyResults([dayOf("2026-08-16", 2, "sk-1")], [dayOf("2026-08-16", 3, "sk-2")]);

    expect(merged[0].breakdown.api_keys["sk-1"].metrics.spend).toBe(2);
    expect(merged[0].breakdown.api_keys["sk-2"].metrics.spend).toBe(3);
  });

  it("sums metric keys it has never heard of so a future backend column cannot silently freeze", () => {
    const withExtra = (spend: number): DailyData => ({
      ...dayOf("2026-08-16", spend),
      metrics: { ...metricsOf(spend), future_savings_spend: spend } as SpendMetrics,
    });
    const merged = mergeDailyResults([withExtra(2)], [withExtra(3)]);

    expect((merged[0].metrics as Record<string, number>).future_savings_spend).toBe(5);
  });
});

describe("usePaginatedDailyActivity page accumulation", () => {
  it("returns one entry per date when a date's rows span multiple pages", async () => {
    const pages = [
      { results: [dayOf("2026-08-16", 2)], metadata: { total_pages: 3, page: 1, total_spend: 2 } },
      { results: [dayOf("2026-08-16", 2)], metadata: { total_pages: 3, page: 2, total_spend: 2 } },
      {
        results: [dayOf("2026-08-16", 1), dayOf("2026-08-15", 0.5)],
        metadata: { total_pages: 3, page: 3, total_spend: 1.5 },
      },
    ];
    const fetchFn = vi.fn((_token: string, _start: Date, _end: Date, page: number) => Promise.resolve(pages[page - 1]));
    const start = new Date("2026-08-10");
    const end = new Date("2026-08-17");

    const { result } = renderHook(() =>
      usePaginatedDailyActivity({ fetchFn, args: ["tok", start, end, null], enabled: true }),
    );

    await waitFor(() => expect(result.current.data.metadata.page).toBe(3), { timeout: 5000 });

    expect(result.current.data.results.map((d) => d.date)).toEqual(["2026-08-16", "2026-08-15"]);
    expect(result.current.data.results[0].metrics.spend).toBe(5);
    expect(result.current.data.metadata.total_spend).toBe(5.5);
  });
});
