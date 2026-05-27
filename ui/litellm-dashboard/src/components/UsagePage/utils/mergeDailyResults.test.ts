import { describe, expect, it } from "vitest";
import { mergeResultsByDate } from "./mergeDailyResults";
import { DailyData } from "../types";

const emptyMetrics = () => ({
  spend: 0,
  prompt_tokens: 0,
  completion_tokens: 0,
  total_tokens: 0,
  api_requests: 0,
  successful_requests: 0,
  failed_requests: 0,
  cache_read_input_tokens: 0,
  cache_creation_input_tokens: 0,
});

const emptyBreakdown = () => ({
  models: {},
  model_groups: {},
  mcp_servers: {},
  providers: {},
  api_keys: {},
  entities: {},
});

function row(date: string, overrides: Partial<DailyData> = {}): DailyData {
  return {
    date,
    metrics: { ...emptyMetrics(), ...(overrides.metrics || {}) },
    breakdown: { ...emptyBreakdown(), ...(overrides.breakdown || {}) },
  };
}

describe("mergeResultsByDate", () => {
  it("is a no-op for an empty input", () => {
    expect(mergeResultsByDate([])).toEqual([]);
  });

  it("is a no-op for a single row", () => {
    const input = [row("2026-05-27", { metrics: { ...emptyMetrics(), spend: 1.5 } })];
    const out = mergeResultsByDate(input);
    expect(out).toHaveLength(1);
    expect(out[0].date).toBe("2026-05-27");
    expect(out[0].metrics.spend).toBe(1.5);
  });

  it("collapses two rows that share the same date and sums metrics", () => {
    // Mirrors the LIT-3383 repro: paginated `Today` returns two pages, both
    // tagged with the same `date`. Without dedupe the BarChart renders two
    // bars sharing one X-axis label.
    const a = row("2026-05-27", {
      metrics: {
        ...emptyMetrics(),
        spend: 1,
        prompt_tokens: 10,
        completion_tokens: 20,
        total_tokens: 30,
        api_requests: 2,
        successful_requests: 2,
      },
    });
    const b = row("2026-05-27", {
      metrics: {
        ...emptyMetrics(),
        spend: 2.5,
        prompt_tokens: 100,
        completion_tokens: 200,
        total_tokens: 300,
        api_requests: 4,
        successful_requests: 3,
        failed_requests: 1,
        cache_read_input_tokens: 7,
        cache_creation_input_tokens: 9,
      },
    });
    const out = mergeResultsByDate([a, b]);
    expect(out).toHaveLength(1);
    expect(out[0].date).toBe("2026-05-27");
    expect(out[0].metrics).toEqual({
      spend: 3.5,
      prompt_tokens: 110,
      completion_tokens: 220,
      total_tokens: 330,
      api_requests: 6,
      successful_requests: 5,
      failed_requests: 1,
      cache_read_input_tokens: 7,
      cache_creation_input_tokens: 9,
    });
  });

  it("preserves distinct dates in first-seen order", () => {
    const out = mergeResultsByDate([
      row("2026-05-26"),
      row("2026-05-27"),
      row("2026-05-26"),
      row("2026-05-25"),
    ]);
    expect(out.map((r) => r.date)).toEqual([
      "2026-05-26",
      "2026-05-27",
      "2026-05-25",
    ]);
  });

  it("merges nested breakdown maps additively", () => {
    const a = row("2026-05-27", {
      breakdown: {
        ...emptyBreakdown(),
        models: {
          "gpt-4": {
            metrics: { ...emptyMetrics(), spend: 1, api_requests: 1 },
            metadata: {},
            api_key_breakdown: {},
          },
        },
        api_keys: {
          "sk-1": {
            metrics: { ...emptyMetrics(), spend: 1, api_requests: 1 },
            metadata: { key_alias: "alpha", team_id: null },
          },
        },
      },
    });
    const b = row("2026-05-27", {
      breakdown: {
        ...emptyBreakdown(),
        models: {
          "gpt-4": {
            metrics: { ...emptyMetrics(), spend: 2, api_requests: 3 },
            metadata: {},
            api_key_breakdown: {},
          },
          "gpt-5": {
            metrics: { ...emptyMetrics(), spend: 9, api_requests: 4 },
            metadata: {},
            api_key_breakdown: {},
          },
        },
        api_keys: {
          "sk-1": {
            metrics: { ...emptyMetrics(), spend: 4, api_requests: 2 },
            metadata: { key_alias: "alpha", team_id: "team-x" },
          },
        },
      },
    });
    const out = mergeResultsByDate([a, b]);
    expect(out).toHaveLength(1);
    const merged = out[0];
    expect(merged.breakdown.models["gpt-4"].metrics.spend).toBe(3);
    expect(merged.breakdown.models["gpt-4"].metrics.api_requests).toBe(4);
    expect(merged.breakdown.models["gpt-5"].metrics.spend).toBe(9);
    expect(merged.breakdown.api_keys["sk-1"].metrics.spend).toBe(5);
    expect(merged.breakdown.api_keys["sk-1"].metadata.key_alias).toBe("alpha");
    // Prefer first non-null team_id; here second row supplied team-x because first was null.
    expect(merged.breakdown.api_keys["sk-1"].metadata.team_id).toBe("team-x");
  });

  it("handles a row with no breakdown maps gracefully", () => {
    const a = row("2026-05-27", {
      metrics: { ...emptyMetrics(), spend: 1 },
      breakdown: {} as any,
    });
    const b = row("2026-05-27", {
      metrics: { ...emptyMetrics(), spend: 2 },
      breakdown: {} as any,
    });
    const out = mergeResultsByDate([a, b]);
    expect(out).toHaveLength(1);
    expect(out[0].metrics.spend).toBe(3);
    expect(out[0].breakdown.models).toEqual({});
  });

  it("does not mutate input rows", () => {
    const original = row("2026-05-27", {
      metrics: { ...emptyMetrics(), spend: 5 },
      breakdown: {
        ...emptyBreakdown(),
        models: {
          "gpt-4": {
            metrics: { ...emptyMetrics(), spend: 5 },
            metadata: {},
            api_key_breakdown: {},
          },
        },
      },
    });
    const dup = row("2026-05-27", {
      metrics: { ...emptyMetrics(), spend: 7 },
      breakdown: {
        ...emptyBreakdown(),
        models: {
          "gpt-4": {
            metrics: { ...emptyMetrics(), spend: 7 },
            metadata: {},
            api_key_breakdown: {},
          },
        },
      },
    });
    const beforeOrig = JSON.parse(JSON.stringify(original));
    const beforeDup = JSON.parse(JSON.stringify(dup));
    mergeResultsByDate([original, dup]);
    expect(original).toEqual(beforeOrig);
    expect(dup).toEqual(beforeDup);
  });

  it("dedupes a realistic paginated `Today` accumulator (LIT-3383 repro)", () => {
    // Three paginated pages, all on the same date — exactly what the user sees
    // when picking the `Today` range with a tag-heavy instance.
    const today = "2026-05-27";
    const pages = [1, 2, 3].map((p) =>
      row(today, {
        metrics: {
          ...emptyMetrics(),
          spend: p * 1.0,
          api_requests: p,
          successful_requests: p,
        },
      }),
    );
    expect(pages).toHaveLength(3);
    const merged = mergeResultsByDate(pages);
    expect(merged).toHaveLength(1);
    expect(merged[0].metrics.spend).toBeCloseTo(6.0);
    expect(merged[0].metrics.api_requests).toBe(6);
    expect(merged[0].metrics.successful_requests).toBe(6);
  });
});
