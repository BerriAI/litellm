import { describe, expect, it } from "vitest";
import { mergeDailyResults } from "./mergeDailyResults";
import type { DailyData, SpendMetrics, MetricWithMetadata, KeyMetricWithMetadata } from "../types";

const z = (over: Partial<SpendMetrics> = {}): SpendMetrics => ({
  spend: 0,
  prompt_tokens: 0,
  completion_tokens: 0,
  total_tokens: 0,
  api_requests: 0,
  successful_requests: 0,
  failed_requests: 0,
  cache_read_input_tokens: 0,
  cache_creation_input_tokens: 0,
  ...over,
});

const emptyBreakdown = () => ({
  models: {},
  model_groups: {},
  mcp_servers: {},
  providers: {},
  api_keys: {},
  entities: {},
});

const row = (date: string, over: Partial<DailyData> = {}): DailyData => ({
  date,
  metrics: z(),
  breakdown: emptyBreakdown(),
  ...over,
});

describe("mergeDailyResults", () => {
  it("returns empty array when input is empty", () => {
    expect(mergeDailyResults([])).toEqual([]);
  });

  it("returns the same shape when no dates duplicate", () => {
    const input: DailyData[] = [
      row("2026-05-26", { metrics: z({ spend: 1, api_requests: 3 }) }),
      row("2026-05-25", { metrics: z({ spend: 2, api_requests: 4 }) }),
    ];
    const out = mergeDailyResults(input);
    expect(out).toHaveLength(2);
    expect(out.map((r) => r.date)).toEqual(["2026-05-26", "2026-05-25"]);
    expect(out[0].metrics.spend).toBe(1);
    expect(out[1].metrics.spend).toBe(2);
  });

  it("collapses paginated duplicates into a single bar per date", () => {
    // Mirrors the bug report: 3 backend pages all return rows for the same calendar day.
    const input: DailyData[] = [
      row("2026-05-26", { metrics: z({ spend: 1.5, api_requests: 10, total_tokens: 1000 }) }),
      row("2026-05-26", { metrics: z({ spend: 0.5, api_requests: 4, total_tokens: 400 }) }),
      row("2026-05-26", { metrics: z({ spend: 2, api_requests: 6, total_tokens: 600 }) }),
    ];
    const out = mergeDailyResults(input);
    expect(out).toHaveLength(1);
    expect(out[0].date).toBe("2026-05-26");
    expect(out[0].metrics.spend).toBeCloseTo(4.0);
    expect(out[0].metrics.api_requests).toBe(20);
    expect(out[0].metrics.total_tokens).toBe(2000);
  });

  it("preserves insertion order of first occurrence per date", () => {
    const input: DailyData[] = [
      row("2026-05-26"),
      row("2026-05-25"),
      row("2026-05-26"),
      row("2026-05-27"),
      row("2026-05-25"),
    ];
    const out = mergeDailyResults(input);
    expect(out.map((r) => r.date)).toEqual(["2026-05-26", "2026-05-25", "2026-05-27"]);
  });

  it("sums every numeric metric, including cache token fields", () => {
    const input: DailyData[] = [
      row("d", {
        metrics: z({
          spend: 1,
          prompt_tokens: 10,
          completion_tokens: 20,
          total_tokens: 30,
          api_requests: 1,
          successful_requests: 1,
          failed_requests: 0,
          cache_read_input_tokens: 5,
          cache_creation_input_tokens: 6,
        }),
      }),
      row("d", {
        metrics: z({
          spend: 2,
          prompt_tokens: 4,
          completion_tokens: 6,
          total_tokens: 10,
          api_requests: 3,
          successful_requests: 2,
          failed_requests: 1,
          cache_read_input_tokens: 7,
          cache_creation_input_tokens: 8,
        }),
      }),
    ];
    const m = mergeDailyResults(input)[0].metrics;
    expect(m).toEqual({
      spend: 3,
      prompt_tokens: 14,
      completion_tokens: 26,
      total_tokens: 40,
      api_requests: 4,
      successful_requests: 3,
      failed_requests: 1,
      cache_read_input_tokens: 12,
      cache_creation_input_tokens: 14,
    });
  });

  it("recursively merges per-model breakdown maps", () => {
    const mk = (spend: number): MetricWithMetadata => ({
      metrics: z({ spend, api_requests: 1 }),
      metadata: {},
      api_key_breakdown: {},
    });
    const input: DailyData[] = [
      row("d", { breakdown: { ...emptyBreakdown(), models: { "gpt-4": mk(1), "claude-3-opus": mk(2) } } }),
      row("d", { breakdown: { ...emptyBreakdown(), models: { "gpt-4": mk(3), "claude-3-sonnet": mk(4) } } }),
    ];
    const out = mergeDailyResults(input)[0];
    expect(Object.keys(out.breakdown.models).sort()).toEqual(["claude-3-opus", "claude-3-sonnet", "gpt-4"]);
    expect(out.breakdown.models["gpt-4"].metrics.spend).toBe(4);
    expect(out.breakdown.models["gpt-4"].metrics.api_requests).toBe(2);
    expect(out.breakdown.models["claude-3-opus"].metrics.spend).toBe(2);
    expect(out.breakdown.models["claude-3-sonnet"].metrics.spend).toBe(4);
  });

  it("merges api_key_breakdown nested inside model entries", () => {
    const ent = (modelSpend: number, keySpend: number): MetricWithMetadata => ({
      metrics: z({ spend: modelSpend }),
      metadata: {},
      api_key_breakdown: {
        "key-A": { metrics: z({ spend: keySpend }), metadata: { key_alias: "A", team_id: null } },
      },
    });
    const input: DailyData[] = [
      row("d", { breakdown: { ...emptyBreakdown(), models: { "gpt-4": ent(1, 0.5) } } }),
      row("d", { breakdown: { ...emptyBreakdown(), models: { "gpt-4": ent(2, 1.5) } } }),
    ];
    const out = mergeDailyResults(input)[0];
    const merged = out.breakdown.models["gpt-4"];
    expect(merged.metrics.spend).toBe(3);
    expect(merged.api_key_breakdown["key-A"].metrics.spend).toBe(2);
    expect(merged.api_key_breakdown["key-A"].metadata.key_alias).toBe("A");
  });

  it("merges api_keys breakdown (KeyMetricWithMetadata, no nested api_key_breakdown)", () => {
    const k = (spend: number): KeyMetricWithMetadata => ({
      metrics: z({ spend }),
      metadata: { key_alias: "k", team_id: "t1" },
    });
    const input: DailyData[] = [
      row("d", { breakdown: { ...emptyBreakdown(), api_keys: { "sk-1": k(1) } } }),
      row("d", { breakdown: { ...emptyBreakdown(), api_keys: { "sk-1": k(2), "sk-2": k(3) } } }),
    ];
    const out = mergeDailyResults(input)[0];
    expect(out.breakdown.api_keys["sk-1"].metrics.spend).toBe(3);
    expect(out.breakdown.api_keys["sk-2"].metrics.spend).toBe(3);
  });

  it("does not mutate the input array or its rows", () => {
    const orig: DailyData[] = [
      row("d", { metrics: z({ spend: 1 }) }),
      row("d", { metrics: z({ spend: 2 }) }),
    ];
    const snapshot = JSON.parse(JSON.stringify(orig));
    mergeDailyResults(orig);
    expect(orig).toEqual(snapshot);
  });

  it("tolerates undefined breakdown maps without throwing", () => {
    const input = [
      { date: "d", metrics: z({ spend: 1 }), breakdown: undefined as any },
      { date: "d", metrics: z({ spend: 2 }), breakdown: emptyBreakdown() },
    ] as unknown as DailyData[];
    const out = mergeDailyResults(input);
    expect(out).toHaveLength(1);
    expect(out[0].metrics.spend).toBe(3);
    expect(out[0].breakdown).toBeDefined();
    expect(out[0].breakdown.models).toEqual({});
  });

  it("skips rows that are null/undefined or missing a date", () => {
    const input = [
      null,
      row("d", { metrics: z({ spend: 1 }) }),
      { date: undefined } as any,
      row("d", { metrics: z({ spend: 2 }) }),
    ] as unknown as DailyData[];
    const out = mergeDailyResults(input);
    expect(out).toHaveLength(1);
    expect(out[0].metrics.spend).toBe(3);
  });
});
