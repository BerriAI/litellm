import { describe, expect, it } from "vitest";

import type { DailyData, SpendMetrics } from "@/components/UsagePage/types";
import type { ToolSpendDailyEntry, ToolSpendEntry } from "@/components/networking";
import { buildDailyToolSeries, computeCacheLeakage, isAnthropicModel, topToolsBySpend } from "./costOptimizationUtils";

const metrics = (overrides: Partial<SpendMetrics>): SpendMetrics => ({
  spend: 0,
  prompt_tokens: 0,
  completion_tokens: 0,
  total_tokens: 0,
  api_requests: 0,
  successful_requests: 0,
  failed_requests: 0,
  cache_read_input_tokens: 0,
  cache_creation_input_tokens: 0,
  ...overrides,
});

const day = (
  date: string,
  keys: Record<string, { alias: string | null; metrics: Partial<SpendMetrics> }>,
): DailyData => ({
  date,
  metrics: metrics({}),
  breakdown: {
    models: {},
    model_groups: {},
    mcp_servers: {},
    providers: {},
    entities: {},
    api_keys: Object.fromEntries(
      Object.entries(keys).map(([hash, v]) => [
        hash,
        { metrics: metrics(v.metrics), metadata: { key_alias: v.alias, team_id: null } },
      ]),
    ),
  },
});

const modelDay = (date: string, models: Record<string, Partial<SpendMetrics>>): DailyData => ({
  date,
  metrics: metrics({}),
  breakdown: {
    models: Object.fromEntries(
      Object.entries(models).map(([name, m]) => [name, { metrics: metrics(m), metadata: {}, api_key_breakdown: {} }]),
    ),
    model_groups: {},
    mcp_servers: {},
    providers: {},
    entities: {},
    api_keys: {},
  },
});

describe("computeCacheLeakage", () => {
  it("aggregates a key's tokens and savings across multiple days", () => {
    const results = [
      day("2026-07-01", { h1: { alias: "svc-a", metrics: { prompt_tokens: 1000, cache_read_input_tokens: 0 } } }),
      day("2026-07-02", { h1: { alias: "svc-a", metrics: { prompt_tokens: 500, cache_read_input_tokens: 0 } } }),
    ];
    const { rows } = computeCacheLeakage(results);
    expect(rows).toHaveLength(1);
    expect(rows[0].uncachedPromptTokens).toBe(1500);
  });

  it("subtracts cache reads and writes from prompt tokens instead of double-counting them", () => {
    const results = [
      day("2026-07-01", {
        h1: {
          alias: "svc-a",
          metrics: { prompt_tokens: 1000, cache_read_input_tokens: 400, cache_creation_input_tokens: 100 },
        },
      }),
    ];
    const { rows } = computeCacheLeakage(results);
    expect(rows).toHaveLength(1);
    expect(rows[0].uncachedPromptTokens).toBe(500);
    expect(rows[0].cacheHitRatio).toBeCloseTo(0.4, 6);
  });

  it("prices leakage at the portfolio's realized cache-read discount and drops fully cached keys", () => {
    const results = [
      day("2026-07-01", {
        cacher: {
          alias: "cacher",
          metrics: { prompt_tokens: 1000, cache_read_input_tokens: 1000, prompt_caching_savings_spend: 2.0 },
        },
        leaker: { alias: "leaker", metrics: { prompt_tokens: 500 } },
      }),
    ];
    const { rows, discountPerToken } = computeCacheLeakage(results);
    expect(discountPerToken).toBeCloseTo(0.002, 6);
    expect(rows.map((r) => r.label)).toEqual(["leaker"]);
    expect(rows[0].potentialSavings).toBeCloseTo(1.0, 6);
  });

  it("returns null estimate and ranks by uncached tokens when nobody used caching", () => {
    const results = [
      day("2026-07-01", {
        big: { alias: "big", metrics: { prompt_tokens: 9000 } },
        small: { alias: "small", metrics: { prompt_tokens: 100 } },
      }),
    ];
    const { rows, discountPerToken } = computeCacheLeakage(results);
    expect(discountPerToken).toBeNull();
    expect(rows.map((r) => r.label)).toEqual(["big", "small"]);
    expect(rows.every((r) => r.potentialSavings === null)).toBe(true);
  });

  it("computes cache hit ratio against total prompt tokens and clamps inconsistent data at zero", () => {
    const results = [
      day("2026-07-01", {
        onlycache: { alias: "onlycache", metrics: { cache_read_input_tokens: 100 } },
        mixed: { alias: "mixed", metrics: { prompt_tokens: 1000, cache_read_input_tokens: 750 } },
      }),
    ];
    const { rows } = computeCacheLeakage(results);
    expect(rows.map((r) => r.label)).toEqual(["mixed"]);
    expect(rows[0].cacheHitRatio).toBeCloseTo(0.75, 6);
    expect(rows[0].uncachedPromptTokens).toBe(250);
  });

  it("respects the row limit", () => {
    const keys = Object.fromEntries(
      Array.from({ length: 15 }, (_, i) => [`h${i}`, { alias: `k${i}`, metrics: { prompt_tokens: i + 1 } }]),
    );
    const { rows } = computeCacheLeakage([day("2026-07-01", keys)], "key", 5);
    expect(rows).toHaveLength(5);
  });
});

describe("computeCacheLeakage by model", () => {
  it("aggregates only Anthropic models and ignores other providers", () => {
    const models: Record<string, Partial<SpendMetrics>> = {
      "claude-sonnet-5": { prompt_tokens: 10000, cache_read_input_tokens: 0 },
      "anthropic/claude-haiku-4-5": { prompt_tokens: 4000, cache_read_input_tokens: 0 },
      "bedrock/anthropic.claude-3-5-sonnet": { prompt_tokens: 2000, cache_read_input_tokens: 0 },
      "gpt-4o": { prompt_tokens: 9000, cache_read_input_tokens: 0 },
      "deepseek-chat": { prompt_tokens: 8000, cache_read_input_tokens: 0 },
    };
    const { rows } = computeCacheLeakage([modelDay("2026-07-01", models)], "model");
    expect(rows.map((r) => r.id)).toEqual([
      "claude-sonnet-5",
      "anthropic/claude-haiku-4-5",
      "bedrock/anthropic.claude-3-5-sonnet",
    ]);
  });

  it("labels model rows by model name with no sublabel", () => {
    const results = [modelDay("2026-07-01", { "claude-sonnet-5": { prompt_tokens: 1000 } })];
    const { rows } = computeCacheLeakage(results, "model");
    expect(rows[0].label).toBe("claude-sonnet-5");
    expect(rows[0].sublabel).toBeNull();
  });

  it("prices model leakage at the Anthropic realized cache-read discount", () => {
    const results = [
      modelDay("2026-07-01", {
        "claude-sonnet-5": { prompt_tokens: 1000, cache_read_input_tokens: 1000, prompt_caching_savings_spend: 2.0 },
        "claude-haiku-4-5": { prompt_tokens: 500 },
      }),
    ];
    const { rows, discountPerToken } = computeCacheLeakage(results, "model");
    expect(discountPerToken).toBeCloseTo(0.002, 6);
    expect(rows.map((r) => r.id)).toEqual(["claude-haiku-4-5"]);
    expect(rows[0].potentialSavings).toBeCloseTo(1.0, 6);
  });
});

describe("isAnthropicModel", () => {
  it("matches Claude-family models across providers and rejects others", () => {
    const anthropic = [
      "claude-sonnet-5",
      "anthropic/claude-haiku-4-5",
      "bedrock/anthropic.claude-3-5-sonnet",
      "vertex_ai/claude-opus-4-8",
    ];
    const others = ["gpt-4o", "deepseek-chat", "gemini-2.5-pro", "mistral-large"];
    expect(anthropic.every(isAnthropicModel)).toBe(true);
    expect(others.some(isAnthropicModel)).toBe(false);
  });
});

describe("buildDailyToolSeries", () => {
  const daily: ToolSpendDailyEntry[] = [
    { date: "2026-07-01", tool_name: "search", spend: 1.0, call_count: 1 },
    { date: "2026-07-01", tool_name: "read", spend: 0.5, call_count: 1 },
    { date: "2026-07-02", tool_name: "search", spend: 2.0, call_count: 1 },
    { date: "2026-07-01", tool_name: "excluded", spend: 9.0, call_count: 1 },
  ];

  it("pivots to per-date points keyed by the selected tools, dropping others", () => {
    const series = buildDailyToolSeries(daily, ["search", "read"]);
    expect(series).toEqual([
      { date: "2026-07-01", search: 1.0, read: 0.5 },
      { date: "2026-07-02", search: 2.0, read: 0 },
    ]);
  });

  it("sums repeated (date, tool) rows", () => {
    const series = buildDailyToolSeries(
      [
        { date: "2026-07-01", tool_name: "search", spend: 1.0, call_count: 1 },
        { date: "2026-07-01", tool_name: "search", spend: 2.5, call_count: 1 },
      ],
      ["search"],
    );
    expect(series[0].search).toBe(3.5);
  });
});

describe("topToolsBySpend", () => {
  const byTool: ToolSpendEntry[] = [
    { tool_name: "a", spend: 1, call_count: 1, total_tokens: 1 },
    { tool_name: "b", spend: 5, call_count: 1, total_tokens: 1 },
    { tool_name: "c", spend: 3, call_count: 1, total_tokens: 1 },
  ];

  it("sorts by spend descending and truncates to the limit", () => {
    expect(topToolsBySpend(byTool, 2).map((t) => t.tool_name)).toEqual(["b", "c"]);
  });
});
