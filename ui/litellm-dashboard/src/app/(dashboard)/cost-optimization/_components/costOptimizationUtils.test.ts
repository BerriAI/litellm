import { describe, expect, it } from "vitest";

import type { DailyData, SpendMetrics } from "@/components/UsagePage/types";
import type { ToolSpendDailyEntry, ToolSpendEntry } from "@/components/networking";
import {
  SAVINGS_COLORS,
  SAVINGS_DRIVERS,
  SAVINGS_SERIES,
  buildDailyToolSeries,
  computeCacheLeakage,
  formatRangeLabel,
  isAnthropicModel,
  localIsoDay,
  toCumulative,
  topToolsBySpend,
  usd,
  withStartAnchor,
} from "./costOptimizationUtils";

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
    const { rows, netSavingsPerCachedToken } = computeCacheLeakage(results);
    expect(netSavingsPerCachedToken).toBeCloseTo(0.002, 6);
    expect(rows.map((r) => r.label)).toEqual(["leaker"]);
    expect(rows[0].potentialSavings).toBeCloseTo(1.0, 6);
  });

  it("divides net savings by cache writes as well as reads, since a new cacher pays write premiums too", () => {
    const results = [
      day("2026-07-01", {
        cacher: {
          alias: "cacher",
          metrics: {
            prompt_tokens: 2000,
            cache_read_input_tokens: 1000,
            cache_creation_input_tokens: 1000,
            prompt_caching_savings_spend: 2.0,
          },
        },
        leaker: { alias: "leaker", metrics: { prompt_tokens: 500 } },
      }),
    ];
    const { rows, netSavingsPerCachedToken } = computeCacheLeakage(results);
    expect(netSavingsPerCachedToken).toBeCloseTo(0.001, 6);
    expect(rows[0].potentialSavings).toBeCloseTo(0.5, 6);
  });

  it("declines to price leakage when write premiums leave caching net negative", () => {
    const results = [
      day("2026-07-01", {
        writer: {
          alias: "writer",
          metrics: {
            prompt_tokens: 2000,
            cache_read_input_tokens: 100,
            cache_creation_input_tokens: 1500,
            prompt_caching_savings_spend: -0.75,
          },
        },
        leaker: { alias: "leaker", metrics: { prompt_tokens: 500 } },
      }),
    ];
    const { rows, netSavingsPerCachedToken } = computeCacheLeakage(results);
    expect(netSavingsPerCachedToken).toBeLessThan(0);
    expect(rows.every((r) => r.potentialSavings === null)).toBe(true);
    expect(rows.map((r) => r.label)).toEqual(["leaker", "writer"]);
  });

  it("returns null estimate and ranks by uncached tokens when nobody used caching", () => {
    const results = [
      day("2026-07-01", {
        big: { alias: "big", metrics: { prompt_tokens: 9000 } },
        small: { alias: "small", metrics: { prompt_tokens: 100 } },
      }),
    ];
    const { rows, netSavingsPerCachedToken } = computeCacheLeakage(results);
    expect(netSavingsPerCachedToken).toBeNull();
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
    const { rows, netSavingsPerCachedToken } = computeCacheLeakage(results, "model");
    expect(netSavingsPerCachedToken).toBeCloseTo(0.002, 6);
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

describe("localIsoDay", () => {
  it("reads the date off the viewer's clock rather than shifting it to UTC", () => {
    expect(localIsoDay(new Date(2026, 6, 23, 23, 30))).toBe("2026-07-23");
    expect(localIsoDay(new Date(2026, 0, 5, 0, 30))).toBe("2026-01-05");
  });
});

describe("toCumulative", () => {
  const point = (date: string, compression: number, caching: number, autorouter: number = 0) => ({
    date,
    Compression: compression,
    "Prompt caching": caching,
    "Auto-router": autorouter,
  });

  it("turns each reading into everything saved up to that point", () => {
    const running = toCumulative([point("Jul 1", 1, 10), point("Jul 2", 2, 20), point("Jul 3", 3, 30)]);
    expect(running.map((p) => p.Compression)).toEqual([1, 3, 6]);
    expect(running.map((p) => p["Prompt caching"])).toEqual([10, 30, 60]);
    expect(running.map((p) => p["Auto-router"])).toEqual([0, 0, 0]);
  });

  it("accumulates each driver on its own, so one flat series cannot lift the other", () => {
    const running = toCumulative([point("Jul 1", 0, 5), point("Jul 2", 0, 5)]);
    expect(running.map((p) => p.Compression)).toEqual([0, 0]);
    expect(running.map((p) => p["Prompt caching"])).toEqual([5, 10]);
    expect(running.map((p) => p["Auto-router"])).toEqual([0, 0]);
  });

  it("never falls, even across a quiet interval", () => {
    const running = toCumulative([point("Jul 1", 4, 0), point("Jul 2", 0, 0), point("Jul 3", 1, 0)]);
    expect(running.map((p) => p.Compression)).toEqual([4, 4, 5]);
  });

  it("keeps the labels and length of the readings it was given", () => {
    const running = toCumulative([point("9am", 1, 1), point("10am", 1, 1)]);
    expect(running.map((p) => p.date)).toEqual(["9am", "10am"]);
    expect(toCumulative([])).toEqual([]);
  });

  it("accumulates auto-router savings like other drivers", () => {
    const running = toCumulative([point("Jul 1", 1, 1, 5), point("Jul 2", 1, 1, 10)]);
    expect(running.map((p) => p["Auto-router"])).toEqual([5, 15]);
  });
});

describe("withStartAnchor", () => {
  const point = (date: string, compression: number, caching: number, autorouter: number = 0) => ({
    date,
    Compression: compression,
    "Prompt caching": caching,
    "Auto-router": autorouter,
  });

  it("lifts a single-day cumulative off a floating dot by prepending a $0 origin", () => {
    const anchored = withStartAnchor([point("Jul 24", 12, 30)], "Jul 24");
    expect(anchored).toEqual([point("Jul 24", 0, 0), point("Jul 24", 12, 30)]);
  });

  it("starts the range at zero without disturbing the running totals that follow", () => {
    const anchored = withStartAnchor([point("Jul 16", 5, 1), point("Jul 17", 9, 4)], "Jul 16");
    expect(anchored.map((p) => p.Compression)).toEqual([0, 5, 9]);
    expect(anchored.map((p) => p["Prompt caching"])).toEqual([0, 1, 4]);
    expect(anchored.map((p) => p["Auto-router"])).toEqual([0, 0, 0]);
  });

  it("leaves an empty series alone so the chart's own no-data state can show", () => {
    expect(withStartAnchor([], "Jul 24")).toEqual([]);
  });
});

describe("formatRangeLabel", () => {
  it("reads as a range across days", () => {
    expect(formatRangeLabel(new Date(2026, 6, 16), new Date(2026, 6, 23))).toBe("Jul 16 \u2013 Jul 23");
  });

  it("collapses to one date when both ends are the same day", () => {
    expect(formatRangeLabel(new Date(2026, 6, 23), new Date(2026, 6, 23))).toBe("Jul 23");
  });

  it("is empty until both ends are picked", () => {
    expect(formatRangeLabel(undefined, new Date(2026, 6, 23))).toBe("");
    expect(formatRangeLabel(new Date(2026, 6, 23), undefined)).toBe("");
  });
});

describe("usd", () => {
  it("keeps four decimals for sub-dollar amounts so small savings stay visible", () => {
    expect(usd(0.05)).toBe("$0.0500");
    expect(usd(1.5)).toBe("$1.50");
    expect(usd(0)).toBe("$0.00");
  });

  it("signs a loss ahead of the symbol and keeps its precision", () => {
    // A driver can be negative once a model switch is charged for its cold cache.
    // Sizing decimals off the raw value would render this as "$-0.00".
    expect(usd(-0.05)).toBe("-$0.0500");
    expect(usd(-0.0004)).toBe("-$0.0004");
    expect(usd(-12.4)).toBe("-$12.40");
  });
});

describe("savings driver colours", () => {
  it("keeps a driver's colour when a driver above it is filtered out", () => {
    // Charts colour by position in the data they are given, and the donut is given
    // only drivers that saved something. Compression is zero on any deployment not
    // running the compression guardrail, so the survivors must not slide onto the
    // colours of the drivers dropped above them.
    const totals = { Compression: 0, "Prompt caching": 4, "Auto-router": 7 } as const;
    const plotted = SAVINGS_DRIVERS.map(({ name, color }) => ({ name, color, usd: totals[name] })).filter(
      (d) => d.usd > 0,
    );

    expect(plotted.map((d) => [d.name, d.color])).toEqual([
      ["Prompt caching", "blue"],
      ["Auto-router", "amber"],
    ]);
  });

  it("agrees with the legend, which is built from the unfiltered list", () => {
    const legend = new Map(SAVINGS_SERIES.map((name, i) => [SAVINGS_COLORS[i], name]));
    for (const { name, color } of SAVINGS_DRIVERS) {
      expect(legend.get(color)).toBe(name);
    }
  });
});
