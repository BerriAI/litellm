import { describe, expect, it } from "vitest";
import { mergeBreakdowns, mergeResultsByDate } from "./usePaginatedDailyActivity";
import type { DailyData } from "../types";

const baseMetrics = {
  spend: 0,
  prompt_tokens: 0,
  completion_tokens: 0,
  total_tokens: 0,
  api_requests: 0,
  successful_requests: 0,
  failed_requests: 0,
  cache_read_input_tokens: 0,
  cache_creation_input_tokens: 0,
};

const baseBreakdown = {
  models: {},
  model_groups: {},
  mcp_servers: {},
  providers: {},
  api_keys: {},
  entities: {},
};

const row = (
  date: string,
  metrics: Partial<typeof baseMetrics>,
  breakdown: Partial<typeof baseBreakdown> = {},
): DailyData => ({
  date,
  metrics: { ...baseMetrics, ...metrics },
  breakdown: { ...baseBreakdown, ...breakdown } as DailyData["breakdown"],
});

const modelEntry = (spend: number, api_requests: number) => ({
  metrics: { ...baseMetrics, spend, api_requests },
  metadata: {},
  api_key_breakdown: {},
});

describe("mergeResultsByDate", () => {
  it("collapses duplicate-date rows into a single row with summed metrics", () => {
    const rows = [
      row("2025-05-26", { spend: 1.1, api_requests: 5 }),
      row("2025-05-26", { spend: 2.2, api_requests: 8 }),
      row("2025-05-26", { spend: 0.55, api_requests: 3 }),
    ];
    const out = mergeResultsByDate(rows);
    expect(out).toHaveLength(1);
    expect(out[0].date).toBe("2025-05-26");
    expect(out[0].metrics.spend).toBeCloseTo(3.85, 5);
    expect(out[0].metrics.api_requests).toBe(16);
  });

  it("preserves distinct dates and merges only matching ones", () => {
    const rows = [
      row("2025-05-25", { spend: 1, api_requests: 1 }),
      row("2025-05-26", { spend: 2, api_requests: 2 }),
      row("2025-05-26", { spend: 3, api_requests: 3 }),
      row("2025-05-27", { spend: 4, api_requests: 4 }),
    ];
    const out = mergeResultsByDate(rows);
    expect(out.map((r) => r.date)).toEqual([
      "2025-05-25",
      "2025-05-26",
      "2025-05-27",
    ]);
    const may26 = out.find((r) => r.date === "2025-05-26")!;
    expect(may26.metrics.spend).toBe(5);
    expect(may26.metrics.api_requests).toBe(5);
  });

  it("merges breakdown.models sub-map across same-date rows", () => {
    const rows = [
      row(
        "2025-05-26",
        { spend: 1, api_requests: 1 },
        { models: { "gpt-4o": modelEntry(1, 1) } },
      ),
      row(
        "2025-05-26",
        { spend: 2, api_requests: 2 },
        { models: { "gpt-4o": modelEntry(2, 2) } },
      ),
      row(
        "2025-05-26",
        { spend: 5, api_requests: 5 },
        { models: { "claude-3-5-sonnet": modelEntry(5, 5) } },
      ),
    ];
    const out = mergeResultsByDate(rows);
    expect(out).toHaveLength(1);
    const models = out[0].breakdown.models as Record<string, any>;
    expect(Object.keys(models).sort()).toEqual([
      "claude-3-5-sonnet",
      "gpt-4o",
    ]);
    expect(models["gpt-4o"].metrics.spend).toBe(3);
    expect(models["gpt-4o"].metrics.api_requests).toBe(3);
    expect(models["claude-3-5-sonnet"].metrics.spend).toBe(5);
  });

  it("preserves first-seen order of dates", () => {
    const rows = [
      row("2025-05-27", { spend: 1, api_requests: 1 }),
      row("2025-05-25", { spend: 1, api_requests: 1 }),
      row("2025-05-27", { spend: 1, api_requests: 1 }),
      row("2025-05-26", { spend: 1, api_requests: 1 }),
    ];
    const out = mergeResultsByDate(rows);
    expect(out.map((r) => r.date)).toEqual([
      "2025-05-27",
      "2025-05-25",
      "2025-05-26",
    ]);
  });

  it("returns empty array for empty input", () => {
    expect(mergeResultsByDate([])).toEqual([]);
  });

  it("returns single row unchanged when no duplicates", () => {
    const rows = [row("2025-05-26", { spend: 1, api_requests: 1 })];
    const out = mergeResultsByDate(rows);
    expect(out).toHaveLength(1);
    expect(out[0].date).toBe("2025-05-26");
    expect(out[0].metrics.spend).toBe(1);
  });
});

describe("mergeBreakdowns", () => {
  it("merges identical model keys by summing metrics", () => {
    const a = { models: { "gpt-4o": modelEntry(1, 1) } };
    const b = { models: { "gpt-4o": modelEntry(2, 2) } };
    const out = mergeBreakdowns(a, b);
    expect(out.models["gpt-4o"].metrics.spend).toBe(3);
    expect(out.models["gpt-4o"].metrics.api_requests).toBe(3);
  });

  it("unions disjoint keys from both sides", () => {
    const a = { models: { "gpt-4o": modelEntry(1, 1) } };
    const b = { models: { "claude-3-5-sonnet": modelEntry(2, 2) } };
    const out = mergeBreakdowns(a, b);
    expect(out.models["gpt-4o"].metrics.spend).toBe(1);
    expect(out.models["claude-3-5-sonnet"].metrics.spend).toBe(2);
  });

  it("handles missing sub-maps gracefully", () => {
    const a = { models: { "gpt-4o": modelEntry(1, 1) } };
    const b = {};
    const out = mergeBreakdowns(a, b);
    expect(out.models["gpt-4o"].metrics.spend).toBe(1);
    expect(out.providers).toEqual({});
  });

  it("recursively merges api_key_breakdown when same key on both sides", () => {
    const keyEntry = (spend: number) => ({
      metrics: { ...baseMetrics, spend, api_requests: 1 },
      metadata: { key_alias: "alias", team_id: null },
    });
    const a = {
      models: {
        "gpt-4o": {
          metrics: { ...baseMetrics, spend: 1, api_requests: 1 },
          metadata: {},
          api_key_breakdown: { "sk-xxx": keyEntry(1) },
        },
      },
    };
    const b = {
      models: {
        "gpt-4o": {
          metrics: { ...baseMetrics, spend: 2, api_requests: 2 },
          metadata: {},
          api_key_breakdown: { "sk-xxx": keyEntry(2) },
        },
      },
    };
    const out = mergeBreakdowns(a, b);
    expect(out.models["gpt-4o"].metrics.spend).toBe(3);
    expect(out.models["gpt-4o"].api_key_breakdown["sk-xxx"].metrics.spend).toBe(3);
  });
});
