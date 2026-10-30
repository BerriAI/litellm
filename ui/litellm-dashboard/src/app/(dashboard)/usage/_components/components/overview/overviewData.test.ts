import { describe, expect, it } from "vitest";
import type { DailyData } from "@/components/UsagePage/types";
import {
  bucketSeries,
  bucketTotals,
  colorForModel,
  labelForDate,
  modelColorMap,
  OTHER_COLOR,
  rollUpBreakdown,
  seriesBy,
} from "./overviewData";

const metrics = (spend: number) => ({
  spend,
  prompt_tokens: 0,
  completion_tokens: 0,
  total_tokens: spend * 10,
  api_requests: 1,
  successful_requests: 1,
  failed_requests: 0,
  cache_read_input_tokens: 0,
  cache_creation_input_tokens: 0,
});

const day = (date: string, total: number, models: Record<string, number>): DailyData =>
  ({
    date,
    metrics: metrics(total),
    breakdown: {
      models: {},
      model_groups: Object.fromEntries(
        Object.entries(models).map(([name, spend]) => [
          name,
          { metrics: metrics(spend), metadata: {}, api_key_breakdown: {} },
        ]),
      ),
      mcp_servers: {},
      providers: {},
      api_keys: {},
      entities: {},
    },
  }) as unknown as DailyData;

describe("seriesBy", () => {
  it("keeps a model named date, label or Other from overwriting the day's own fields or the remainder", () => {
    const series = seriesBy([day("2026-10-01", 10, { date: 3, label: 2, Other: 4 })], "model_groups", "spend");
    const [point] = series.data;
    expect(point.date).toBe("2026-10-01");
    expect(point.label).toBe("Oct 1");
    const valueOf = (name: string) => point[series.keys[series.labels.indexOf(name)]];
    expect(valueOf("Other")).toBe(4);
    expect(valueOf("date")).toBe(3);
    expect(valueOf("label")).toBe(2);
    // The real remainder (10 - 9) is a separate trailing series, not merged into the model named "Other".
    expect(series.labels).toHaveLength(4);
    expect(point[series.keys[3]]).toBeCloseTo(1);
  });

  it("folds everything past the top N into a trailing Other in slate", () => {
    const series = seriesBy([day("2026-10-01", 10, { a: 5, b: 3, c: 2 })], "model_groups", "spend", 2);
    expect(series.labels).toEqual(["a", "b", "Other"]);
    expect(series.colors[2]).toBe("#94a3b8");
  });
});

describe("bucketTotals", () => {
  it("keys totals by ISO date, so the same day in two years does not collide", () => {
    const series = seriesBy([day("2025-01-01", 5, { a: 5 }), day("2026-01-01", 9, { a: 9 })], "model_groups", "spend");
    const totals = bucketTotals(series);
    expect(totals.get("2025-01-01")).toBe(5);
    expect(totals.get("2026-01-01")).toBe(9);
    expect(labelForDate(series, "2025-01-01")).toBe("Jan 1");
  });
});

describe("bucketSeries", () => {
  it("sums days into 7-day buckets anchored on the first day", () => {
    const series = seriesBy(
      [day("2026-10-01", 1, { a: 1 }), day("2026-10-03", 2, { a: 2 }), day("2026-10-09", 4, { a: 4 })],
      "model_groups",
      "spend",
    );
    const weekly = bucketSeries(series, "week");
    expect(weekly.data.map((point) => point.date)).toEqual(["2026-10-01", "2026-10-08"]);
    expect(bucketTotals(weekly).get("2026-10-01")).toBe(3);
    expect(bucketTotals(weekly).get("2026-10-08")).toBe(4);
  });
});

describe("rollUpBreakdown", () => {
  it("sums a dimension across days without changing the input", () => {
    const days = [day("2026-10-01", 3, { a: 3 }), day("2026-10-02", 2, { a: 2 })];
    const before = JSON.stringify(days);
    expect(rollUpBreakdown(days, "model_groups")).toEqual([
      expect.objectContaining({ key: "a", spend: 5, requests: 2 }),
    ]);
    expect(JSON.stringify(days)).toBe(before);
  });
});

describe("model colors", () => {
  it("should include deployment IDs without replacing the main chart's group colors or guessing aliases", () => {
    const result = day("2026-10-01", 12, { "claude-sonnet": 10, "gpt-5": 2 });
    const original = modelColorMap([result]);
    result.breakdown.models = {
      "claude-sonnet": result.breakdown.model_groups["claude-sonnet"],
      "provider-deployment-123": result.breakdown.model_groups["gpt-5"],
    };
    const colors = modelColorMap([result]);
    const deployments = seriesBy([result], "models", "spend", { top: 8, colorByKey: colors });

    expect(colors.get("claude-sonnet")).toBe(original.get("claude-sonnet"));
    expect(colors.get("gpt-5")).toBe(original.get("gpt-5"));
    expect(colors.size).toBe(3);
    expect(colors.get("provider-deployment-123")).not.toBe(colors.get("gpt-5"));
    expect(deployments.colors).toEqual([colors.get("claude-sonnet"), colors.get("provider-deployment-123")]);
  });

  it("should preserve the overview's existing spend palette and reuse it when a client ranks models differently", () => {
    const overview = [day("2026-10-01", 15, { "claude-sonnet": 10, "gpt-5": 4, "gemini-pro": 1 })];
    const client = [day("2026-10-01", 5, { "gpt-5": 4, "claude-sonnet": 1 })];
    const colors = modelColorMap(overview);
    const originalSeries = seriesBy(overview, "model_groups", "spend");
    const clientSeries = seriesBy(client, "model_groups", "spend", { top: 8, colorByKey: colors });

    expect(originalSeries.colors).toEqual(originalSeries.labels.map((model) => colors.get(model)));
    expect(clientSeries.labels).toEqual(["gpt-5", "claude-sonnet"]);
    expect(clientSeries.colors).toEqual([colors.get("gpt-5"), colors.get("claude-sonnet")]);
    expect(clientSeries.colors[0]).not.toBe(originalSeries.colors[0]);
    expect(bucketTotals(clientSeries).get("2026-10-01")).toBe(5);
  });

  it("should retain model colors when token ranking and weekly grouping change the chart", () => {
    const result = day("2026-10-01", 12, { "claude-sonnet": 10, "gpt-5": 2 });
    result.breakdown.model_groups["claude-sonnet"].metrics.total_tokens = 20;
    result.breakdown.model_groups["gpt-5"].metrics.total_tokens = 100;
    const colors = modelColorMap([result]);
    const tokens = seriesBy([result], "model_groups", "tokens", { top: 8, colorByKey: colors });
    const weekly = bucketSeries(tokens, "week");

    expect(tokens.labels).toEqual(["gpt-5", "claude-sonnet"]);
    expect(tokens.colors).toEqual([colors.get("gpt-5"), colors.get("claude-sonnet")]);
    expect(weekly.colors).toEqual(tokens.colors);
    expect(bucketTotals(weekly).get("2026-10-01")).toBe(120);
  });

  it("should give a client-only model a deterministic color when it changes rank", () => {
    const colors = modelColorMap([day("2026-10-01", 10, { "claude-sonnet": 10 })]);
    const first = seriesBy([day("2026-10-01", 10, { "claude-sonnet": 8, "new-model": 2 })], "model_groups", "spend", {
      top: 8,
      colorByKey: colors,
    });
    const second = seriesBy([day("2026-10-01", 10, { "new-model": 8, "claude-sonnet": 2 })], "model_groups", "spend", {
      top: 8,
      colorByKey: colors,
    });

    expect(first.colors[1]).toBe(second.colors[0]);
    expect(first.colors[1]).toBe(colorForModel("new-model", colors));
    expect(colors.has("new-model")).toBe(false);
  });

  it("should keep a model named Other distinct from the grey remainder and retain colors outside the top eight", () => {
    const models = { Other: 10, a: 9, b: 8, c: 7, d: 6, e: 5, f: 4, g: 3, h: 2 };
    const results = [day("2026-10-01", 55, models)];
    const colors = modelColorMap(results);
    const series = seriesBy(results, "model_groups", "spend", { top: 8, colorByKey: colors });
    const client = seriesBy([day("2026-10-01", 2, { h: 2 })], "model_groups", "spend", {
      top: 8,
      colorByKey: colors,
    });

    expect(series.labels[0]).toBe("Other");
    expect(series.colors[0]).toBe(colors.get("Other"));
    expect(series.colors[0]).not.toBe(OTHER_COLOR);
    expect(series.labels.at(-1)).toBe("Other");
    expect(series.colors.at(-1)).toBe(OTHER_COLOR);
    expect(series.labels).not.toContain("h");
    expect(colorForModel("h", colors)).toBe(client.colors[0]);
    expect(bucketTotals(series).get("2026-10-01")).toBe(55);
  });
});
