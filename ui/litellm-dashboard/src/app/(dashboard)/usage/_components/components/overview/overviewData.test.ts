import { describe, expect, it } from "vitest";
import type { DailyData } from "@/components/UsagePage/types";
import { bucketSeries, bucketTotals, labelForDate, rollUpBreakdown, seriesBy } from "./overviewData";

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
