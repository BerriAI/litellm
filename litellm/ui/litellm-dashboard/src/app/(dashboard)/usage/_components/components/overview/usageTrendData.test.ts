import { describe, expect, it } from "vitest";
import type { DailyData, SpendMetrics } from "@/components/UsagePage/types";
import { modelColorMap, OTHER_COLOR } from "./overviewData";
import {
  bucketTickLabel,
  bucketTooltipLabel,
  defaultGranularity,
  formatDayRange,
  observedRate,
  previousPeriod,
  trendBuckets,
  usageTrend,
  type DayRange,
  type TrendGranularity,
} from "./usageTrendData";

interface Usage {
  spend: number;
  tokens: number;
  requests?: number;
  cacheRead?: number;
}

const metricsOf = ({ spend, tokens, requests = 1, cacheRead = 0 }: Usage): SpendMetrics => ({
  spend,
  prompt_tokens: Math.round(tokens * 0.8),
  completion_tokens: tokens - Math.round(tokens * 0.8),
  total_tokens: tokens,
  api_requests: requests,
  successful_requests: requests,
  failed_requests: 0,
  cache_read_input_tokens: cacheRead,
  cache_creation_input_tokens: 0,
});

const sumUsage = (usages: readonly Usage[]): Usage => ({
  spend: usages.reduce((acc, usage) => acc + usage.spend, 0),
  tokens: usages.reduce((acc, usage) => acc + usage.tokens, 0),
  requests: usages.reduce((acc, usage) => acc + (usage.requests ?? 1), 0),
});

const day = (date: string, models: Record<string, Usage>, total: Usage = sumUsage(Object.values(models))): DailyData =>
  ({
    date,
    metrics: metricsOf(total),
    breakdown: {
      models: {},
      model_groups: Object.fromEntries(
        Object.entries(models).map(([name, usage]) => [
          name,
          { metrics: metricsOf(usage), metadata: {}, api_key_breakdown: {} },
        ]),
      ),
      mcp_servers: {},
      providers: {},
      api_keys: {},
      entities: {},
      endpoints: {},
    },
  }) as DailyData;

const trendOf = (results: readonly DailyData[], range: DayRange, granularity: TrendGranularity) => {
  const input = { results, range, granularity, colorByModel: modelColorMap(results) };
  return usageTrend(input);
};

const valueOf = (trend: ReturnType<typeof trendOf>, metric: "tokens" | "spend", bucket: string, label: string) => {
  const series = trend.series[metric];
  const row = series.data.find((point) => point.date === bucket);
  return Number(row?.[series.keys[series.labels.indexOf(label)]] ?? Number.NaN);
};

describe("trendBuckets", () => {
  it("cuts a range across a year boundary into calendar months, flagging the clipped ends as partial", () => {
    const buckets = trendBuckets({ start: "2025-12-15", end: "2026-02-10" }, "month");
    expect(buckets).toEqual([
      { key: "2025-12", start: "2025-12-15", end: "2025-12-31", partial: true },
      { key: "2026-01", start: "2026-01-01", end: "2026-01-31", partial: false },
      { key: "2026-02", start: "2026-02-01", end: "2026-02-10", partial: true },
    ]);
  });

  it("ends February on its real last day in a leap year", () => {
    expect(trendBuckets({ start: "2028-02-01", end: "2028-02-29" }, "month")).toEqual([
      { key: "2028-02", start: "2028-02-01", end: "2028-02-29", partial: false },
    ]);
  });

  it("anchors weeks on the first day and marks a short final week partial", () => {
    const buckets = trendBuckets({ start: "2026-10-01", end: "2026-10-10" }, "week");
    expect(buckets.map(({ start, end, partial }) => [start, end, partial])).toEqual([
      ["2026-10-01", "2026-10-07", false],
      ["2026-10-08", "2026-10-10", true],
    ]);
  });
});

describe("bucket labels", () => {
  it("names the year on the first and January ticks and spells out a partial month's days", () => {
    const buckets = trendBuckets({ start: "2025-11-01", end: "2026-02-12" }, "month");
    expect(buckets.map((bucket, index) => bucketTickLabel(bucket, "month", index === 0))).toEqual([
      "Nov 2025",
      "Dec",
      "Jan 2026",
      "Feb 1–12",
    ]);
  });

  it("adds years to a range label only when it spans two years", () => {
    expect(formatDayRange({ start: "2026-10-01", end: "2026-10-12" })).toBe("Oct 1 – Oct 12");
    expect(formatDayRange({ start: "2025-12-30", end: "2026-01-02" })).toBe("Dec 30, 2025 – Jan 2, 2026");
  });
});

describe("bucketTooltipLabel", () => {
  it("names the day, the week with its length, the month, and a clipped month's days", () => {
    const [oneDay] = trendBuckets({ start: "2026-10-03", end: "2026-10-03" }, "day");
    const [week] = trendBuckets({ start: "2026-10-01", end: "2026-10-10" }, "week");
    const [fullMonth, clippedMonth] = trendBuckets({ start: "2026-10-01", end: "2026-11-12" }, "month");

    expect(bucketTooltipLabel(oneDay, "day")).toBe("Sat, Oct 3, 2026");
    expect(bucketTooltipLabel(week, "week")).toBe("Week of Oct 1 – Oct 7 (7 days)");
    expect(bucketTooltipLabel(fullMonth, "month")).toBe("October 2026");
    expect(bucketTooltipLabel(clippedMonth, "month")).toBe("November 2026, Nov 1 – Nov 12 only");
  });
});

describe("defaultGranularity", () => {
  it.each([
    [{ start: "2026-10-13", end: "2026-11-12" }, "day"],
    [{ start: "2026-09-13", end: "2026-11-12" }, "month"],
    [{ start: "2026-10-01", end: "2026-11-05" }, "week"],
    [{ start: "2025-12-01", end: "2026-11-12" }, "month"],
  ] as const)("picks %o -> %s", (range, granularity) => {
    expect(defaultGranularity(range)).toBe(granularity);
  });
});

describe("previousPeriod", () => {
  it.each([
    ["month to date meets the same elapsed days of last month", "2026-11-01:2026-11-12", "2026-10-01:2026-10-12"],
    ["a full month meets the full month before", "2026-03-01:2026-03-31", "2026-02-01:2026-02-28"],
    ["whole months across New Year meet whole months", "2026-01-01:2026-02-28", "2025-11-01:2025-12-31"],
    ["an elapsed day past the shorter month's end is clamped", "2026-03-01:2026-03-30", "2026-02-01:2026-02-28"],
    ["twelve calendar months to date meet the twelve before", "2025-12-01:2026-11-12", "2024-12-01:2025-11-12"],
    ["a rolling range meets the same number of days before it", "2026-11-05:2026-11-11", "2026-10-29:2026-11-04"],
    ["a rolling range across New Year", "2026-01-02:2026-01-05", "2025-12-29:2026-01-01"],
  ])("%s", (_name, current, previous) => {
    const [start, end] = current.split(":");
    const [previousStart, previousEnd] = previous.split(":");
    expect(previousPeriod({ start, end })).toEqual({ start: previousStart, end: previousEnd });
  });
});

describe("observedRate", () => {
  it("reports spend per million tokens only when tokens were recorded", () => {
    expect(observedRate(3, 2_000_000)).toEqual({ kind: "rate", usdPerMillion: 1.5 });
    expect(observedRate(0.04, 0)).toEqual({ kind: "noTokens" });
    expect(observedRate(0, 5000)).toEqual({ kind: "noSpend" });
    expect(observedRate(0, 0)).toEqual({ kind: "none" });
  });
});

describe("usageTrend", () => {
  const models = Object.fromEntries(
    Array.from({ length: 10 }, (_, index) => [
      `model-${index}`,
      { spend: 100 - index * 5, tokens: 1000 * (index + 1) },
    ]),
  );

  it("stacks tokens and spend by the same eight leaders in the same order and colors, the rest as Other", () => {
    const results = [day("2026-10-03", models), day("2026-11-02", models)];
    const trend = trendOf(results, { start: "2026-10-01", end: "2026-11-12" }, "month");

    const expectedLeaders = Array.from({ length: 8 }, (_, index) => `model-${index}`);
    expect(trend.series.tokens.labels).toEqual([...expectedLeaders, "Other"]);
    expect(trend.series.spend.labels).toEqual(trend.series.tokens.labels);
    expect(trend.series.spend.colors).toEqual(trend.series.tokens.colors);
    expect(trend.series.requests.labels).toEqual(trend.series.tokens.labels);
    expect(trend.series.tokens.colors.at(-1)).toBe(OTHER_COLOR);
    expect(new Set(trend.series.tokens.colors.slice(0, 8)).size).toBe(8);

    expect(valueOf(trend, "tokens", "2026-10", "Other")).toBe(19000);
    expect(valueOf(trend, "spend", "2026-11", "Other")).toBe(115);
  });

  it("keeps every model in the table and reconciles each chart's bar totals with the table total", () => {
    const results = [day("2026-10-03", models), day("2026-11-02", models)];
    const trend = trendOf(results, { start: "2026-10-01", end: "2026-11-12" }, "month");

    expect(trend.rows.map((row) => row.key)).toHaveLength(10);
    expect(trend.rows.filter((row) => row.grouped).map((row) => row.key)).toEqual(["model-8", "model-9"]);
    expect(trend.rows.filter((row) => row.grouped).every((row) => row.color === OTHER_COLOR)).toBe(true);
    const barSum = (metric: "tokens" | "spend") => [...trend.totals[metric].values()].reduce((a, b) => a + b, 0);
    expect(barSum("tokens")).toBe(trend.total.tokens);
    expect(barSum("spend")).toBeCloseTo(trend.total.spend, 10);
    expect(trend.rows.reduce((acc, row) => acc + (row.spendShare ?? 0), 0)).toBeCloseTo(100, 10);
    expect(trend.unattributed).toEqual({ spend: 0, tokens: 0, requests: 0 });
  });

  it("keeps the same leaders and colors when the mix shifts between months", () => {
    const october = day("2026-10-03", { alpha: { spend: 90, tokens: 9000 }, beta: { spend: 10, tokens: 1000 } });
    const november = day("2026-11-02", { alpha: { spend: 5, tokens: 500 }, beta: { spend: 80, tokens: 8000 } });
    const trend = trendOf([october, november], { start: "2026-10-01", end: "2026-11-12" }, "month");

    expect(trend.series.spend.labels).toEqual(["alpha", "beta"]);
    expect(trend.series.tokens.labels).toEqual(["alpha", "beta"]);
    expect(trend.series.tokens.colors).toEqual(trend.series.spend.colors);
    expect(trend.series.tokens.data.map((point) => point.date)).toEqual(["2026-10", "2026-11"]);
  });

  it("counts total tokens once, with cached prompt tokens already inside them", () => {
    const results = [day("2026-10-03", { cached: { spend: 1, tokens: 1000, cacheRead: 600 } })];
    const trend = trendOf(results, { start: "2026-10-01", end: "2026-10-31" }, "month");

    expect(trend.totals.tokens.get("2026-10")).toBe(1000);
    expect(trend.total.tokens).toBe(1000);
  });

  it("shows a month with no activity as an empty bucket rather than dropping it", () => {
    const results = [
      day("2026-08-05", { a: { spend: 2, tokens: 10 } }),
      day("2026-10-05", { a: { spend: 4, tokens: 20 } }),
    ];
    const trend = trendOf(results, { start: "2026-08-01", end: "2026-10-31" }, "month");

    expect([...trend.totals.spend.entries()]).toEqual([
      ["2026-08", 2],
      ["2026-09", 0],
      ["2026-10", 4],
    ]);
  });

  it("buckets days at a year boundary into their own year's month", () => {
    const results = [
      day("2025-12-31", { a: { spend: 1, tokens: 10 } }),
      day("2026-01-01", { a: { spend: 3, tokens: 30 } }),
    ];
    const trend = trendOf(results, { start: "2025-12-01", end: "2026-01-31" }, "month");

    expect(trend.totals.spend.get("2025-12")).toBe(1);
    expect(trend.totals.spend.get("2026-01")).toBe(3);
  });

  it("keeps a day the server returned outside the requested range instead of dropping its spend", () => {
    const results = [day("2026-11-13", { a: { spend: 7, tokens: 70 } })];
    const trend = trendOf(results, { start: "2026-11-01", end: "2026-11-12" }, "day");

    expect(trend.totals.spend.get("2026-11-13")).toBe(7);
    expect(trend.total.spend).toBe(7);
  });

  it("states a gap between day totals and the model breakdown instead of drawing it as Other", () => {
    const results = [day("2026-10-03", { a: { spend: 6, tokens: 600 } }, { spend: 10, tokens: 1000 })];
    const trend = trendOf(results, { start: "2026-10-01", end: "2026-10-31" }, "month");

    expect(trend.series.spend.labels).toEqual(["a"]);
    expect(trend.totals.spend.get("2026-10")).toBe(6);
    expect(trend.unattributed.spend).toBeCloseTo(4, 10);
    expect(trend.unattributed.tokens).toBe(400);
  });

  it("gives zero-token spend and zero-spend tokens an honest rate and no share of an empty total", () => {
    const mixed = { images: { spend: 0.4, tokens: 0 }, subscription: { spend: 0, tokens: 5000 } };
    const results = [day("2026-10-03", mixed)];
    const trend = trendOf(results, { start: "2026-10-01", end: "2026-10-31" }, "month");
    const row = (key: string) => trend.rows.find((candidate) => candidate.key === key);

    expect(row("images")?.rate).toEqual({ kind: "noTokens" });
    expect(row("images")?.tokenShare).toBe(0);
    expect(row("subscription")?.rate).toEqual({ kind: "noSpend" });
    expect(row("subscription")?.spendShare).toBe(0);

    const october = { start: "2026-10-01", end: "2026-10-31" };
    const tokenless = trendOf([day("2026-10-03", { images: { spend: 0.4, tokens: 0 } })], october, "month");
    expect(tokenless.rows[0].tokenShare).toBeNull();
    expect(tokenless.total.tokenShare).toBeNull();
  });

  it("stacks a client's deployments from the models breakdown when asked for that dimension", () => {
    const client = day("2026-10-03", {});
    const withModels = {
      ...client,
      metrics: metricsOf({ spend: 3, tokens: 300 }),
      breakdown: {
        ...client.breakdown,
        models: { "claude-x": { metrics: metricsOf({ spend: 3, tokens: 300 }), metadata: {}, api_key_breakdown: {} } },
      },
    } as DailyData;
    const october = { start: "2026-10-01", end: "2026-10-31" };
    const input = {
      results: [withModels],
      range: october,
      granularity: "month",
      colorByModel: new Map<string, string>(),
      dimension: "models",
    } as const;
    const trend = usageTrend(input);

    expect(trend.series.spend.labels).toEqual(["claude-x"]);
    expect(trend.totals.tokens.get("2026-10")).toBe(300);
    expect(trend.unattributed).toEqual({ spend: 0, tokens: 0, requests: 0 });
  });

  it("ranks leaders by the charted metric while keeping each model's color", () => {
    const results = [
      day("2026-10-03", {
        pricey: { spend: 50, tokens: 100 },
        chatty: { spend: 1, tokens: 9000 },
        image: { spend: 5, tokens: 0 },
      }),
    ];
    const colorByModel = modelColorMap(results);
    const ranked = (rankBy: "tokens" | "spend") => {
      const input = {
        results,
        range: { start: "2026-10-01", end: "2026-10-31" },
        granularity: "month",
        colorByModel,
        rankBy,
      } as const;
      return usageTrend(input);
    };

    expect(ranked("tokens").series.tokens.labels).toEqual(["chatty", "pricey"]);
    expect(ranked("spend").series.spend.labels).toEqual(["pricey", "image", "chatty"]);
    expect(ranked("tokens").rows.map((row) => row.key)).toEqual(["chatty", "pricey", "image"]);
    const colorOf = (rankBy: "tokens" | "spend", label: string) => {
      const series = ranked(rankBy).series[rankBy];
      return series.colors[series.labels.indexOf(label)];
    };
    expect(colorOf("tokens", "chatty")).toBe(colorOf("spend", "chatty"));
  });

  it("omits Other entirely when every model fits among the leaders", () => {
    const october = { start: "2026-10-01", end: "2026-10-31" };
    const trend = trendOf([day("2026-10-03", { a: { spend: 1, tokens: 1 } })], october, "month");
    expect(trend.series.tokens.labels).toEqual(["a"]);
  });
});
