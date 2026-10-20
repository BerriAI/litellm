import { describe, expect, it } from "vitest";

import { buildBucketTotals, buildSeries, DailyMetric, formatMetric, modelOrder, rankModels } from "./modelInsightsData";

const row = (over: Partial<DailyMetric>): DailyMetric => ({
  model_group: "a",
  model: "a",
  provider: "openai",
  date: "2026-01-01",
  spend: 0,
  prompt_tokens: 0,
  completion_tokens: 0,
  requests: 0,
  successful_requests: 0,
  failed_requests: 0,
  ...over,
});

describe("buildSeries", () => {
  const range = { start: "2026-01-01", end: "2026-01-15" };

  it("sums days into 7-day buckets per model", () => {
    const rows = [
      row({ date: "2026-01-01", requests: 1 }),
      row({ date: "2026-01-07", requests: 2 }),
      row({ date: "2026-01-08", requests: 4 }),
      row({ date: "2026-01-02", model_group: "b", requests: 8 }),
    ];
    expect(buildSeries(rows, ["a", "b"], "requests", { ...range, granularity: "week" })).toEqual([
      { date: "2026-01-01", a: 3, b: 8 },
      { date: "2026-01-08", a: 4, b: 0 },
      { date: "2026-01-15", a: 0, b: 0 },
    ]);
  });

  it("keeps weeks with no usage as zero instead of dropping them", () => {
    const rows = [row({ date: "2026-01-01", requests: 1 }), row({ date: "2026-01-15", requests: 2 })];
    expect(
      buildSeries(rows, ["a"], "requests", { ...range, granularity: "week" }).map((week) => [week.date, week.a]),
    ).toEqual([
      ["2026-01-01", 1],
      ["2026-01-08", 0],
      ["2026-01-15", 2],
    ]);
  });

  it("gives every day its own bucket with that day's token total", () => {
    const rows = [
      row({ date: "2026-01-01", prompt_tokens: 100, completion_tokens: 50 }),
      row({ date: "2026-01-01", prompt_tokens: 10, completion_tokens: 5 }),
      row({ date: "2026-01-03", prompt_tokens: 7, completion_tokens: 3 }),
    ];
    const daily = buildSeries(rows, ["a"], "tokens", { start: "2026-01-01", end: "2026-01-03", granularity: "day" });
    expect(daily).toEqual([
      { date: "2026-01-01", a: 165 },
      { date: "2026-01-02", a: 0 },
      { date: "2026-01-03", a: 10 },
    ]);
  });

  it("starts the chart at the first bucket with usage so bars stay wide on a long range", () => {
    const rows = [row({ date: "2026-03-10", requests: 1 }), row({ date: "2026-03-20", requests: 2 })];
    const daily = buildSeries(rows, ["a"], "requests", { start: "2025-03-21", end: "2026-03-20", granularity: "day" });
    expect(daily[0]).toEqual({ date: "2026-02-19", a: 0 });
    expect(daily).toHaveLength(30);
    expect(daily.at(-1)).toEqual({ date: "2026-03-20", a: 2 });

    const early = [row({ date: "2025-12-01", requests: 1 }), ...rows];
    const fromFirstUse = buildSeries(early, ["a"], "requests", {
      start: "2025-03-21",
      end: "2026-03-20",
      granularity: "day",
    });
    expect(fromFirstUse[0]).toEqual({ date: "2025-12-01", a: 1 });
    expect(fromFirstUse.at(-1)?.date).toBe("2026-03-20");
  });

  it("keeps weekly buckets on the original grid when trimming idle weeks", () => {
    const rows = [row({ date: "2026-03-20", requests: 3 })];
    const weekly = buildSeries(rows, ["a"], "requests", {
      start: "2025-03-21",
      end: "2026-03-20",
      granularity: "week",
    });
    expect(weekly).toHaveLength(12);
    expect(weekly.map((week) => (Date.parse(String(week.date)) - Date.parse("2025-03-21")) % (7 * 86_400_000))).toEqual(
      Array(12).fill(0),
    );
    expect(weekly.at(-1)).toEqual({ date: "2026-03-20", a: 3 });
  });
});

describe("buildBucketTotals", () => {
  const total = (date: string, prompt_tokens: number) => ({
    date,
    spend: 0,
    prompt_tokens,
    completion_tokens: 1,
    requests: 0,
  });
  const totals = [total("2026-01-01", 9), total("2026-01-03", 4), total("2026-01-08", 99)];

  it("keys each day's gateway-wide total by its own date", () => {
    const daily = buildBucketTotals(totals, "tokens", { start: "2026-01-01", end: "2026-01-08", granularity: "day" });
    expect([...daily]).toEqual([
      ["2026-01-01", 10],
      ["2026-01-03", 5],
      ["2026-01-08", 100],
    ]);
  });

  it("sums days into the same week start used by the chart's x-axis", () => {
    const window = { start: "2026-01-01", end: "2026-01-08", granularity: "week" } as const;
    const weekly = buildBucketTotals(totals, "tokens", window);
    expect([...weekly]).toEqual([
      ["2026-01-01", 15],
      ["2026-01-08", 100],
    ]);
    expect([...weekly.keys()]).toEqual(buildSeries([], [], "tokens", window).map((bucket) => bucket.date));
  });
});

describe("modelOrder", () => {
  it("orders models by the selected metric, largest first", () => {
    const rows = [row({ model_group: "a", spend: 1, requests: 9 }), row({ model_group: "b", spend: 5, requests: 1 })];
    expect(modelOrder(rows, "spend")).toEqual(["b", "a"]);
    expect(modelOrder(rows, "requests")).toEqual(["a", "b"]);
  });
});

describe("rankModels", () => {
  const range = { start: "2026-01-01", end: "2026-01-10" };
  const totals = [row({ model_group: "a", requests: 40 }), row({ model_group: "b", requests: 40 })];

  it("computes share and the change in share between the first and second half of the range", () => {
    const daily = [
      row({ date: "2026-01-01", model_group: "a", requests: 30 }),
      row({ date: "2026-01-01", model_group: "b", requests: 10 }),
      row({ date: "2026-01-10", model_group: "a", requests: 10 }),
      row({ date: "2026-01-10", model_group: "b", requests: 30 }),
    ];
    const ranked = rankModels(totals, daily, "requests", range);
    expect(ranked.find((m) => m.model_group === "a")).toMatchObject({ share: 50, delta: -50 });
    expect(ranked.find((m) => m.model_group === "b")).toMatchObject({ share: 50, delta: 50 });
  });

  it("splits at the middle of the range, not the middle of the days that had usage", () => {
    const daily = [
      row({ date: "2026-01-01", model_group: "a", requests: 10 }),
      row({ date: "2026-01-02", model_group: "b", requests: 10 }),
      row({ date: "2026-01-03", model_group: "b", requests: 10 }),
    ];
    const ranked = rankModels(totals, daily, "requests", range);
    expect(ranked.find((m) => m.model_group === "a")?.delta).toBe(0);
  });

  it("shows no change when one half of the range has no usage to compare against", () => {
    const daily = [row({ date: "2026-01-10", model_group: "a", requests: 10 })];
    const ranked = rankModels(totals, daily, "requests", range);
    expect(ranked.map((m) => m.delta)).toEqual([0, 0]);
  });
});

describe("formatMetric", () => {
  it("formats spend as currency and counts compactly", () => {
    expect(formatMetric(12.5, "spend")).toBe("$12.50");
    expect(formatMetric(1_500_000, "tokens")).toBe("1.5M");
  });
});
