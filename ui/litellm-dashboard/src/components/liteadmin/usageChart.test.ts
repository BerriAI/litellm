import { describe, expect, it } from "vitest";
import {
  buildUsageChart,
  formatUsageChartValue,
  usageChartRequestSchema,
  usageChartSource,
  type UsageChartSource,
} from "./usageChart";

const query = { start_date: "2031-05-01", end_date: "2031-05-02" };
const admin = { mode: "admin" } as const;
const report = { daily: [{ date: query.start_date, spend: 0 }], top_models: [], top_providers: [] };
const source = (data: UsageChartSource["report"]): UsageChartSource => ({
  report: data,
  period: `${query.start_date} to ${query.end_date} (UTC)`,
  scope: "Gateway usage",
});

describe("usage chart sources", () => {
  it("should retain validated usage data with its date range and account scope", () => {
    expect(usageChartSource(report, query, admin)).toEqual(source(report));
    expect(
      usageChartSource(
        report,
        { ...query, user_id: "viewer" },
        { mode: "usage", scope: { type: "user", userId: "viewer" } },
      ),
    ).toMatchObject({ scope: "Your usage" });
  });

  it.each(["All teams", "All tags"] as const)(
    "should label an unfiltered %s report and retain combined filters",
    (scope) => {
      expect(usageChartSource(report, query, admin, scope)?.scope).toBe(scope);
      const filters = { ...query, user_id: "member", team_ids: "team-a", tags: "production" };
      expect(usageChartSource(report, filters, admin, scope)?.scope).toBe(
        "User member · Teams team-a · Tags production",
      );
    },
  );

  it.each([-1, Number.NaN, Number.POSITIVE_INFINITY, "12"])("should reject invalid metric %s", (spend) => {
    expect(usageChartSource({ ...report, daily: [{ date: query.start_date, spend }] }, query, admin)).toBeUndefined();
  });

  it("should reject malformed reports or date ranges before creating a chart source", () => {
    expect(usageChartSource({ daily: report.daily }, query, admin)).toBeUndefined();
    expect(usageChartSource({ ...report, daily: [{ date: "yesterday", spend: 2 }] }, query, admin)).toBeUndefined();
    expect(usageChartSource(report, { ...query, start_date: "2031-02-30" }, admin)).toBeUndefined();
  });

  it("should accept only a source reference and a supported view in chart requests", () => {
    const request = { source_id: "recorded-lookup", view: "daily_spend" };
    expect(usageChartRequestSchema.safeParse(request).success).toBe(true);
    expect(usageChartRequestSchema.safeParse({ ...request, data: [{ value: 999 }] }).success).toBe(false);
    expect(usageChartRequestSchema.safeParse({ ...request, source_id: "" }).success).toBe(false);
    expect(usageChartRequestSchema.safeParse({ ...request, view: "arbitrary_chart" }).success).toBe(false);
  });
});

describe("usage chart values", () => {
  it.each([
    ["daily_spend", "spend", "usd"],
    ["daily_requests", "api_requests", "requests"],
    ["daily_tokens", "total_tokens", "tokens"],
  ] as const)(
    "should sort %s chronologically and preserve missing values separately from zero",
    (view, metric, unit) => {
      const data = {
        ...report,
        daily: [
          { date: "2031-05-03", [metric]: 7 },
          { date: "2031-05-01", [metric]: 0 },
          { date: "2031-05-02", [metric]: null },
        ],
      };
      const chart = buildUsageChart(source(data), view);
      const expectedChart = {
        kind: "line",
        unit,
        total: null,
        data: [
          { label: "2031-05-01", value: 0 },
          { label: "2031-05-02", value: null },
          { label: "2031-05-03", value: 7 },
        ],
      };
      expect(chart).toMatchObject(expectedChart);
      expect(data.daily[0].date).toBe("2031-05-03");
    },
  );

  it("should preserve independent prompt, completion and cache totals without adding cache to total tokens", () => {
    const chart = buildUsageChart(
      source({
        ...report,
        daily: [
          {
            date: query.start_date,
            total_tokens: 50,
            prompt_tokens: 40,
            completion_tokens: 10,
            cache_read_input_tokens: 15,
            cache_creation_input_tokens: 5,
          },
          {
            date: query.end_date,
            total_tokens: 75,
            prompt_tokens: 60,
            completion_tokens: 15,
            cache_read_input_tokens: 20,
            cache_creation_input_tokens: 10,
          },
        ],
      }),
      "token_breakdown",
    );
    const expectedChart = {
      kind: "bar",
      unit: "tokens",
      total: 125,
      data: [
        { label: "Prompt", value: 100 },
        { label: "Completion", value: 25 },
        { label: "Cache read", value: 35 },
        { label: "Cache creation", value: 15 },
      ],
    };
    expect(chart).toMatchObject(expectedChart);
  });

  it("should leave an incomplete token counter unavailable while retaining known zeros", () => {
    const chart = buildUsageChart(
      source({
        ...report,
        daily: [
          { date: query.start_date, total_tokens: 8, prompt_tokens: 8, cache_read_input_tokens: 0 },
          { date: query.end_date, total_tokens: 12, prompt_tokens: null, cache_read_input_tokens: 0 },
        ],
      }),
      "token_breakdown",
    );
    expect(chart?.total).toBe(20);
    expect(chart?.data).toEqual([
      { label: "Prompt", value: null },
      { label: "Completion", value: null },
      { label: "Cache read", value: 0 },
      { label: "Cache creation", value: null },
    ]);
  });

  it.each([
    ["model_spend", "top_models"],
    ["provider_spend", "top_providers"],
  ] as const)("should show the six leading entries and all remaining recorded spend for %s", (view, dimension) => {
    const ranked = [5, 20, 1, 10, 15, 25, 2, 4].map((spend) => ({ name: `entry-${spend}`, spend }));
    const chart = buildUsageChart(
      source({ ...report, daily: [{ date: query.start_date, spend: 100 }], [dimension]: ranked }),
      view,
    );
    expect(chart?.data).toEqual([
      { label: "entry-25", value: 25 },
      { label: "entry-20", value: 20 },
      { label: "entry-15", value: 15 },
      { label: "entry-10", value: 10 },
      { label: "entry-5", value: 5 },
      { label: "entry-4", value: 4 },
      { label: "Other", value: 21 },
    ]);
    expect(chart?.data.reduce((sum, row) => sum + (row.value ?? 0), 0)).toBe(chart?.total);
    expect(ranked[0].spend).toBe(5);
  });

  it("should not invent remaining spend when the daily total is incomplete", () => {
    const chart = buildUsageChart(
      source({
        ...report,
        daily: [{ date: query.start_date, spend: null }],
        top_models: [{ name: "Known model", spend: 5 }],
      }),
      "model_spend",
    );
    expect(chart?.total).toBeNull();
    expect(chart?.data).toEqual([{ label: "Known model", value: 5 }]);
  });

  it.each(["daily_spend", "model_spend", "provider_spend", "token_breakdown"] as const)(
    "should omit %s when its source has no recorded rows",
    (view) => expect(buildUsageChart(source({ daily: [], top_models: [], top_providers: [] }), view)).toBeUndefined(),
  );

  it("should display tiny dollar amounts without rounding them to zero and label unavailable values", () => {
    expect(formatUsageChartValue(0.000123, "usd")).toBe("$0.000123");
    expect(formatUsageChartValue(0, "usd")).toBe("$0.00");
    expect(formatUsageChartValue(null, "tokens")).toBe("Unavailable");
    expect(formatUsageChartValue(12345, "requests")).toBe("12,345");
  });
});
