import { describe, expect, it } from "vitest";
import {
  bucketTotals,
  OTHER_COLOR,
  type Series,
  type UsageMetric,
} from "@/app/(dashboard)/usage/_components/components/overview/overviewData";
import { stackedUsageColor } from "@/components/shared/charts";
import { EMPTY_DAILY_ACTIVITY_METADATA } from "@/components/UsagePage/dailyActivityApi";
import type { DailyData, MetricWithMetadata, SpendMetrics } from "@/components/UsagePage/types";
import i18n from "@/i18n/i18n";
import type { CustomerHierarchyUsage } from "./customerUsageApi";
import { customerUsageSeries } from "./customerUsageChartData";
import type { CustomerUsageDimension } from "./CustomerUsageRankings";

const metrics = (spend: number, tokens: number, requests: number): SpendMetrics => ({
  spend,
  prompt_tokens: tokens,
  completion_tokens: 0,
  total_tokens: tokens,
  api_requests: requests,
  successful_requests: requests,
  failed_requests: 0,
  cache_read_input_tokens: 0,
  cache_creation_input_tokens: 0,
});

const entry = (spend: number, tokens: number, requests: number): MetricWithMetadata => ({
  metrics: metrics(spend, tokens, requests),
  metadata: {},
  api_key_breakdown: {},
});

const day = (date: string, total: SpendMetrics, breakdown: Partial<DailyData["breakdown"]> = {}): DailyData => ({
  date,
  metrics: total,
  breakdown: { models: {}, model_groups: {}, providers: {}, mcp_servers: {}, api_keys: {}, entities: {}, ...breakdown },
});

const snapshot = (): CustomerHierarchyUsage => ({
  organization: { id: "org", label: "Organization", spend: 0, budgetLimit: null, budgetResetAt: null },
  team: null,
  member: null,
  memberBudgets: [],
  metadata: { ...EMPTY_DAILY_ACTIVITY_METADATA },
  results: [
    day("2026-11-01", metrics(100, 1_000, 100), {
      models: { deployed: entry(100, 1_000, 100) },
      model_groups: { alpha: entry(60, 300, 20), beta: entry(40, 700, 80) },
      api_keys: {
        "hash-a": { metrics: metrics(30, 800, 70), metadata: { key_alias: "Support", team_id: null } },
        "hash-b": { metrics: metrics(70, 200, 30), metadata: { key_alias: "Search", team_id: null } },
      },
    }),
    day("2026-11-02", metrics(60, 900, 30)),
    day("2026-11-03", metrics(50, 500, 40)),
  ],
  teamResults: [
    day("2026-11-01", metrics(75, 800, 60), {
      entities: { "team-a": entry(25, 100, 40), "team-b": entry(50, 700, 20) },
    }),
    day("2026-11-02", metrics(10, 90, 5), { entities: { "team-a": entry(10, 90, 5) } }),
    day("2026-11-09", metrics(999, 999, 999), { entities: { "team-a": entry(999, 999, 999) } }),
  ],
  memberResults: [
    day("2026-11-01", metrics(80, 700, 80), {
      entities: { "member-a": entry(20, 200, 30), "member-b": entry(60, 500, 50) },
    }),
  ],
  teams: [
    { id: "team-a", label: "Product", spend: 35, tokens: 190, requests: 45 },
    { id: "team-b", label: "Operations", spend: 50, tokens: 700, requests: 20 },
  ],
  members: [
    { id: "member-a", label: "Member A", spend: 20, tokens: 200, requests: 30 },
    { id: "member-b", label: "Member B", spend: 60, tokens: 500, requests: 50 },
  ],
  api_keys: [
    { id: "hash-a", label: "Support", spend: 30, tokens: 800, requests: 70 },
    { id: "hash-b", label: "Search", spend: 70, tokens: 200, requests: 30 },
  ],
});

const modelColors = new Map([
  ["alpha", stackedUsageColor(3)],
  ["beta", stackedUsageColor(4)],
]);
const valueAt = (series: Series, label: string, date = "2026-11-01"): number => {
  const index = series.labels.indexOf(label);
  return Number(series.data.find((row) => row.date === date)?.[series.keys[index]] ?? 0);
};
const colorOf = (series: Series, label: string): string => series.colors[series.labels.indexOf(label)];
const other = () => i18n.t("usage.overview.otherSeries");

describe("customerUsageSeries", () => {
  it.each<[CustomerUsageDimension, string, number]>([
    ["models", "alpha", 60],
    ["api_keys", "Support", 30],
    ["teams", "Product", 25],
  ])("should use %s values without adding the other breakdowns", (dimension, label, value) => {
    const series = customerUsageSeries(snapshot(), dimension, "spend", modelColors);

    expect(valueAt(series, label)).toBe(value);
    expect(bucketTotals(series).get("2026-11-01")).toBe(100);
    expect(series.labels).not.toContain("deployed");
  });

  it.each<[UsageMetric, number, number, number]>([
    ["spend", 25, 50, 25],
    ["tokens", 100, 700, 200],
    ["requests", 40, 20, 40],
  ])("should use %s for both named teams and the unattributed remainder", (metric, first, second, remainder) => {
    const series = customerUsageSeries(snapshot(), "teams", metric, modelColors);

    expect(valueAt(series, "Product")).toBe(first);
    expect(valueAt(series, "Operations")).toBe(second);
    expect(valueAt(series, other())).toBe(remainder);
    expect(colorOf(series, other())).toBe(OTHER_COLOR);
  });

  it("should preserve all selected-scope dates and totals when companion records are missing", () => {
    const series = customerUsageSeries(snapshot(), "teams", "spend", modelColors);

    expect(series.data.map((row) => row.date)).toEqual(["2026-11-01", "2026-11-02", "2026-11-03"]);
    expect(valueAt(series, "Product", "2026-11-02")).toBe(10);
    expect(valueAt(series, other(), "2026-11-02")).toBe(50);
    expect(valueAt(series, "Product", "2026-11-03")).toBe(0);
    expect(valueAt(series, other(), "2026-11-03")).toBe(50);
    expect([...bucketTotals(series).values()]).toEqual([100, 60, 50]);
  });

  it("should include removed and unnamed descendants in Other rather than a misleading named bucket", () => {
    const data = snapshot();
    const removed = day("2026-11-01", metrics(100, 1_000, 100), {
      entities: { "team-a": entry(25, 100, 40), removed: entry(10, 150, 10), "": entry(5, 50, 5) },
    });
    const series = customerUsageSeries({ ...data, teamResults: [removed] }, "teams", "tokens", modelColors);

    expect(series.labels).toEqual(["Product", other()]);
    expect(valueAt(series, "Product")).toBe(100);
    expect(valueAt(series, other())).toBe(900);
  });

  it("should keep spend-ranked palette colors attached to API keys when the selected metric changes", () => {
    const spend = customerUsageSeries(snapshot(), "api_keys", "spend", modelColors);
    const tokens = customerUsageSeries(snapshot(), "api_keys", "tokens", modelColors);

    expect(spend.labels.slice(0, 2)).toEqual(["Search", "Support"]);
    expect(tokens.labels.slice(0, 2)).toEqual(["Support", "Search"]);
    expect(colorOf(spend, "Search")).toBe(stackedUsageColor(0));
    expect(colorOf(tokens, "Search")).toBe(stackedUsageColor(0));
    expect(colorOf(spend, "Support")).toBe(stackedUsageColor(1));
    expect(colorOf(tokens, "Support")).toBe(stackedUsageColor(1));
  });

  it("should use key aliases and hide hashes even when a label is missing or repeats the hash", () => {
    const data = snapshot();
    const series = customerUsageSeries(
      { ...data, api_keys: [{ ...data.api_keys[0], label: " hash-a " }] },
      "api_keys",
      "spend",
      modelColors,
    );

    expect(series.labels).not.toContain("hash-a");
    expect(series.labels).not.toContain("hash-b");
    expect(series.labels.filter((label) => label === i18n.t("customerPortal.analytics.unnamedKey"))).toHaveLength(2);
    expect(bucketTotals(series).get("2026-11-01")).toBe(100);
  });

  it("should preserve model colors and fall back to the models breakdown when groups are absent", () => {
    const data = snapshot();
    const color = stackedUsageColor(5);
    const series = customerUsageSeries(
      { ...data, results: [day("2026-11-01", metrics(7, 70, 2), { models: { deployed: entry(7, 70, 2) } })] },
      "models",
      "tokens",
      new Map([["deployed", color]]),
    );

    expect(series.labels).toEqual(["deployed"]);
    expect(valueAt(series, "deployed")).toBe(70);
    expect(colorOf(series, "deployed")).toBe(color);
  });
});
