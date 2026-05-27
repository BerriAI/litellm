import { describe, expect, it } from "vitest";

import type { DailyData } from "../types";
import {
  SINGLE_DAY_TIME_LABEL,
  collapseDailyResults,
  getDailySpendChartData,
  isSingleDayRange,
} from "./dailySpendChart";

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

function row(date: string, partial: Partial<typeof baseMetrics> = {}): DailyData {
  return {
    date,
    metrics: { ...baseMetrics, ...partial },
    breakdown: {
      models: {},
      model_groups: {},
      mcp_servers: {},
      providers: {},
      api_keys: {},
      entities: {},
    },
  };
}

describe("isSingleDayRange", () => {
  it("returns true when from and to are on the same local calendar day", () => {
    const from = new Date(2026, 4, 26, 0, 0, 0);
    const to = new Date(2026, 4, 26, 23, 59, 59);
    expect(isSingleDayRange({ from, to })).toBe(true);
  });

  it("returns false when from and to span two calendar days", () => {
    const from = new Date(2026, 4, 26, 0, 0, 0);
    const to = new Date(2026, 4, 27, 0, 0, 0);
    expect(isSingleDayRange({ from, to })).toBe(false);
  });

  it("returns false when either end is missing", () => {
    expect(isSingleDayRange(undefined)).toBe(false);
    expect(isSingleDayRange(null)).toBe(false);
    expect(isSingleDayRange({ from: new Date(2026, 4, 26), to: undefined })).toBe(false);
    expect(isSingleDayRange({ from: undefined, to: new Date(2026, 4, 26) })).toBe(false);
  });

  it("returns false across midnight when the calendar dates differ", () => {
    const from = new Date(2026, 2, 7, 23, 30, 0);
    const to = new Date(2026, 2, 8, 0, 30, 0);
    expect(isSingleDayRange({ from, to })).toBe(false);
  });
});

describe("collapseDailyResults", () => {
  it("sums metrics across rows and uses the supplied label as date", () => {
    const rows = [
      row("2026-05-26", { spend: 1.5, total_tokens: 10, api_requests: 2 }),
      row("2026-05-27", { spend: 2.5, total_tokens: 30, api_requests: 1 }),
    ];
    const collapsed = collapseDailyResults(rows, "12 AM");
    expect(collapsed.date).toBe("12 AM");
    expect(collapsed.metrics.spend).toBe(4);
    expect(collapsed.metrics.total_tokens).toBe(40);
    expect(collapsed.metrics.api_requests).toBe(3);
  });

  it("returns zero metrics when given an empty array", () => {
    const collapsed = collapseDailyResults([], "12 AM");
    expect(collapsed.date).toBe("12 AM");
    expect(collapsed.metrics.spend).toBe(0);
    expect(collapsed.metrics.total_tokens).toBe(0);
  });

  it("tolerates rows with missing optional metric fields", () => {
    const rows: DailyData[] = [
      {
        date: "2026-05-26",
        metrics: {
          ...baseMetrics,
          spend: 1,
          successful_requests: undefined as unknown as number,
          failed_requests: undefined as unknown as number,
        },
        breakdown: {
          models: {},
          model_groups: {},
          mcp_servers: {},
          providers: {},
          api_keys: {},
          entities: {},
        },
      },
    ];
    expect(() => collapseDailyResults(rows, SINGLE_DAY_TIME_LABEL)).not.toThrow();
  });
});

describe("getDailySpendChartData", () => {
  const singleDay = {
    from: new Date(2026, 4, 26, 0, 0, 0),
    to: new Date(2026, 4, 26, 23, 59, 59),
  };
  const multiDay = {
    from: new Date(2026, 4, 24, 0, 0, 0),
    to: new Date(2026, 4, 26, 23, 59, 59),
  };

  it("collapses to a single time-of-day-labeled bar for a single-day range", () => {
    const rows = [
      row("2026-05-26", { spend: 1.5 }),
      row("2026-05-27", { spend: 2.5 }),
    ];
    const out = getDailySpendChartData(rows, singleDay);
    expect(out).toHaveLength(1);
    expect(out[0].date).toBe(SINGLE_DAY_TIME_LABEL);
    expect(out[0].metrics.spend).toBe(4);
  });

  it("keeps and sorts rows ascending for a multi-day range", () => {
    const rows = [
      row("2026-05-25", { spend: 2 }),
      row("2026-05-24", { spend: 1 }),
      row("2026-05-26", { spend: 3 }),
    ];
    const out = getDailySpendChartData(rows, multiDay);
    expect(out.map((r) => r.date)).toEqual(["2026-05-24", "2026-05-25", "2026-05-26"]);
    expect(out.map((r) => r.metrics.spend)).toEqual([1, 2, 3]);
  });

  it("returns an empty array when there are no rows", () => {
    expect(getDailySpendChartData([], singleDay)).toEqual([]);
    expect(getDailySpendChartData([], multiDay)).toEqual([]);
  });

  it("does not mutate the input array when sorting", () => {
    const rows = [row("2026-05-25"), row("2026-05-24")];
    const before = rows.map((r) => r.date);
    getDailySpendChartData(rows, multiDay);
    expect(rows.map((r) => r.date)).toEqual(before);
  });

  it("preserves multi-day behavior when the date range is undefined", () => {
    const rows = [row("2026-05-25"), row("2026-05-24")];
    const out = getDailySpendChartData(rows, undefined);
    expect(out.map((r) => r.date)).toEqual(["2026-05-24", "2026-05-25"]);
  });
});
