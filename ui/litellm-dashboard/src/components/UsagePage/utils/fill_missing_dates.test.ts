import { afterEach, describe, expect, it } from "vitest";
import { DailyData } from "../types";
import { fillMissingDates } from "./fill_missing_dates";

const originalTz = process.env.TZ;

const row = (date: string, spend: number): DailyData => ({
  date,
  metrics: {
    spend,
    prompt_tokens: 1,
    completion_tokens: 1,
    total_tokens: 2,
    api_requests: 1,
    successful_requests: 1,
    failed_requests: 0,
    cache_read_input_tokens: 0,
    cache_creation_input_tokens: 0,
  },
  breakdown: { models: {}, model_groups: {}, mcp_servers: {}, providers: {}, api_keys: {}, entities: {} },
});

describe("fillMissingDates", () => {
  afterEach(() => {
    process.env.TZ = originalTz;
  });

  it("adds a zero row for every day in the range with no data and keeps real rows", () => {
    process.env.TZ = "UTC";
    const monday = row("2026-09-21", 3);
    const wednesday = row("2026-09-23", 5);

    const filled = fillMissingDates([monday, wednesday], new Date(2026, 8, 20, 9), new Date(2026, 8, 24, 18));

    expect(filled.map((day) => day.date)).toEqual([
      "2026-09-20",
      "2026-09-21",
      "2026-09-22",
      "2026-09-23",
      "2026-09-24",
    ]);
    expect(filled[1]).toBe(monday);
    expect(filled[3]).toBe(wednesday);
    expect([filled[0], filled[2], filled[4]].map((day) => day.metrics.spend)).toEqual([0, 0, 0]);
    expect(filled[2].metrics.api_requests).toBe(0);
    expect(filled[2].breakdown.models).toEqual({});
  });

  it("keys days by the local calendar date, matching the backend's local-day buckets", () => {
    process.env.TZ = "Asia/Kolkata";
    const withSpend = row("2026-09-23", 5);

    const filled = fillMissingDates([withSpend], new Date(2026, 8, 17, 2), new Date(2026, 8, 23, 2));

    expect(filled.map((day) => day.date)).toEqual([
      "2026-09-17",
      "2026-09-18",
      "2026-09-19",
      "2026-09-20",
      "2026-09-21",
      "2026-09-22",
      "2026-09-23",
    ]);
    expect(filled[6]).toBe(withSpend);
  });

  it("counts each day once across a daylight saving transition", () => {
    process.env.TZ = "America/New_York";

    const filled = fillMissingDates([], new Date(2026, 2, 7), new Date(2026, 2, 9));

    expect(filled.map((day) => day.date)).toEqual(["2026-03-07", "2026-03-08", "2026-03-09"]);
  });

  it("returns the data unchanged when the range is incomplete", () => {
    const data = [row("2026-09-21", 3)];

    expect(fillMissingDates(data, undefined, new Date(2026, 8, 24))).toBe(data);
    expect(fillMissingDates(data, new Date(2026, 8, 20), undefined)).toBe(data);
  });
});
