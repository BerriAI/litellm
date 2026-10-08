import { describe, expect, it } from "vitest";

import { activeWindow, bucketRunStats, formatSpend, percentile, runStats } from "./runStats";
import type { TraceSummary } from "../types";

const HOUR = 3600 * 1000;
const START = Date.UTC(2026, 8, 30, 0, 0, 0);
const range = { startMs: START, endMs: START + 4 * HOUR };

const run = (offsetMs: number, overrides: Partial<TraceSummary> = {}): TraceSummary =>
  ({
    trace_id: `t${offsetMs}`,
    start_time: new Date(START + offsetMs).toISOString(),
    status: "ok",
    duration_ms: 100,
    spend: 0.5,
    priced_calls: 1,
    llm_calls: 1,
    input_tokens: 10,
    output_tokens: 5,
    agent_names: ["a"],
    service: "",
    ...overrides,
  }) as TraceSummary;

describe("percentile", () => {
  it("returns null for no values and the nearest-rank value otherwise", () => {
    expect(percentile([], 50)).toBeNull();
    expect(percentile([40, 10, 30, 20], 50)).toBe(20);
    expect(percentile([40, 10, 30, 20], 95)).toBe(40);
    expect(percentile([7], 95)).toBe(7);
  });
});

describe("runStats", () => {
  it("totals runs, failures, spend, tokens and distinct agents", () => {
    const failedRun: Partial<TraceSummary> = { status: "error", spend: 1.25, input_tokens: 100, output_tokens: 50 };
    const stats = runStats(
      [run(0, { agent_names: ["a", "b"] }), run(HOUR, failedRun), run(2 * HOUR, { agent_names: ["b"] })],
      range,
    );
    const expected = { runs: 3, failed: 1, agents: 2, spend: 2.25, inputTokens: 120, outputTokens: 60 };
    expect(stats).toMatchObject(expected);
  });

  it("counts runs whose cost is missing or only partly priced as unpriced, without adding to spend", () => {
    const stats = runStats(
      [run(0, { spend: null }), run(1, { priced_calls: 1, llm_calls: 3, spend: 0.1 }), run(2)],
      range,
    );
    expect(stats.unpriced).toBe(2);
    expect(stats.spend).toBeCloseTo(0.6);
  });

  it("reports p50 and p95 duration and leaves them empty with no runs", () => {
    const durations = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000];
    const stats = runStats(
      durations.map((ms, i) => run(i, { duration_ms: ms })),
      range,
    );
    expect(stats.p50).toBe(500);
    expect(stats.p95).toBe(1000);
    const empty = { runs: 0, p50: null, p95: null, spend: 0 };
    expect(runStats([], range)).toMatchObject(empty);
  });
});

describe("bucketRunStats", () => {
  it("places each run in the bucket covering its start and sums that bucket", () => {
    const buckets = bucketRunStats(
      [run(0, { spend: 1 }), run(HOUR / 2, { spend: 2, duration_ms: 300 }), run(3 * HOUR, { spend: 4 })],
      range,
      4,
    );
    expect(buckets.map((b) => b.runs)).toEqual([2, 0, 0, 1]);
    expect(buckets.map((b) => b.spend)).toEqual([3, 0, 0, 4]);
    expect(buckets[0].p50).toBe(100);
    expect(buckets[0].tokens).toBe(30);
  });

  it("drops runs outside the window", () => {
    const buckets = bucketRunStats([run(-1), run(4 * HOUR), run(HOUR)], range, 4);
    expect(buckets.reduce((total, b) => total + b.runs, 0)).toBe(1);
  });
});

describe("activeWindow", () => {
  it("spans the earliest to the latest run inside the window", () => {
    expect(activeWindow([run(3 * HOUR), run(2 * HOUR), run(-HOUR)], range)).toEqual({
      startMs: START + 2 * HOUR,
      endMs: START + 3 * HOUR + 1,
    });
  });

  it("falls back to the full window when no run is inside it", () => {
    expect(activeWindow([run(-HOUR)], range)).toEqual(range);
  });

  it("spreads a late burst across several sparkline buckets instead of the last one", () => {
    const late = [0, 1, 2, 3].map((i) => run(3 * HOUR + i * 10 * 60 * 1000));
    expect(runStats(late, range).buckets.filter((b) => b.runs > 0).length).toBeGreaterThan(1);
  });
});

describe("formatSpend", () => {
  it("keeps sub-cent spend visible and compacts large spend", () => {
    expect(formatSpend(0)).toBe("$0.00");
    expect(formatSpend(0.0042)).toBe("$0.0042");
    expect(formatSpend(12.345)).toBe("$12.35");
    expect(formatSpend(12_400)).toBe("$12.4K");
  });
});
