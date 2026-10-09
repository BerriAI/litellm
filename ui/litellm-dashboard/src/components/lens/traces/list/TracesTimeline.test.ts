import { describe, expect, it } from "vitest";

import { bucketRuns } from "./TracesTimeline";
import type { TraceSummary } from "../types";

const HOUR = 3600 * 1000;
const START = Date.UTC(2026, 8, 30, 0, 0, 0);
const range = { startMs: START, endMs: START + 10 * HOUR };

const run = (offsetMs: number, errorCount = 0, status: TraceSummary["status"] = "ok"): TraceSummary =>
  ({
    trace_id: `t${offsetMs}`,
    start_time: new Date(START + offsetMs).toISOString(),
    error_count: errorCount,
    status,
  }) as TraceSummary;

describe("bucketRuns", () => {
  it("splits the window into equal buckets that tile it exactly", () => {
    const buckets = bucketRuns([], range, 10);
    expect(buckets).toHaveLength(10);
    expect(buckets[0].startMs).toBe(range.startMs);
    expect(buckets.at(-1)?.endMs).toBe(range.endMs);
    const fourthBucket = { startMs: START + 3 * HOUR, endMs: START + 4 * HOUR, total: 0, failed: 0 };
    expect(buckets[3]).toMatchObject(fourthBucket);
  });

  it("puts each run in the bucket covering its start time", () => {
    const buckets = bucketRuns([run(0), run(30 * 60 * 1000), run(2.5 * HOUR), run(10 * HOUR - 1)], range, 10);
    expect(buckets.map((b) => b.total)).toEqual([2, 0, 1, 0, 0, 0, 0, 0, 0, 1]);
  });

  it("drops runs that start before or at/after the window", () => {
    const buckets = bucketRuns([run(-1), run(10 * HOUR), run(20 * HOUR), run(HOUR)], range, 10);
    expect(buckets.reduce((sum, b) => sum + b.total, 0)).toBe(1);
    expect(buckets[1].total).toBe(1);
  });

  it("counts failed runs without treating recovered tool errors as run failures", () => {
    const buckets = bucketRuns([run(HOUR, 8), run(HOUR + 1), run(HOUR + 2, 1, "error"), run(5 * HOUR)], range, 10);
    expect(buckets[1]).toMatchObject({ total: 3, failed: 1 });
    expect(buckets[5]).toMatchObject({ total: 1, failed: 0 });
  });
});
