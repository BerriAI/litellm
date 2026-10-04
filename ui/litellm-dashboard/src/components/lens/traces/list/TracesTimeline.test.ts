import { describe, expect, it } from "vitest";

import { bandForWindow, bucketRuns, dragUpdate, formatSpan } from "./TracesTimeline";
import type { TraceSummary } from "../types";

const HOUR = 3600 * 1000;
const START = Date.UTC(2026, 8, 30, 0, 0, 0);
const range = { startMs: START, endMs: START + 10 * HOUR };

const run = (offsetMs: number, errorCount = 0): TraceSummary =>
  ({
    trace_id: `t${offsetMs}`,
    start_time: new Date(START + offsetMs).toISOString(),
    error_count: errorCount,
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

  it("counts runs with any errors as failed", () => {
    const buckets = bucketRuns([run(HOUR, 3), run(HOUR + 1), run(HOUR + 2, 1), run(5 * HOUR)], range, 10);
    expect(buckets[1]).toMatchObject({ total: 3, failed: 2 });
    expect(buckets[5]).toMatchObject({ total: 1, failed: 0 });
  });
});

describe("formatSpan", () => {
  it("prints the largest two units, dropping zero parts", () => {
    expect(formatSpan(45 * 60 * 1000)).toBe("45m");
    expect(formatSpan(6 * HOUR + 12 * 60 * 1000)).toBe("6h 12m");
    expect(formatSpan(24 * HOUR)).toBe("1d");
    expect(formatSpan(152 * 24 * HOUR + 23 * HOUR)).toBe("152d 23h");
    expect(formatSpan(-5)).toBe("0m");
  });
});

describe("dragUpdate", () => {
  const band = { lo: 10, hi: 14 };

  it("selects between the press point and the pointer, in either direction", () => {
    expect(dragUpdate({ mode: "select", origin: 20, band: { lo: 20, hi: 20 } }, 25)).toEqual({ lo: 20, hi: 25 });
    expect(dragUpdate({ mode: "select", origin: 20, band: { lo: 20, hi: 20 } }, 12)).toEqual({ lo: 12, hi: 20 });
  });

  it("resizes one edge without letting it cross the other", () => {
    expect(dragUpdate({ mode: "resize-lo", origin: 10, band }, 4)).toEqual({ lo: 4, hi: 14 });
    expect(dragUpdate({ mode: "resize-lo", origin: 10, band }, 30)).toEqual({ lo: 14, hi: 14 });
    expect(dragUpdate({ mode: "resize-hi", origin: 14, band }, 40)).toEqual({ lo: 10, hi: 40 });
    expect(dragUpdate({ mode: "resize-hi", origin: 14, band }, 2)).toEqual({ lo: 10, hi: 10 });
  });

  it("pans the band keeping its width, clamped to the strip", () => {
    expect(dragUpdate({ mode: "move", origin: 12, band }, 20)).toEqual({ lo: 18, hi: 22 });
    expect(dragUpdate({ mode: "move", origin: 12, band }, -50)).toEqual({ lo: 0, hi: 4 });
    expect(dragUpdate({ mode: "move", origin: 12, band }, 500, 60)).toEqual({ lo: 55, hi: 59 });
  });
});

describe("bandForWindow", () => {
  it("maps a selected window back to the buckets it covers", () => {
    const buckets = bucketRuns([], range, 10);
    expect(bandForWindow(buckets, { startMs: START + 2 * HOUR, endMs: START + 5 * HOUR })).toEqual({ lo: 2, hi: 4 });
    expect(bandForWindow(buckets, null)).toBeNull();
  });
});
