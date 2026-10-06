import { describe, expect, it } from "vitest";

import { barGeometry, tickLabel, timeTicks } from "./timeline";

describe("barGeometry", () => {
  it("places a span by its share of the run", () => {
    expect(barGeometry(2000, 5000, 10_000)).toEqual({ left: 20, width: 50 });
    expect(barGeometry(0, 10_000, 10_000)).toEqual({ left: 0, width: 100 });
  });

  it("keeps instant spans visible and inside the track", () => {
    expect(barGeometry(5000, 0, 10_000)).toEqual({ left: 50, width: 0.6 });
    expect(barGeometry(10_000, 0, 10_000)).toEqual({ left: 99.4, width: 0.6 });
  });

  it("clips a span that overruns the run's end and clamps clock skew before the start", () => {
    expect(barGeometry(8000, 5000, 10_000)).toEqual({ left: 80, width: 20 });
    expect(barGeometry(-500, 1000, 10_000)).toEqual({ left: 0, width: 10 });
  });

  it("draws a sliver for a run with no duration", () => {
    expect(barGeometry(0, 0, 0)).toEqual({ left: 0, width: 0.6 });
  });
});

describe("tickLabel", () => {
  it.each([
    [0, "0"],
    [20, "20ms"],
    [500, "500ms"],
    [2000, "2s"],
    [1500, "1.5s"],
    [60_000, "1m"],
    [90_000, "1.5m"],
  ])("labels %d ms as %s", (ms, label) => {
    expect(tickLabel(ms)).toBe(label);
  });
});

describe("timeTicks", () => {
  it("picks the smallest round step that fits", () => {
    expect(timeTicks(45_400)).toEqual([0, 10_000, 20_000, 30_000, 40_000]);
    expect(timeTicks(6870)).toEqual([0, 2000, 4000, 6000]);
    expect(timeTicks(52)).toEqual([0, 10, 20, 30, 40, 50]);
  });

  it("respects the tick budget", () => {
    expect(timeTicks(45_400, 3)).toEqual([0, 30_000]);
    expect(timeTicks(10_000, 10)).toEqual([0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10_000]);
  });

  it("falls back to an even split past the largest step and to a single tick for an empty run", () => {
    expect(timeTicks(3_600_000)).toEqual([0, 600_000, 1_200_000, 1_800_000, 2_400_000, 3_000_000, 3_600_000]);
    expect(timeTicks(0)).toEqual([0]);
  });
});
