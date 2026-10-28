import { describe, expect, it } from "vitest";

import { bandForWindow, dragUpdate, formatSpan, timelineTicks, type TimeBucket } from "./Timeline";

const HOUR = 3600 * 1000;
const START = Date.UTC(2026, 8, 30, 0, 0, 0);

const emptyBuckets = (count: number): TimeBucket[] =>
  Array.from({ length: count }, (_, i) => ({
    startMs: START + i * HOUR,
    endMs: START + (i + 1) * HOUR,
    total: 0,
    failed: 0,
    series: [],
  }));

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
    const buckets = emptyBuckets(10);
    expect(bandForWindow(buckets, { startMs: START + 2 * HOUR, endMs: START + 5 * HOUR })).toEqual({ lo: 2, hi: 4 });
    expect(bandForWindow(buckets, null)).toBeNull();
  });
});

describe("timelineTicks", () => {
  it("keeps both edges visible and reduces labels in narrow timelines", () => {
    const range = { startMs: 0, endMs: 24 * HOUR };
    const narrow = timelineTicks(range, 280);
    const wide = timelineTicks(range, 1000);
    expect(narrow[0]).toBe(0);
    expect(narrow.at(-1)).toBe(1);
    expect(narrow.length).toBeLessThan(wide.length);
    expect(280 / (narrow.length - 1)).toBeGreaterThanOrEqual(80);
    expect(timelineTicks(range, 0)).toEqual([0, 1]);
    expect(timelineTicks({ ...range, endMs: 30 * 24 * HOUR }, 280)).toEqual([0, 1]);
  });
});
