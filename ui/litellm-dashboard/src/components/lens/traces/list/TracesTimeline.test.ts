import { describe, expect, it } from "vitest";

import { bandForWindow, type Bucket, dragUpdate, formatSpan } from "./TracesTimeline";

const HOUR = 3600 * 1000;
const START = Date.UTC(2026, 8, 30, 0, 0, 0);
const range = { startMs: START, endMs: START + 10 * HOUR };

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
    const buckets: Bucket[] = Array.from({ length: 10 }, (_, i) => ({
      startMs: range.startMs + i * HOUR,
      endMs: range.startMs + (i + 1) * HOUR,
      total: 0,
      failed: 0,
      series: [],
    }));
    expect(bandForWindow(buckets, { startMs: START + 2 * HOUR, endMs: START + 5 * HOUR })).toEqual({ lo: 2, hi: 4 });
    expect(bandForWindow(buckets, null)).toBeNull();
  });
});
