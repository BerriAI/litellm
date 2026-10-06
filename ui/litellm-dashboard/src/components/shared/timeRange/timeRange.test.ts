import { describe, expect, it } from "vitest";

import { isRangeHours, presetLabel, timeWindow } from "./timeRange";

const HOUR = 3600 * 1000;

describe("timeWindow", () => {
  it("rolls a live range forward with now so live tail keeps a fixed-length window", () => {
    const mountedAt = Date.parse("2026-09-30T10:00:00");
    const tenHoursLater = mountedAt + 10 * HOUR;
    const live = { hours: 24, anchorMs: null };
    expect(timeWindow(live, tenHoursLater)).toEqual({ startMs: tenHoursLater - 24 * HOUR, endMs: tenHoursLater });
    expect(timeWindow(live, tenHoursLater).startMs).toBeGreaterThan(timeWindow(live, mountedAt).startMs);
  });

  it("keeps a paused range pinned to its anchor whatever now is", () => {
    const anchorMs = Date.parse("2026-09-02T00:00");
    expect(timeWindow({ hours: 24, anchorMs }, Date.parse("2026-09-30T00:00:00"))).toEqual({
      startMs: anchorMs - 24 * HOUR,
      endMs: anchorMs,
    });
  });
});

describe("presets", () => {
  it("accepts only the preset lengths from the URL and names them", () => {
    expect(isRangeHours(168)).toBe(true);
    expect(isRangeHours(2)).toBe(false);
    expect(presetLabel(168)).toBe("Last 7 days");
    expect(presetLabel(2)).toBe("Last 2 hours");
  });
});
