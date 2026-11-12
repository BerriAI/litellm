import moment from "moment";
import { afterEach, describe, expect, it, vi } from "vitest";
import { relativeTimeOptions } from "./date_range_presets";

describe("last 30 days", () => {
  afterEach(() => vi.useRealTimers());

  it("should include today and the preceding 29 calendar days across a clock change", () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date("2026-11-12T12:00:00"));
    const range = relativeTimeOptions.find((option) => option.shortLabel === "30d")!.getValue();

    expect(moment(range.to).startOf("day").diff(moment(range.from), "days") + 1).toBe(30);
    expect(moment(range.to).isSame(moment(), "day")).toBe(true);
    expect(moment(range.from).isSame(moment(range.from).startOf("day"))).toBe(true);
    expect(moment(range.to).isSame(moment(range.to).endOf("day"))).toBe(true);
  });
});
