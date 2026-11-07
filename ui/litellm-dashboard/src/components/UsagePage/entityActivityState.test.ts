import { afterEach, describe, expect, it, vi } from "vitest";
import { activityPresets, localActivityDay, parseActivityDay } from "./entityActivityState";

afterEach(() => vi.useRealTimers());

describe("activity date links", () => {
  it.each(["2026-11-05", "2024-02-29"])("accepts a calendar day: %s", (value) => {
    expect(parseActivityDay.parse(value)).toBe(value);
  });
  it.each(["", "today", "2026-2-1", "2026-02-29", "2026-11-05T12:00:00Z", "2026-13-01"])("rejects %s", (value) => {
    expect(parseActivityDay.parse(value)).toBeNull();
  });
});

describe("UTC activity presets", () => {
  it.each([
    ["today", "2026-01-01"],
    ["last_7_days", "2025-12-26"],
    ["last_30_days", "2025-12-03"],
    ["month_to_date", "2026-01-01"],
    ["year_to_date", "2026-01-01"],
  ])("uses inclusive UTC dates for %s", (preset, from) => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date("2026-01-01T23:30:00Z"));
    const range = activityPresets.find((option) => option.settingValue === preset)!.getValue();
    expect(localActivityDay(range.from)).toBe(from);
    expect(localActivityDay(range.to)).toBe("2026-01-01");
  });
});
