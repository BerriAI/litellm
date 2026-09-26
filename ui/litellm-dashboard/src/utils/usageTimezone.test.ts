import { describe, expect, it } from "vitest";
import { usageCalendarDate } from "./usageTimezone";

describe("usageCalendarDate", () => {
  it.each([
    ["2026-09-25T17:00:00Z", "Asia/Singapore", [2026, 9, 26]],
    ["2026-09-30T16:00:00Z", "Asia/Singapore", [2026, 10, 1]],
    ["2026-09-25T18:15:00Z", "Asia/Kathmandu", [2026, 9, 26]],
    ["2026-03-08T07:59:59Z", "America/Los_Angeles", [2026, 3, 7]],
    ["2026-03-08T08:00:00Z", "America/Los_Angeles", [2026, 3, 8]],
  ])("uses the reporting calendar for %s in %s", (instant, zone, expected) => {
    const result = usageCalendarDate(new Date(instant), zone);
    expect([result.getFullYear(), result.getMonth() + 1, result.getDate()]).toEqual(expected);
  });

  it("preserves the browser instant when reporting timezone is unset", () => {
    const instant = new Date("2026-09-25T17:00:00Z");
    expect(usageCalendarDate(instant).getTime()).toBe(instant.getTime());
  });
});
