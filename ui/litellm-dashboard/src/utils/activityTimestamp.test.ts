import { describe, expect, it } from "vitest";
import { formatActivityTimestamp } from "./activityTimestamp";

describe("Activity timestamps", () => {
  it("shows the date, seconds and timezone for the same instant across storage formats", () => {
    const value = "2026-10-01T20:42:17.123Z";
    const formatted = formatActivityTimestamp(value);
    const options: Intl.DateTimeFormatOptions = {
      year: "numeric",
      month: "short",
      day: "numeric",
      hour: "numeric",
      minute: "2-digit",
      second: "2-digit",
      timeZoneName: "short",
    };
    const parts = new Intl.DateTimeFormat(undefined, options).formatToParts(new Date(value));
    for (const part of parts.filter((part) => ["year", "month", "day", "second", "timeZoneName"].includes(part.type)))
      expect(formatted).toContain(part.value);
    expect(formatActivityTimestamp("2026-10-01 20:42:17.123")).toBe(formatted);
    expect(formatActivityTimestamp("2026-10-01T13:42:17.123-07:00")).toBe(formatted);
  });
  it("preserves an invalid timestamp rather than inventing a date", () => {
    expect(formatActivityTimestamp("unavailable")).toBe("unavailable");
  });
});
