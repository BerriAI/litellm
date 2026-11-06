import { afterEach, describe, expect, it } from "vitest";

import { budgetPeriodRange, describeBudgetResetSchedule, formatBudgetReset, formatBudgetResetUtc } from "./budgetUtils";

const localIso = (year: number, monthIndex: number, day: number, hour = 12): string =>
  new Date(year, monthIndex, day, hour, 0, 0).toISOString();

const originalTimezone: string | undefined = process.env.TZ;

afterEach(() => {
  if (originalTimezone === undefined) {
    delete process.env.TZ;
    return;
  }
  process.env.TZ = originalTimezone;
});

describe("formatBudgetResetUtc", () => {
  it("formats the reset date in UTC regardless of the local timezone", () => {
    process.env.TZ = "America/Los_Angeles";
    const resetAt: string = "2026-10-20T00:00:00Z";

    expect(formatBudgetResetUtc(resetAt)).toBe("Oct 20, 2026");
    expect(formatBudgetReset(resetAt)).toBe("Oct 19, 2026");
  });
});

describe("describeBudgetResetSchedule", () => {
  it("reports lifetime spend with no period when the budget has no duration and no scheduled reset", () => {
    const schedule = describeBudgetResetSchedule(null, null);
    expect(schedule.label).toBe("Never resets");
    expect(schedule.resetAt).toBeNull();
    expect(schedule.period).toBeNull();
    expect(schedule.tooltip).toContain("lifetime spend");
  });

  it("reports Unknown when a duration is set but no reset has been scheduled yet", () => {
    const schedule = describeBudgetResetSchedule("30d", null);
    expect(schedule.label).toBe("Unknown");
    expect(schedule.period).toBeNull();
    expect(schedule.tooltip).toContain("every 30d");
  });

  it("shows the next reset with its time zone and the rollover carryover note", () => {
    const resetAt = new Date(2026, 6, 15, 12, 0, 0);
    const schedule = describeBudgetResetSchedule("30d", resetAt.toISOString());
    expect(schedule.label).toBe("Jul 15, 2026 12:00");
    expect(schedule.resetAt).toBe(resetAt.toISOString());
    expect(schedule.tooltip).toContain(Intl.DateTimeFormat().resolvedOptions().timeZone);
    expect(schedule.tooltip).toContain("carries into the next period");
  });

  it("treats 30d as one calendar month back from the reset, like the proxy does", () => {
    const schedule = describeBudgetResetSchedule("30d", localIso(2026, 10, 1, 0));
    expect(schedule.period).toBe("Oct 1 - Nov 1, 2026");
    expect(schedule.tooltip).toContain("covers Oct 1 - Nov 1, 2026 and resets every 30d");
  });

  it("subtracts the exact duration for day and week windows", () => {
    expect(describeBudgetResetSchedule("7d", localIso(2026, 6, 15)).period).toBe("Jul 8 - Jul 15, 2026");
    expect(describeBudgetResetSchedule("2w", localIso(2026, 6, 15)).period).toBe("Jul 1 - Jul 15, 2026");
    expect(describeBudgetResetSchedule("weekly", localIso(2026, 6, 15)).period).toBe("Jul 8 - Jul 15, 2026");
  });

  it("walks the month back in UTC so a non-UTC viewer sees the proxy's window", () => {
    const originalTz = process.env.TZ;
    process.env.TZ = "America/New_York";
    try {
      const schedule = describeBudgetResetSchedule("30d", "2026-03-01T00:00:00Z");
      expect(schedule.period).toBe("Jan 31 - Feb 28, 2026");
    } finally {
      process.env.TZ = originalTz;
    }
  });

  it("clamps month windows to the last day of a shorter month", () => {
    expect(describeBudgetResetSchedule("1mo", localIso(2026, 2, 31)).period).toBe("Feb 28 - Mar 31, 2026");
  });

  it("spells out both years when the period crosses a year boundary", () => {
    expect(describeBudgetResetSchedule("30d", localIso(2027, 0, 10)).period).toBe("Dec 10, 2026 - Jan 10, 2027");
  });

  it("keeps both clock times when an hourly window crosses midnight", () => {
    expect(describeBudgetResetSchedule("1h", localIso(2026, 6, 16, 0)).period).toBe(
      "Jul 15, 2026 23:00 - Jul 16, 2026 00:00",
    );
  });

  it("shows seconds for a window shorter than a minute", () => {
    expect(describeBudgetResetSchedule("30s", localIso(2026, 6, 15)).period).toBe("Jul 15, 2026 11:59:30 - 12:00:00");
  });

  it("shows clock times for a window that starts and ends on the same day", () => {
    expect(describeBudgetResetSchedule("1h", localIso(2026, 6, 15)).period).toBe("Jul 15, 2026 11:00 - 12:00");
  });

  it("falls back to a one day window for a duration the proxy would not recognise", () => {
    expect(describeBudgetResetSchedule("fortnightly", localIso(2026, 6, 15)).period).toBe("Jul 14 - Jul 15, 2026");
  });

  it("shows no period when the reset date is unparseable", () => {
    const schedule = describeBudgetResetSchedule("30d", "not-a-date");
    expect(schedule.label).toBe("Unknown");
    expect(schedule.period).toBeNull();
    expect(schedule.tooltip).not.toMatch(/NaN|Invalid/);
  });

  it("never describes the window as a cycle", () => {
    const schedules = [
      describeBudgetResetSchedule("30d", localIso(2026, 6, 15)),
      describeBudgetResetSchedule("30d", null),
      describeBudgetResetSchedule(null, null),
    ];
    for (const schedule of schedules) {
      expect(`${schedule.label} ${schedule.tooltip}`).not.toMatch(/cycle/i);
    }
  });
});

describe("budgetPeriodRange", () => {
  const resetAt = "2026-04-01T00:00:00Z";

  it.each([
    ["30d", "2026-03-01T00:00:00.000Z"],
    ["monthly", "2026-03-01T00:00:00.000Z"],
    ["1mo", "2026-03-01T00:00:00.000Z"],
    ["2mo", "2026-02-01T00:00:00.000Z"],
    ["7d", "2026-03-25T00:00:00.000Z"],
    ["weekly", "2026-03-25T00:00:00.000Z"],
    ["1w", "2026-03-25T00:00:00.000Z"],
    ["15d", "2026-03-17T00:00:00.000Z"],
    ["24h", "2026-03-31T00:00:00.000Z"],
    ["daily", "2026-03-31T00:00:00.000Z"],
    ["1h", "2026-03-31T23:00:00.000Z"],
    ["hourly", "2026-03-31T23:00:00.000Z"],
  ])("computes the period start for %s the way the proxy does", (duration, start) => {
    expect(budgetPeriodRange(duration, resetAt)).toEqual({ start, end: "2026-04-01T00:00:00.000Z" });
  });

  it("spans a calendar month for 30d even when the viewer is in a DST-shifting timezone", () => {
    process.env.TZ = "America/New_York";
    expect(budgetPeriodRange("30d", "2026-04-01T00:00:00Z")?.start).toBe("2026-03-01T00:00:00.000Z");
    expect(budgetPeriodRange("30d", "2026-03-01T00:00:00Z")?.start).toBe("2026-02-01T00:00:00.000Z");
  });

  it.each([
    ["bogus", resetAt],
    ["0d", resetAt],
    ["30d", "not-a-date"],
    ["30d", null],
    [null, resetAt],
    ["", resetAt],
  ])("returns null for duration %s and reset %s instead of an invalid date", (duration, reset) => {
    expect(budgetPeriodRange(duration, reset)).toBeNull();
  });
});
