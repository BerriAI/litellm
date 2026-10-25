import { afterEach, describe, expect, it } from "vitest";

import { describeBudgetResetSchedule, formatBudgetReset, formatBudgetResetUtc } from "./budgetUtils";

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
  it("reports Never resets when the budget has no duration and no scheduled reset", () => {
    const schedule = describeBudgetResetSchedule(null, null);
    expect(schedule.label).toBe("Never resets");
    expect(schedule.resetAt).toBeNull();
  });

  it("reports Unknown when a duration is set but no reset has been scheduled yet", () => {
    const schedule = describeBudgetResetSchedule("30d", null);
    expect(schedule.label).toBe("Unknown");
    expect(schedule.tooltip).toContain("every 30d");
  });

  it("shows the next reset with its time zone and the rollover carryover note", () => {
    const resetAt = new Date(2026, 6, 15, 12, 0, 0);
    const schedule = describeBudgetResetSchedule("30d", resetAt.toISOString());
    expect(schedule.label).toBe("Jul 15, 2026 12:00");
    expect(schedule.resetAt).toBe(resetAt.toISOString());
    expect(schedule.tooltip).toContain(Intl.DateTimeFormat().resolvedOptions().timeZone);
    expect(schedule.tooltip).toContain("carries into the next cycle");
  });
});
