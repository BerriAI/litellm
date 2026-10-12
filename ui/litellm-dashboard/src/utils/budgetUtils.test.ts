import { afterEach, describe, expect, it } from "vitest";

import { formatBudgetReset, formatBudgetResetUtc } from "./budgetUtils";

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
