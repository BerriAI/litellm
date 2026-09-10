import { describe, expect, it } from "vitest";
import { lastBudgetResetAt } from "./budgetWindow";

describe("lastBudgetResetAt", () => {
  it.each([
    [null, "30d"],
    [undefined, "30d"],
    ["2026-10-01T00:00:00Z", null],
    ["2026-10-01T00:00:00Z", undefined],
  ])("returns null for missing inputs", (budgetResetAt, budgetDuration) => {
    expect(lastBudgetResetAt(budgetResetAt, budgetDuration)).toBeNull();
  });

  it("returns null for an invalid reset date", () => {
    expect(lastBudgetResetAt("not-a-date", "30d")).toBeNull();
  });

  it("returns null for an unknown duration", () => {
    expect(lastBudgetResetAt("2026-10-01T00:00:00Z", "abc")).toBeNull();
  });

  it.each([
    ["30d", "2026-09-01T00:00:00.000Z"],
    ["1mo", "2026-09-01T00:00:00.000Z"],
    ["monthly", "2026-09-01T00:00:00.000Z"],
    ["7d", "2026-09-24T00:00:00.000Z"],
    ["weekly", "2026-09-24T00:00:00.000Z"],
    ["1d", "2026-09-30T00:00:00.000Z"],
    ["daily", "2026-09-30T00:00:00.000Z"],
    ["12h", "2026-09-30T12:00:00.000Z"],
    ["45m", "2026-09-30T23:15:00.000Z"],
  ])("subtracts the current budget period for %s", (budgetDuration, expected) => {
    expect(lastBudgetResetAt("2026-10-01T00:00:00Z", budgetDuration)?.toISOString()).toBe(expected);
  });
});
