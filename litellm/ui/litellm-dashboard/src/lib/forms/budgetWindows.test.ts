import { describe, expect, it } from "vitest";
import { CALENDAR_BUDGET_DURATIONS, mergeBudgetLimits } from "./budgetWindows";

describe("mergeBudgetLimits", () => {
  it("keeps hidden custom windows and their identities when calendar caps are removed", () => {
    const custom = { budget_duration: "7d", max_budget: 3, reset_at: "2027-01-04T00:00:00Z" };
    const alias = { budget_duration: "30d", max_budget: 10 };
    expect(
      mergeBudgetLimits([{ budget_duration: "1mo", max_budget: 50 }, custom, alias], [], CALENDAR_BUDGET_DURATIONS),
    ).toEqual([custom, alias]);
    expect(mergeBudgetLimits([custom], [alias], CALENDAR_BUDGET_DURATIONS)[1]).toBe(custom);
  });

  it("replaces explicitly edited custom windows without duplicating them", () => {
    const saved = [{ budget_duration: "7d", max_budget: 3 }];
    expect(mergeBudgetLimits(saved, [{ ...saved[0], max_budget: 0 }], ["7d"])).toEqual([
      { budget_duration: "7d", max_budget: 0 },
    ]);
    expect(mergeBudgetLimits(saved, [], ["7d"])).toEqual([]);
    expect(mergeBudgetLimits(null, [], [])).toEqual([]);
    expect(mergeBudgetLimits(undefined, saved, [])).toEqual(saved);
    expect(mergeBudgetLimits(saved, [{ budget_duration: "2d", max_budget: 3 }], ["7d"])).toEqual([
      { budget_duration: "2d", max_budget: 3 },
    ]);
  });
});
