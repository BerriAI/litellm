import { describe, expect, it } from "vitest";
import { appBudgetReached, appBudgetWindows, appSpend } from "./appBudgetPresentation";

describe("app budget presentation", () => {
  it.each([
    [0, "$0.00"],
    [0.0005481, "<$0.01"],
    [0.000000001, "<$0.01"],
    [0.01, "$0.01"],
    [12.345, "$12.35"],
    [100, "$100.00"],
  ])("should format %s without hiding positive usage", (amount, expected) => {
    expect(appSpend(Number(amount))).toBe(expected);
  });
  it("should prioritize an exhausted yearly limit over an available monthly limit", () => {
    const windows = [
      { source: "window" as const, budget_duration: "1mo", max_budget: 100, spend: 7, reset_at: null },
      { source: "window" as const, budget_duration: "12mo", max_budget: 1000, spend: 1000, reset_at: null },
    ];
    const result = appBudgetWindows(null, windows);
    expect(result.primary).toEqual(windows[1]);
    expect(result.windows).toHaveLength(2);
  });
  it("should distinguish exhausted zero budgets from a tiny positive balance", () => {
    expect(appBudgetReached({ max_budget: 0, spend: 0 })).toBe(true);
    expect(appBudgetReached({ max_budget: 1, spend: 0.9999 })).toBe(false);
  });
});
