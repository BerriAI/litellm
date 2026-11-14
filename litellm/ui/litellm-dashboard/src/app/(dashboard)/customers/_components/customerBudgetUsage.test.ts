import { describe, expect, it } from "vitest";
import { customerBudgetPeriod, customerBudgetStatus, customerBudgetUsage } from "./customerBudgetUsage";
import type { CustomerUsageEntity } from "./customerUsageApi";

const entity: CustomerUsageEntity = {
  id: "member",
  label: "Member",
  spend: 40,
  budgetLimit: 50,
  budgetDuration: "1mo",
  budgetResetAt: null,
  budgetLimits: [
    { budget_duration: "1mo", max_budget: 10 },
    { budget_duration: "12mo", max_budget: 100 },
  ],
};

describe("customer budget usage", () => {
  it("keeps independent counters and their limits paired, even with the same duration", () => {
    const budgetUsage = [
      { source: "scalar" as const, budget_duration: "1mo", max_budget: 50, spend: 40 },
      { source: "window" as const, budget_duration: "1mo", max_budget: 10, spend: 8 },
      { source: "window" as const, budget_duration: "12mo", max_budget: 100, spend: 62 },
    ];
    expect(customerBudgetUsage({ ...entity, budgetUsage })).toEqual(budgetUsage);
    expect(customerBudgetStatus(budgetUsage)).toBe("customerPortal.analytics.withinBudget");
  });

  it.each([0, 10, 100])("marks a recorded exhausted limit including zero (%s)", (limit) => {
    expect(customerBudgetStatus([{ source: "window", budget_duration: "12mo", max_budget: limit, spend: limit }])).toBe(
      "customerPortal.analytics.reachedLimit",
    );
  });

  it("distinguishes unavailable window usage from no budget and zero spend", () => {
    const usage = customerBudgetUsage(entity);
    expect(usage.map((budget) => [budget.max_budget, budget.spend])).toEqual([
      [50, 40],
      [10, null],
      [100, null],
    ]);
    expect(customerBudgetStatus(usage)).toBe("customerPortal.analytics.budgetUsageUnavailable");
    expect(customerBudgetStatus([])).toBe("customerPortal.analytics.noLimit");
    expect(customerBudgetStatus([{ source: "window", budget_duration: "1mo", max_budget: 10, spend: 0 }])).toBe(
      "customerPortal.analytics.withinBudget",
    );
  });

  it.each([
    ["1mo", "monthly"],
    ["monthly", "monthly"],
    ["30days", "monthly"],
    ["12mo", "yearly"],
    ["12months", "yearly"],
    ["365d", "other"],
    [null, "other"],
  ] as const)("labels the configured duration %s without computing its reset", (duration, period) => {
    expect(customerBudgetPeriod(duration)).toBe(period);
  });
});
