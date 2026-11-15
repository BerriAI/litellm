import { describe, expect, it } from "vitest";

import { userBudgetSchema } from "./userBudgetForm";

describe("userBudgetSchema", () => {
  it.each(["0", "0.001", "42.765"])("accepts a finite non-negative budget of %s", (max_budget) => {
    expect(userBudgetSchema.safeParse({ unlimited: false, max_budget, budget_duration: "7d" }).success).toBe(true);
  });

  it.each(["", " ", "-1", "NaN", "Infinity", "12abc"])("rejects an invalid finite budget of %s", (max_budget) => {
    expect(userBudgetSchema.safeParse({ unlimited: false, max_budget, budget_duration: null }).success).toBe(false);
  });

  it("represents unlimited as null without requiring a disabled amount", () => {
    expect(userBudgetSchema.safeParse({ unlimited: true, max_budget: "", budget_duration: null }).success).toBe(true);
  });
});
