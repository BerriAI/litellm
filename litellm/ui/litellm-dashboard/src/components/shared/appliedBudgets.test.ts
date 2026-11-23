import { describe, expect, it } from "vitest";

import type { InheritedBudgetGate } from "./InheritedBudgetHint";
import { appliedBudgets, budgetUsagePercent, closestToLimit, type AppliedBudget } from "./appliedBudgets";

const teamGate: InheritedBudgetGate = {
  scope: "Team",
  alias: "Platform",
  maxBudget: 1000,
  budgetDuration: "30d",
  spend: 900,
};
const orgGate: InheritedBudgetGate = { scope: "Organization", alias: "Acme", maxBudget: 5000, budgetDuration: null };

const budget = (overrides: Partial<AppliedBudget>): AppliedBudget => ({
  scope: "Team",
  name: "Platform",
  maxBudget: 100,
  spend: 0,
  budgetDuration: null,
  ...overrides,
});

describe("appliedBudgets", () => {
  it("puts the key's own budget first, followed by every inherited budget", () => {
    expect(appliedBudgets({ max_budget: 25, spend: 5, budget_duration: "7d" }, "ci-key", [teamGate, orgGate])).toEqual([
      { scope: "Key", name: "ci-key", maxBudget: 25, spend: 5, budgetDuration: "7d" },
      { scope: "Team", name: "Platform", maxBudget: 1000, spend: 900, budgetDuration: "30d" },
      { scope: "Organization", name: "Acme", maxBudget: 5000, spend: null, budgetDuration: null },
    ]);
  });

  it("leaves the key out when it has no budget of its own", () => {
    expect(appliedBudgets({ max_budget: null, spend: 5 }, "ci-key", [orgGate]).map((b) => b.scope)).toEqual([
      "Organization",
    ]);
  });

  it("keeps a zero key budget, since it blocks every request", () => {
    expect(appliedBudgets({ max_budget: 0, spend: 0 }, "ci-key", [])).toHaveLength(1);
  });
});

describe("budgetUsagePercent", () => {
  it("returns spend as a share of the limit", () => {
    expect(budgetUsagePercent(budget({ spend: 30, maxBudget: 120 }))).toBe(25);
  });

  it("returns null when spend is unknown or the limit is not positive", () => {
    expect(budgetUsagePercent(budget({ spend: null }))).toBeNull();
    expect(budgetUsagePercent(budget({ maxBudget: 0, spend: 1 }))).toBeNull();
  });
});

describe("closestToLimit", () => {
  it("picks the budget with the highest share used, not the smallest limit", () => {
    const key = budget({ scope: "Key", maxBudget: 25, spend: 5 });
    const team = budget({ scope: "Team", maxBudget: 1000, spend: 900 });
    expect(closestToLimit([key, team])).toBe(team);
  });

  it("ignores budgets whose spend is unknown", () => {
    const team = budget({ scope: "Team", maxBudget: 1000, spend: 10 });
    const org = budget({ scope: "Organization", maxBudget: 1, spend: null });
    expect(closestToLimit([team, org])).toBe(team);
  });

  it("highlights nothing when only one budget applies or no spend is known", () => {
    expect(closestToLimit([budget({ spend: 99 })])).toBeNull();
    expect(closestToLimit([budget({ spend: null }), budget({ scope: "User", spend: null })])).toBeNull();
  });
});
