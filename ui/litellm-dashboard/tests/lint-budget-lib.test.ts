import { describe, it, expect } from "vitest";
import { countBudgetViolations } from "../scripts/lint-budget-lib.mjs";

const budgets = {
  "@typescript-eslint/no-explicit-any": { max: 10, target: 5 },
  complexity: { max: 10, target: 5 },
};

const file = (...ruleIds: (string | null)[]) => ({ messages: ruleIds.map((ruleId) => ({ ruleId })) });

describe("countBudgetViolations", () => {
  it("counts only budgeted rules and sums across files", () => {
    const report = [
      file("@typescript-eslint/no-explicit-any", "complexity", "max-depth"),
      file("@typescript-eslint/no-explicit-any", "no-var", null),
    ];
    expect(countBudgetViolations(report, budgets)).toEqual({
      "@typescript-eslint/no-explicit-any": 2,
      complexity: 1,
    });
  });

  it("reports 0 for a budgeted rule with no violations", () => {
    expect(countBudgetViolations([file("complexity")], budgets)).toEqual({
      "@typescript-eslint/no-explicit-any": 0,
      complexity: 1,
    });
  });

  it("emits keys in sorted order so the committed snapshot diffs stably", () => {
    const unsorted = { complexity: { max: 1, target: 1 }, "@typescript-eslint/no-explicit-any": { max: 1, target: 1 } };
    expect(Object.keys(countBudgetViolations([], unsorted))).toEqual([
      "@typescript-eslint/no-explicit-any",
      "complexity",
    ]);
  });
});
