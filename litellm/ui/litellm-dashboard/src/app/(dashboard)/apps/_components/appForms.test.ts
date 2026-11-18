import { describe, expect, it } from "vitest";
import { appBudgetFields, appBudgetValue, appTierBudgetValue } from "./appForms";

describe("app budget forms", () => {
  it("preserves limits not edited by the shared fields", () => {
    const original = { max_budget: 10, budget_limits: [{ budget_duration: "1mo" as const, max_budget: 5 }] };
    const result = appBudgetValue({ ...appBudgetFields(original), maxBudget: "20" }, original);
    expect(result).toMatchObject({ ok: true, value: { max_budget: 20, budget_limits: original.budget_limits } });
  });
  it("clears a disabled budget pointer and every disabled tier limit", () => {
    expect(appBudgetValue(appBudgetFields())).toEqual({ ok: true, value: null });
    expect(appTierBudgetValue({ ...appBudgetFields({ max_budget: 10 }), budgetEnabled: false })).toMatchObject({
      ok: true,
      value: { max_budget: null, budget_limits: null, rpm_limit: null },
    });
  });
  it.each(["-1", "NaN", "Infinity", "1.2.3"])("rejects invalid numeric input %s", (maxBudget) => {
    expect(appBudgetValue({ ...appBudgetFields({}), maxBudget }).ok).toBe(false);
  });
});
