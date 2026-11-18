import { describe, expect, it } from "vitest";
import { appBudgetFields, appBudgetValue } from "./appForms";

describe("app budget forms", () => {
  it("should preserve limits not edited by the shared fields", () => {
    const original = { max_budget: 10, budget_limits: [{ budget_duration: "1mo" as const, max_budget: 5 }] };
    const result = appBudgetValue({ ...appBudgetFields(original), maxBudget: "20" }, original);
    expect(result).toMatchObject({ ok: true, value: { max_budget: 20, budget_limits: original.budget_limits } });
  });
  it("should clear a disabled budget pointer", () => {
    expect(appBudgetValue(appBudgetFields())).toEqual({ ok: true, value: null });
  });
  it("should edit an existing monthly window without leaving its old cap active", () => {
    const original = { budget_limits: [{ budget_duration: "1mo" as const, max_budget: 5 }] };
    const fields = appBudgetFields(original);
    fields.budgetWindows[0].maxBudget = "25";
    expect(appBudgetValue(fields, original)).toMatchObject({
      ok: true,
      value: { budget_limits: [{ budget_duration: "1mo", max_budget: 25 }] },
    });
  });
  it("should reject a cleared window amount until the limit is explicitly removed", () => {
    const fields = appBudgetFields({ budget_limits: [{ budget_duration: "1mo", max_budget: 5 }] });
    fields.budgetWindows[0].maxBudget = "";
    expect(appBudgetValue(fields).ok).toBe(false);
  });
  it.each(["-1", "NaN", "Infinity", "1.2.3"])("should reject invalid numeric input %s", (maxBudget) => {
    expect(appBudgetValue({ ...appBudgetFields({}), maxBudget }).ok).toBe(false);
  });
});
