import { describe, expect, it } from "vitest";
import { appBudgetFields, appBudgetValue, appSurchargeField, appSurchargeValue } from "./appForms";

describe("app budget forms", () => {
  it("saves a rate-only budget and preserves zero limits", () => {
    expect(appBudgetValue({ ...appBudgetFields({}), tpm: "50000", rpm: "0" })).toMatchObject({
      ok: true,
      value: { tpm_limit: 50000, rpm_limit: 0 },
    });
  });

  it("omits blank first-custom fields so the backend can retain inherited restrictions", () => {
    expect(appBudgetValue({ ...appBudgetFields(), budgetEnabled: true, maxBudget: "0" }, null)).toEqual({
      ok: true,
      value: { max_budget: 0 },
    });
  });

  it("keeps explicit clears when editing an existing private budget", () => {
    const original = {
      max_budget: 20,
      tpm_limit: 1700,
      rpm_limit: 17,
      allowed_models: ["test-model"],
      budget_limits: [{ budget_duration: "12mo", max_budget: 6000 }],
    };
    const fields = { ...appBudgetFields(original), tpm: "", rpm: "", allowedModels: [], budgetWindows: [] };
    expect(appBudgetValue(fields, original)).toMatchObject({
      ok: true,
      value: { max_budget: 20, tpm_limit: null, rpm_limit: null, allowed_models: null, budget_limits: null },
    });
  });

  it("preserves spend windows and model rules when changing RPM", () => {
    const original = {
      tpm_limit: 50000,
      rpm_limit: 60,
      model_tpm_limit: { "test-model": 10000 },
      budget_limits: [{ budget_duration: "1mo", max_budget: 100, reset_at: "2027-01-01T00:00:00Z" }],
    };
    expect(appBudgetValue({ ...appBudgetFields(original), rpm: "" }, original)).toMatchObject({
      ok: true,
      value: { ...original, rpm_limit: null },
    });
  });

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

describe("app surcharge form", () => {
  it.each([
    ["2", 0.02],
    [" 2.5 ", 0.025],
    ["0", 0],
    ["1000", 10],
  ])("should send %s percent as the fraction %s", (text, fraction) => {
    expect(appSurchargeValue(text)).toEqual({ ok: true, value: fraction });
  });

  it("should send an empty field as null so the app falls back to the gateway surcharge", () => {
    expect(appSurchargeValue("  ")).toEqual({ ok: true, value: null });
  });

  it.each(["-1", "1000.1", "two", "Infinity"])("should refuse %s", (text) => {
    expect(appSurchargeValue(text)).toEqual({ ok: false });
  });

  it.each([
    [0.02, "2"],
    [0.025, "2.5"],
    [0, "0"],
    [null, ""],
    [undefined, ""],
  ])("should show the stored fraction %s as %s percent", (margin, text) => {
    expect(appSurchargeField(margin)).toBe(text);
  });
});
