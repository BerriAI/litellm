import {
  budgetEditorFields,
  budgetEditorValue,
  compactBudgetEditorFields,
  compactBudgetEditorValue,
  budgetPeriodAmount,
  updateBudgetPeriod,
  permanentBudgetEditorValue,
} from "../../customers/_components/customerBudgetForm";
import { describe, expect, it } from "vitest";
import { appSurchargeField, appSurchargeValue } from "./appForms";

describe("app budget forms", () => {
  it("saves a rate-only budget and preserves zero limits", () => {
    expect(budgetEditorValue({ ...budgetEditorFields({}), tpm: "50000", rpm: "0" })).toMatchObject({
      ok: true,
      value: { tpm_limit: 50000, rpm_limit: 0 },
    });
  });

  it("omits blank first-custom fields so the backend can retain inherited restrictions", () => {
    expect(
      permanentBudgetEditorValue({ ...compactBudgetEditorFields(), budgetEnabled: true, maxBudget: "0" }, null),
    ).toEqual({
      ok: true,
      value: { max_budget: 0, budget_duration: "1mo" },
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
    const fields = { ...compactBudgetEditorFields(original), tpm: "", rpm: "", allowedModels: [], budgetWindows: [] };
    expect(permanentBudgetEditorValue(fields, original)).toEqual({
      ok: true,
      value: { tpm_limit: null, rpm_limit: null, allowed_models: null, budget_limits: null },
    });
  });

  it("preserves spend windows and model rules when changing RPM", () => {
    const original = {
      tpm_limit: 50000,
      rpm_limit: 60,
      model_tpm_limit: { "test-model": 10000 },
      budget_limits: [{ budget_duration: "1mo", max_budget: 100, reset_at: "2027-01-01T00:00:00Z" }],
    };
    expect(budgetEditorValue({ ...budgetEditorFields(original), rpm: "" }, original)).toMatchObject({
      ok: true,
      value: { ...original, rpm_limit: null },
    });
  });

  it("should preserve limits not edited by the shared fields", () => {
    const original = { max_budget: 10, budget_limits: [{ budget_duration: "1mo" as const, max_budget: 5 }] };
    const result = budgetEditorValue({ ...budgetEditorFields(original), maxBudget: "20" }, original);
    expect(result).toMatchObject({ ok: true, value: { max_budget: 20, budget_limits: original.budget_limits } });
  });
  it("should clear a disabled budget pointer", () => {
    expect(budgetEditorValue(budgetEditorFields())).toEqual({ ok: true, value: null });
  });
  it("should edit an existing monthly window without leaving its old cap active", () => {
    const original = { budget_limits: [{ budget_duration: "1mo" as const, max_budget: 5 }] };
    const fields = budgetEditorFields(original);
    fields.budgetWindows[0].maxBudget = "25";
    expect(budgetEditorValue(fields, original)).toMatchObject({
      ok: true,
      value: { budget_limits: [{ budget_duration: "1mo", max_budget: 25 }] },
    });
  });
  it("should reject a cleared window amount until the limit is explicitly removed", () => {
    const fields = budgetEditorFields({ budget_limits: [{ budget_duration: "1mo", max_budget: 5 }] });
    fields.budgetWindows[0].maxBudget = "";
    expect(budgetEditorValue(fields).ok).toBe(false);
  });
  it.each(["-1", "NaN", "Infinity", "1.2.3"])("should reject invalid numeric input %s", (maxBudget) => {
    expect(budgetEditorValue({ ...budgetEditorFields({}), maxBudget }).ok).toBe(false);
  });
});

describe("compact app budget forms", () => {
  it("saves monthly and yearly caps together with rate limits", () => {
    const monthly = updateBudgetPeriod({ ...compactBudgetEditorFields(), budgetEnabled: true }, "1mo", "100");
    const yearly = updateBudgetPeriod(monthly, "12mo", "1000");
    expect(compactBudgetEditorValue({ ...yearly, tpm: "50000", rpm: "100" })).toMatchObject({
      ok: true,
      value: {
        max_budget: 100,
        budget_duration: "1mo",
        budget_limits: [{ budget_duration: "12mo", max_budget: 1000 }],
        tpm_limit: 50000,
        rpm_limit: 100,
      },
    });
  });

  it.each(["1mo", "12mo"] as const)("saves only the selected %s cap, including zero", (period) => {
    const fields = updateBudgetPeriod({ ...compactBudgetEditorFields(), budgetEnabled: true }, period, "0");
    const result = compactBudgetEditorValue(fields);
    expect(result.ok).toBe(true);
    if (!result.ok) return;
    expect(budgetPeriodAmount(compactBudgetEditorFields(result.value), period)).toBe("0");
    expect(budgetPeriodAmount(compactBudgetEditorFields(result.value), period === "1mo" ? "12mo" : "1mo")).toBe("");
  });

  it("updates every matching cap without keeping an old lower limit, while preserving reset dates and other limits", () => {
    const original = {
      max_budget: 80,
      budget_duration: "1mo",
      soft_budget: 50,
      model_tpm_limit: { "test-model": 5000 },
      budget_limits: [
        { budget_duration: "1mo", max_budget: 100, reset_at: "2027-01-01T00:00:00Z" },
        { budget_duration: "12mo", max_budget: 1000, reset_at: "2027-07-01T00:00:00Z" },
        { budget_duration: "1d", max_budget: 20, reset_at: "2026-12-15T00:00:00Z" },
      ],
    };
    const fields = compactBudgetEditorFields(original);
    expect(budgetPeriodAmount(fields, "1mo")).toBe("80");
    const updated = updateBudgetPeriod(fields, "1mo", "150");
    expect(compactBudgetEditorValue(updated, original)).toMatchObject({
      ok: true,
      value: {
        ...original,
        max_budget: 150,
        budget_limits: [{ ...original.budget_limits[0], max_budget: 150 }, ...original.budget_limits.slice(1)],
      },
    });
  });

  it("removes both scalar and window caps when a period is cleared", () => {
    const original = {
      max_budget: 80,
      budget_duration: "1mo",
      budget_limits: [
        { budget_duration: "1mo", max_budget: 100 },
        { budget_duration: "12mo", max_budget: 1000 },
      ],
    };
    const monthlyCleared = updateBudgetPeriod(compactBudgetEditorFields(original), "1mo", " ");
    expect(compactBudgetEditorValue(monthlyCleared, original)).toMatchObject({
      ok: true,
      value: { max_budget: null, budget_limits: [original.budget_limits[1]] },
    });
    const allCleared = updateBudgetPeriod(monthlyCleared, "12mo", "");
    expect(compactBudgetEditorValue({ ...allCleared, tpm: "25000" }, original)).toMatchObject({
      ok: true,
      value: { max_budget: null, budget_limits: null, tpm_limit: 25000 },
    });
  });

  it.each(["-1", "Infinity", "bad"])("rejects invalid yearly cap %s", (amount) => {
    const fields = updateBudgetPeriod({ ...compactBudgetEditorFields(), budgetEnabled: true }, "12mo", amount);
    expect(compactBudgetEditorValue(fields).ok).toBe(false);
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

describe("permanent budget patches", () => {
  const preset = {
    max_budget: 500,
    budget_duration: "1mo",
    tpm_limit: 10000,
    rpm_limit: 50,
    allowed_models: ["test-model"],
    model_tpm_limit: { "test-model": 1000 },
    budget_limits: [{ budget_duration: "12mo", max_budget: 5000, reset_at: "2027-01-01T00:00:00Z" }],
  };
  it("changes only the edited cap so backend snapshots retain untouched preset settings", () => {
    const fields = updateBudgetPeriod(compactBudgetEditorFields(preset), "1mo", "750");
    expect(permanentBudgetEditorValue(fields, preset)).toEqual({
      ok: true,
      value: { max_budget: 750, budget_duration: "1mo" },
    });
  });
  it("distinguishes unchanged blank values from explicit clears, including zero", () => {
    expect(permanentBudgetEditorValue({ ...compactBudgetEditorFields(preset), tpm: "", rpm: "0" }, preset)).toEqual({
      ok: true,
      value: { tpm_limit: null, rpm_limit: 0 },
    });
    expect(permanentBudgetEditorValue({ ...compactBudgetEditorFields(null), budgetEnabled: true }, null)).toEqual({
      ok: true,
      value: {},
    });
  });
  it("clears a yearly cap intentionally without clearing model or rate policy", () => {
    const fields = updateBudgetPeriod(compactBudgetEditorFields(preset), "12mo", "");
    expect(permanentBudgetEditorValue(fields, preset)).toEqual({ ok: true, value: { budget_limits: null } });
  });
});
