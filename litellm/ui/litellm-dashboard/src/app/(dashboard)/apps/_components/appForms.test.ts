import {
  budgetEditorFields,
  budgetEditorValue,
  compactBudgetEditorFields,
  compactBudgetEditorValue,
  permanentBudgetEditorValue,
  type Budget,
} from "../../customers/_components/customerBudgetForm";
import { describe, expect, it } from "vitest";
import { appSurchargeField, appSurchargeValue } from "./appForms";

const preset = {
  max_budget: 500,
  budget_duration: "1mo",
  soft_budget: 80,
  tpm_limit: 10000,
  rpm_limit: 50,
  allowed_models: ["test-model"],
  model_tpm_limit: { "test-model": 1000 },
  model_rpm_limit: { "test-model": 10 },
  cap_policy: "pause_allow_increase",
  budget_limits: [
    { budget_duration: "1mo", max_budget: 100, reset_at: "2027-01-01T00:00:00Z" },
    { budget_duration: "12mo", max_budget: 5000, reset_at: "2027-07-01T00:00:00Z" },
  ],
} satisfies Budget;

describe("app budget forms", () => {
  it("saves a rate-only budget and preserves zero limits", () => {
    expect(budgetEditorValue({ ...budgetEditorFields({}), tpm: "50000", rpm: "0" })).toEqual({
      ok: true,
      value: {
        max_budget: null,
        budget_duration: null,
        soft_budget: null,
        budget_limits: null,
        allowed_models: null,
        tpm_limit: 50000,
        rpm_limit: 0,
      },
    });
  });

  it.each([
    { field: "main spend", changes: { maxBudget: "750" }, expected: { max_budget: 750 } },
    { field: "RPM", changes: { rpm: "" }, expected: { rpm_limit: null } },
  ])("preserves stored windows, reset dates and model policy when changing $field", ({ changes, expected }) => {
    expect(compactBudgetEditorValue({ ...compactBudgetEditorFields(preset), ...changes }, preset)).toEqual({
      ok: true,
      value: { ...preset, ...expected },
    });
  });

  it("edits an additional cap to zero without changing the main cap, other window or reset dates", () => {
    const fields = compactBudgetEditorFields(preset);
    expect(
      compactBudgetEditorValue(
        {
          ...fields,
          budgetWindows: fields.budgetWindows.map((window, index) =>
            index === 0 ? { ...window, maxBudget: "0" } : window,
          ),
        },
        preset,
      ),
    ).toEqual({
      ok: true,
      value: { ...preset, budget_limits: [{ ...preset.budget_limits[0], max_budget: 0 }, preset.budget_limits[1]] },
    });
  });

  it("clears a disabled budget pointer", () => {
    expect(budgetEditorValue({ ...budgetEditorFields(preset), budgetEnabled: false }, preset)).toEqual({
      ok: true,
      value: null,
    });
  });

  it.each(["", "-1", "Infinity", "bad"])("rejects an invalid stored window amount %s", (maxBudget) => {
    const fields = budgetEditorFields(preset);
    expect(
      budgetEditorValue(
        {
          ...fields,
          budgetWindows: fields.budgetWindows.map((window) => ({ ...window, maxBudget })),
        },
        preset,
      ),
    ).toEqual({ ok: false, error: "customerPortal.hierarchy.invalidNumber" });
  });

  it.each(["-1", "NaN", "Infinity", "1.2.3"])("rejects invalid main spend %s", (maxBudget) => {
    expect(budgetEditorValue({ ...budgetEditorFields({}), maxBudget })).toEqual({
      ok: false,
      error: "customerPortal.hierarchy.invalidNumber",
    });
  });
});

describe("permanent budget patches", () => {
  it.each(["", "1mo", "12mo", "7d"])(
    "saves zero main spend with period %s without sending blank limits",
    (budgetDuration) => {
      const fields = { ...compactBudgetEditorFields(), budgetEnabled: true, maxBudget: "0", budgetDuration };
      expect(permanentBudgetEditorValue(fields, null)).toEqual({
        ok: true,
        value: { max_budget: 0, budget_duration: budgetDuration || null },
      });
    },
  );

  it("changes only the main cap so backend snapshots retain untouched preset settings", () => {
    expect(permanentBudgetEditorValue({ ...compactBudgetEditorFields(preset), maxBudget: "750" }, preset)).toEqual({
      ok: true,
      value: { max_budget: 750, budget_duration: "1mo" },
    });
  });

  it("keeps explicit clears and zero without resending unchanged spend or model rates", () => {
    const fields = { ...compactBudgetEditorFields(preset), tpm: "", rpm: "0", allowedModels: [], budgetWindows: [] };
    expect(permanentBudgetEditorValue(fields, preset)).toEqual({
      ok: true,
      value: { tpm_limit: null, rpm_limit: 0, allowed_models: null, budget_limits: null },
    });
  });

  it("clears main and additional caps independently", () => {
    const fields = compactBudgetEditorFields(preset);
    expect(permanentBudgetEditorValue({ ...fields, maxBudget: " " }, preset)).toEqual({
      ok: true,
      value: { max_budget: null },
    });
    expect(
      permanentBudgetEditorValue(
        {
          ...fields,
          budgetWindows: fields.budgetWindows.map((window, index) =>
            index === 0 ? { ...window, maxBudget: " " } : window,
          ),
        },
        preset,
      ),
    ).toEqual({ ok: true, value: { budget_limits: [preset.budget_limits[1]] } });
  });

  it.each([null, preset])("omits untouched fields for an enabled budget %#", (original) => {
    expect(
      permanentBudgetEditorValue({ ...compactBudgetEditorFields(original), budgetEnabled: true }, original),
    ).toEqual({
      ok: true,
      value: {},
    });
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
