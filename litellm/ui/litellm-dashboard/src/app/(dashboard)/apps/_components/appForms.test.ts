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
const fields = compactBudgetEditorFields(preset);
const [monthly, yearly] = fields.budgetWindows;

describe("app budget forms", () => {
  it.each(["1mo", "01mo", "30d", "monthly"])("rejects a second monthly limit expressed as %s", (period) => {
    const budget = {
      max_budget: 100,
      budget_duration: "1mo",
      budget_limits: [{ max_budget: 200, budget_duration: period }],
    };
    expect(
      compactBudgetEditorValue(compactBudgetEditorFields(budget), { max_budget: 100, budget_duration: "1mo" }),
    ).toEqual({
      ok: false,
      error: "customerPortal.budgetEditor.duplicatePeriod",
    });
  });

  it("allows a monthly limit alongside a no-reset cap but rejects duplicate additional periods", () => {
    const budget = { max_budget: 500, budget_limits: [{ max_budget: 100, budget_duration: "1mo" }] };
    const fields = compactBudgetEditorFields(budget);
    expect(compactBudgetEditorValue(fields)).toMatchObject({ ok: true });
    expect(
      compactBudgetEditorValue({ ...fields, budgetWindows: [...fields.budgetWindows, ...fields.budgetWindows] }),
    ).toEqual({ ok: false, error: "customerPortal.budgetEditor.duplicatePeriod" });
  });

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
    {
      field: "additional cap to zero",
      changes: { budgetWindows: [{ ...monthly, maxBudget: "0" }, yearly] },
      expected: { budget_limits: [{ ...preset.budget_limits[0], max_budget: 0 }, preset.budget_limits[1]] },
    },
  ])("preserves stored windows, reset dates and model policy when changing $field", ({ changes, expected }) => {
    expect(compactBudgetEditorValue({ ...fields, ...changes }, preset)).toEqual({
      ok: true,
      value: { ...preset, ...expected },
    });
  });

  it("clears a disabled budget pointer", () => {
    expect(budgetEditorValue({ ...budgetEditorFields(preset), budgetEnabled: false }, preset)).toEqual({
      ok: true,
      value: null,
    });
  });

  it.each([
    ...["", "-1", "Infinity", "bad"].map((amount) => ["window", amount] as const),
    ...["-1", "NaN", "Infinity", "1.2.3"].map((amount) => ["main", amount] as const),
  ])("rejects invalid %s spend %s", (location, maxBudget) => {
    const invalid =
      location === "main"
        ? { ...budgetEditorFields({}), maxBudget }
        : { ...fields, budgetWindows: fields.budgetWindows.map((window) => ({ ...window, maxBudget })) };
    expect(budgetEditorValue(invalid, location === "main" ? undefined : preset)).toEqual({
      ok: false,
      error: "customerPortal.hierarchy.invalidNumber",
    });
  });

  it.each(["6days", "6mo"])("rejects unsupported primary and additional periods %s", (budgetDuration) => {
    const invalidPeriod = { ok: false, error: "customerPortal.budgetEditor.invalidPeriod" };
    expect(budgetEditorValue({ ...fields, budgetDuration })).toEqual(invalidPeriod);
    expect(
      budgetEditorValue({
        ...fields,
        budgetWindows: [{ ...monthly, budget: { ...monthly.budget, budget_duration: budgetDuration } }],
      }),
    ).toEqual(invalidPeriod);
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

  it.each([
    {
      change: "main cap",
      edits: { maxBudget: "750" },
      expected: { max_budget: 750, budget_duration: "1mo" },
    },
    {
      change: "explicit clears and zero",
      edits: { tpm: "", rpm: "0", allowedModels: [], budgetWindows: [] },
      expected: { tpm_limit: null, rpm_limit: 0, allowed_models: null, budget_limits: null },
    },
    { change: "clear main cap", edits: { maxBudget: " " }, expected: { max_budget: null } },
    {
      change: "clear additional cap",
      edits: { budgetWindows: [{ ...monthly, maxBudget: " " }, yearly] },
      expected: { budget_limits: [preset.budget_limits[1]] },
    },
  ])("sends only $change without resending unchanged policy", ({ edits, expected }) => {
    expect(permanentBudgetEditorValue({ ...fields, ...edits }, preset)).toEqual({ ok: true, value: expected });
  });

  it.each([null, preset])("omits untouched fields for an enabled budget %#", (original) => {
    expect(
      permanentBudgetEditorValue({ ...compactBudgetEditorFields(original), budgetEnabled: true }, original),
    ).toEqual({
      ok: true,
      value: {},
    });
  });

  it.each(["monthly", "6days"])("preserves the unchanged stored period %s in sparse edits", (budget_duration) => {
    const original = { ...preset, budget_duration };
    expect(permanentBudgetEditorValue({ ...compactBudgetEditorFields(original), rpm: "25" }, original)).toEqual({
      ok: true,
      value: { rpm_limit: 25 },
    });
  });
});

describe("app surcharge form", () => {
  it.each([
    ["2", 0.02],
    [" 2.5 ", 0.025],
    ["0", 0],
    ["1000", 10],
    ["  ", null],
  ])("should send %s percent as the fraction %s", (text, fraction) => {
    expect(appSurchargeValue(text)).toEqual({ ok: true, value: fraction });
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
