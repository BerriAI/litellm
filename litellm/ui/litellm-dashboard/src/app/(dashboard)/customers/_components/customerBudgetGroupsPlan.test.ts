import { describe, expect, it } from "vitest";
import type { Organization } from "./customerHierarchyForms";
import { customerBudgetGroupRows, planCustomerBudgetGroups } from "./customerBudgetGroupsPlan";
import { budgetEditorFields } from "./customerBudgetForm";

const organization: Organization = {
  customer_org_id: "org-1",
  budget_groups_revision: "a".repeat(64),
  default_budget_template_id: "standard",
  created_at: "",
  updated_at: "",
  created_by: "",
  updated_by: "",
  spend: 0,
  member_count: 0,
  default_budgets: [
    {
      budget_id: "standard",
      name: "Standard",
      max_budget: 25,
      budget_duration: "7d",
      soft_budget: 20,
      allowed_models: ["model-a", "model-b"],
      model_tpm_limit: { "model-a": 300 },
      model_rpm_limit: { "model-a": 3 },
      budget_limits: [
        { budget_duration: "1mo", max_budget: 10, reset_at: "2027-01-15T00:00:00Z" },
        { budget_duration: "12mo", max_budget: 100, reset_at: "2027-11-15T00:00:00Z" },
      ],
    },
    { budget_id: "legacy", name: "Legacy", max_budget: 25, budget_duration: "6days", rpm_limit: 30 },
    { budget_id: "rate", name: "Rate only", tpm_limit: 500 },
  ],
};
const storedWindows = organization.default_budgets[0].budget_limits!;

describe("planCustomerBudgetGroups", () => {
  it("keeps custom caps and their reset identity when calendar caps are edited or cleared", () => {
    const custom = { budget_duration: "7d", max_budget: 3, reset_at: "2027-01-04T00:00:00Z" };
    const original = {
      ...organization,
      default_budgets: [
        { ...organization.default_budgets[0], max_budget: null, budget_duration: null, budget_limits: [custom] },
      ],
    };
    const rows = customerBudgetGroupRows(original);
    expect(planCustomerBudgetGroups(original, rows, "standard")).toMatchObject({ changed: false });
    const monthly = { budget_duration: "1mo", max_budget: 5 };
    const edited = { ...rows[0], budgetWindows: [{ budget: monthly, maxBudget: "5" }, ...rows[0].budgetWindows] };
    expect(planCustomerBudgetGroups(original, [edited], "standard")).toMatchObject({
      body: { default_budgets: [{ budget_limits: [{ budget_duration: "1mo", max_budget: 5 }, custom] }] },
    });
    const withCalendar = {
      ...original,
      default_budgets: [
        { ...original.default_budgets[0], budget_limits: [...organization.default_budgets[0].budget_limits!, custom] },
      ],
    };
    expect(planCustomerBudgetGroups(withCalendar, rows, "standard")).toMatchObject({
      body: { default_budgets: [{ budget_limits: [custom] }] },
    });
  });

  it("should preserve an explicitly unset default while editing existing groups", () => {
    const original = { ...organization, default_budget_template_id: null };
    const rows = customerBudgetGroupRows(original);
    expect(planCustomerBudgetGroups(original, [{ ...rows[0], name: "Renamed" }, ...rows.slice(1)], "")).toMatchObject({
      ok: true,
      body: { default_budget_template: null },
    });
  });

  it("should omit unchanged rules, including scalar and rate-only templates", () => {
    expect(planCustomerBudgetGroups(organization, customerBudgetGroupRows(organization), "standard")).toEqual({
      ok: true,
      changed: false,
      body: {
        expected_revision: organization.budget_groups_revision,
        default_budget_template: "Standard",
        default_budgets: organization.default_budgets.map(({ budget_id, name }) => ({ budget_id, name })),
      },
    });
  });

  it("should group additions, deletions, name swaps, caps, policies and the default in one request", () => {
    const rows = customerBudgetGroupRows(organization);
    const result = planCustomerBudgetGroups(
      organization,
      [
        {
          ...rows[0],
          name: "Legacy",
          budgetWindows: [{ ...rows[0].budgetWindows[0], maxBudget: "0" }],
        },
        { ...rows[1], name: "Standard", capPolicy: "pause_allow_increase" },
        {
          ...budgetEditorFields({ max_budget: 30, budget_duration: "1mo" }),
          id: "new",
          name: " New ",
          capPolicy: "pause_until_reset",
        },
      ],
      "new",
    );
    expect(result).toEqual({
      ok: true,
      changed: true,
      body: {
        expected_revision: organization.budget_groups_revision,
        default_budget_template: "New",
        default_budgets: [
          {
            budget_id: "standard",
            name: "Legacy",
            budget_limits: [{ ...organization.default_budgets[0].budget_limits![0], max_budget: 0 }],
          },
          { budget_id: "legacy", name: "Standard", cap_policy: "pause_allow_increase" },
          {
            name: "New",
            cap_policy: "pause_until_reset",
            max_budget: 30,
            budget_duration: "1mo",
          },
        ],
      },
    });
  });

  it.each([
    [{ name: "" }, "groupNameRequired"],
    [{ name: "Legacy" }, "duplicateGroups"],
    [{ maxBudget: "-1" }, "invalidBudget"],
    [{ softBudget: "Infinity" }, "invalidBudget"],
    [{ budgetDuration: "6mo" }, "invalidBudget"],
  ])("should reject invalid edits before sending a request", (patch, error) => {
    const rows = customerBudgetGroupRows(organization);
    expect(planCustomerBudgetGroups(organization, [{ ...rows[0], ...patch }, ...rows.slice(1)], "standard")).toEqual({
      ok: false,
      error,
    });
  });

  it.each([[[]], [[" "]]])("rejects new groups with no spend, rate, or model limit %j", (allowedModels) => {
    const rows = customerBudgetGroupRows(organization);
    expect(
      planCustomerBudgetGroups(
        organization,
        [...rows, { ...rows[1], id: "new", name: "New", maxBudget: "", rpm: "", allowedModels }],
        "standard",
      ),
    ).toEqual({ ok: false, error: "groupLimitRequired" });
  });

  it.each([
    [
      { tpm: "50000", rpm: "0" },
      { tpm_limit: 50000, rpm_limit: 0 },
    ],
    [
      { maxBudget: "0", budgetDuration: "" },
      { max_budget: 0, budget_duration: null },
    ],
    [{ softBudget: "" }, { soft_budget: null }],
    [{ allowedModels: [" model-b ", "model-a", "model-a"] }, {}],
    [{ allowedModels: [" model-c ", "model-c", " "] }, { allowed_models: ["model-c"] }],
    [{ allowedModels: [] }, { allowed_models: null }],
    [{ budgetWindows: [] }, { budget_limits: null }],
    [
      {
        budgetWindows: storedWindows.map((budget, index) => ({
          budget,
          maxBudget: index === 0 ? "0" : String(budget.max_budget),
        })),
      },
      { budget_limits: [{ ...storedWindows[0], max_budget: 0 }, storedWindows[1]] },
    ],
  ])("patches only edited fields without clearing stored windows or hidden policies", (patch, expected) => {
    const rows = customerBudgetGroupRows(organization);
    const result = planCustomerBudgetGroups(
      organization,
      [{ ...rows[0], ...patch }, { ...rows[1], rpm: "" }, rows[2]],
      "standard",
    );
    expect(result).toEqual({
      ok: true,
      changed: true,
      body: {
        expected_revision: organization.budget_groups_revision,
        default_budget_template: "Standard",
        default_budgets: [
          { budget_id: "standard", name: "Standard", ...expected },
          { budget_id: "legacy", name: "Legacy", rpm_limit: null },
          { budget_id: "rate", name: "Rate only" },
        ],
      },
    });
  });

  it("allows a new rate-only organization group without a spend budget", () => {
    const result = planCustomerBudgetGroups(
      organization,
      [
        ...customerBudgetGroupRows(organization),
        {
          ...budgetEditorFields({ rpm_limit: 60 }),
          id: "new",
          name: "Rate plan",
          capPolicy: "pause_until_reset",
        },
      ],
      "",
      false,
    );
    expect(result).toMatchObject({
      ok: true,
      changed: true,
      body: {
        default_budgets: [
          { budget_id: "standard", name: "Standard" },
          { budget_id: "legacy", name: "Legacy" },
          { budget_id: "rate", name: "Rate only" },
          { name: "Rate plan", rpm_limit: 60 },
        ],
      },
    });
  });

  it.each(["-1", "0.5", "Infinity", "9007199254740992"])("rejects invalid group rates %s", (value) => {
    const rows = customerBudgetGroupRows(organization);
    for (const field of ["tpm", "rpm"] as const) {
      expect(
        planCustomerBudgetGroups(organization, [{ ...rows[0], [field]: value }, ...rows.slice(1)], "standard"),
      ).toEqual({ ok: false, error: "invalidRateLimit" });
    }
  });
});
