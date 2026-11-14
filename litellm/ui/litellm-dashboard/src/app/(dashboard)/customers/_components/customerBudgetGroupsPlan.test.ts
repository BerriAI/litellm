import { describe, expect, it } from "vitest";
import type { Organization } from "./customerHierarchyForms";
import { customerBudgetGroupRows, planCustomerBudgetGroups } from "./customerBudgetGroupsPlan";

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
      budget_limits: [
        { budget_duration: "1mo", max_budget: 10, reset_at: "2027-01-15T00:00:00Z" },
        { budget_duration: "12mo", max_budget: 100, reset_at: "2027-11-15T00:00:00Z" },
      ],
    },
    { budget_id: "legacy", name: "Legacy", max_budget: 25, budget_duration: "7d", rpm_limit: 30 },
    { budget_id: "rate", name: "Rate only", tpm_limit: 500 },
  ],
};

describe("planCustomerBudgetGroups", () => {
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
        { ...rows[0], name: "Legacy", monthlyBudget: "0", yearlyBudget: "" },
        { ...rows[1], name: "Standard", capPolicy: "pause_allow_increase" },
        { id: "new", name: " New ", monthlyBudget: "30", yearlyBudget: "300", capPolicy: "pause_until_reset" },
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
          { budget_id: "standard", name: "Legacy", budget_limits: [{ budget_duration: "1mo", max_budget: 0 }] },
          { budget_id: "legacy", name: "Standard", cap_policy: "pause_allow_increase" },
          {
            name: "New",
            cap_policy: "pause_until_reset",
            budget_limits: [
              { budget_duration: "1mo", max_budget: 30 },
              { budget_duration: "12mo", max_budget: 300 },
            ],
          },
        ],
      },
    });
  });

  it("should clear existing calendar caps without clearing other budget rules", () => {
    const rows = customerBudgetGroupRows(organization);
    expect(
      planCustomerBudgetGroups(
        organization,
        [{ ...rows[0], monthlyBudget: "", yearlyBudget: "" }, ...rows.slice(1)],
        "legacy",
      ),
    ).toMatchObject({
      ok: true,
      changed: true,
      body: {
        default_budget_template: "Legacy",
        default_budgets: [
          { budget_id: "standard", name: "Standard", budget_limits: [] },
          { budget_id: "legacy", name: "Legacy" },
          { budget_id: "rate", name: "Rate only" },
        ],
      },
    });
  });

  it.each([
    [{ name: "" }, "groupNameRequired"],
    [{ name: "Legacy" }, "duplicateGroups"],
    [{ monthlyBudget: "-1" }, "invalidBudget"],
    [{ monthlyBudget: "Infinity" }, "invalidBudget"],
  ])("should reject invalid edits before sending a request", (patch, error) => {
    const rows = customerBudgetGroupRows(organization);
    expect(planCustomerBudgetGroups(organization, [{ ...rows[0], ...patch }, ...rows.slice(1)], "standard")).toEqual({
      ok: false,
      error,
    });
  });

  it("should still require a limit for a newly created group", () => {
    const rows = customerBudgetGroupRows(organization);
    expect(
      planCustomerBudgetGroups(organization, [...rows, { ...rows[1], id: "new", name: "New" }], "standard"),
    ).toEqual({ ok: false, error: "groupLimitRequired" });
  });
});
