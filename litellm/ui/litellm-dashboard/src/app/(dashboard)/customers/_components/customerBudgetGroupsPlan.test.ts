import { describe, expect, it } from "vitest";
import type { Organization } from "./customerHierarchyForms";
import { customerBudgetGroupRows, planCustomerBudgetGroups } from "./customerBudgetGroupsPlan";

const organization: Organization = {
  customer_org_id: "org-1",
  alias: "Example",
  default_budget_template_id: "standard",
  created_at: "",
  updated_at: "",
  created_by: "",
  updated_by: "",
  spend: 0,
  team_count: 0,
  member_count: 0,
  default_budgets: [
    {
      budget_id: "standard",
      name: "Standard",
      cap_policy: "pause_allow_increase",
      budget_limits: [
        { budget_duration: "1mo", max_budget: 10 },
        { budget_duration: "12mo", max_budget: 100 },
        { budget_duration: "1d", max_budget: 5 },
      ],
    },
    { budget_id: "old", name: "Old", budget_limits: [{ budget_duration: "1mo", max_budget: 20 }] },
  ],
};

describe("planCustomerBudgetGroups", () => {
  it("should produce a no-op for unchanged groups regardless of limit order", () => {
    expect(planCustomerBudgetGroups(organization, customerBudgetGroupRows(organization), "standard")).toEqual({
      ok: true,
      creates: [],
      updates: [],
      removes: [],
      defaultChange: null,
      changed: false,
    });
  });

  it("should plan additions, renames, cleared yearly limits, removals and a newly created default", () => {
    const rows = customerBudgetGroupRows(organization);
    expect(
      planCustomerBudgetGroups(
        organization,
        [
          { ...rows[0], name: "Renamed", monthlyBudget: "0", yearlyBudget: "" },
          { id: "new", name: "New", monthlyBudget: "30", yearlyBudget: "300" },
        ],
        "new",
      ),
    ).toEqual({
      ok: true,
      changed: true,
      creates: [
        {
          rowId: "new",
          body: {
            name: "New",
            budget_limits: [
              { budget_duration: "1mo", max_budget: 30 },
              { budget_duration: "12mo", max_budget: 300 },
            ],
          },
        },
      ],
      updates: [
        {
          budgetId: "standard",
          body: {
            name: "Renamed",
            budget_limits: [
              { budget_duration: "1d", max_budget: 5 },
              { budget_duration: "1mo", max_budget: 0 },
            ],
          },
        },
      ],
      removes: ["old"],
      defaultChange: { rowId: "new", name: "New" },
    });
  });

  it("should send only a name when only renaming and select an existing default", () => {
    const rows = customerBudgetGroupRows(organization);
    const result = planCustomerBudgetGroups(organization, [{ ...rows[0], name: "Renamed" }, rows[1]], "old");
    expect(result).toMatchObject({
      ok: true,
      updates: [{ budgetId: "standard", body: { name: "Renamed" } }],
      defaultChange: { rowId: "old" },
    });
  });

  it("should reject swaps and reusing a removed name before sending requests", () => {
    const rows = customerBudgetGroupRows(organization);
    expect(
      planCustomerBudgetGroups(
        organization,
        [
          { ...rows[0], name: "Old" },
          { ...rows[1], name: "Standard" },
        ],
        "standard",
      ),
    ).toEqual({ ok: false, error: "renameCollision" });
  });

  it.each([
    [[], "", "defaultRequired"],
    [[{ id: "a", name: "", monthlyBudget: "1", yearlyBudget: "" }], "a", "groupNameRequired"],
    [[{ id: "a", name: "A", monthlyBudget: "", yearlyBudget: "" }], "a", "groupLimitRequired"],
    [[{ id: "a", name: "A", monthlyBudget: "-1", yearlyBudget: "" }], "a", "invalidBudget"],
  ] as const)("should validate invalid edits", (groups, defaultId, error) => {
    expect(planCustomerBudgetGroups(organization, [...groups], defaultId)).toEqual({ ok: false, error });
  });
});
