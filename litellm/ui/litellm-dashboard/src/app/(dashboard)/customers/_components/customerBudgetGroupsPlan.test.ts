import { describe, expect, it } from "vitest";
import type { Organization } from "./customerHierarchyForms";
import { customerBudgetGroupRows, planCustomerBudgetGroups } from "./customerBudgetGroupsPlan";
import { defaultCustomerCapPolicy } from "./customerOrganizationForm";

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
    const expected = {
      ok: true,
      creates: [],
      updates: [],
      removes: [],
      defaultChange: null,
      changed: false,
    };
    expect(planCustomerBudgetGroups(organization, customerBudgetGroupRows(organization), "standard")).toEqual(expected);
  });

  it("should plan additions, renames, cleared yearly limits, removals and a newly created default", () => {
    const rows = customerBudgetGroupRows(organization);
    const result = planCustomerBudgetGroups(
      organization,
      [
        { ...rows[0], name: "Renamed", monthlyBudget: "0", yearlyBudget: "" },
        { id: "new", name: "New", monthlyBudget: "30", yearlyBudget: "300", capPolicy: "pause_allow_increase" },
      ],
      "new",
    );
    const expected = {
      ok: true,
      changed: true,
      creates: [
        {
          rowId: "new",
          body: {
            name: "New",
            cap_policy: "pause_allow_increase",
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
    };
    expect(result).toEqual(expected);
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

  it("should patch only the cap policy when it is the only change", () => {
    const rows = customerBudgetGroupRows(organization);
    expect(rows[0].capPolicy).toBe("pause_allow_increase");
    expect(rows[1].capPolicy).toBe("pause_until_reset");
    const expected = {
      ok: true,
      creates: [],
      updates: [{ budgetId: "standard", body: { cap_policy: "pause_until_reset" } }],
      removes: [],
      defaultChange: null,
      changed: true,
    };
    expect(
      planCustomerBudgetGroups(organization, [{ ...rows[0], capPolicy: "pause_until_reset" }, rows[1]], "standard"),
    ).toEqual(expected);
  });

  it("should treat null and absent policies as an unchanged pause-until-reset policy", () => {
    const original = {
      ...organization,
      default_budgets: organization.default_budgets.map((group) => ({ ...group, cap_policy: null })),
    };
    expect(planCustomerBudgetGroups(original, customerBudgetGroupRows(original), "standard")).toMatchObject({
      ok: true,
      updates: [],
      changed: false,
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
    [
      [{ id: "a", name: "", monthlyBudget: "1", yearlyBudget: "", capPolicy: defaultCustomerCapPolicy }],
      "a",
      "groupNameRequired",
    ],
    [
      [{ id: "a", name: "A", monthlyBudget: "", yearlyBudget: "", capPolicy: defaultCustomerCapPolicy }],
      "a",
      "groupLimitRequired",
    ],
    [
      [{ id: "a", name: "A", monthlyBudget: "-1", yearlyBudget: "", capPolicy: defaultCustomerCapPolicy }],
      "a",
      "invalidBudget",
    ],
  ] as const)("should validate invalid edits", (groups, defaultId, error) => {
    expect(planCustomerBudgetGroups(organization, [...groups], defaultId)).toEqual({ ok: false, error });
  });
});
