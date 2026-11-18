import { describe, expect, it } from "vitest";
import {
  buildCustomerOrganizationBody,
  defaultCustomerCapPolicy,
  initialCustomerOrganizationFields,
} from "./customerOrganizationForm";

const fields = () => ({
  ...initialCustomerOrganizationFields(),
  name: " Example customer ",
  customerOrgId: " customer-001 ",
});

describe("buildCustomerOrganizationBody", () => {
  it("creates the organization, overall calendar budget, and independent member caps together", () => {
    const input = {
      ...fields(),
      monthlyBudget: "1000.25",
      tpm: "3000",
      rpm: "25",
      groups: [
        {
          id: "custom",
          name: " power ",
          monthlyBudget: "200.50",
          yearlyBudget: "1500",
          capPolicy: defaultCustomerCapPolicy,
        },
      ],
      defaultGroupId: "custom",
    };
    const result = buildCustomerOrganizationBody(input);
    expect(result).toEqual({
      ok: true,
      body: {
        customer_org_id: "customer-001",
        alias: "Example customer",
        budget: { max_budget: 1000.25, budget_duration: "1mo", tpm_limit: 3000, rpm_limit: 25 },
        default_budgets: [
          {
            name: "power",
            cap_policy: "pause_until_reset",
            budget_limits: [
              { budget_duration: "1mo", max_budget: 200.5 },
              { budget_duration: "12mo", max_budget: 1500 },
            ],
          },
        ],
        default_budget_template: "power",
      },
    });
  });

  it("allows the server to generate an ID and omits an unlimited organization budget", () => {
    const result = buildCustomerOrganizationBody({ ...fields(), customerOrgId: "  " });
    expect(result.ok).toBe(true);
    if (!result.ok) return;
    expect(result.body).not.toHaveProperty("customer_org_id");
    expect(result.body).not.toHaveProperty("budget");
    expect(result.body.default_budget_template).toBe("default_user");
  });

  it("keeps zero caps and permits either calendar limit to be omitted", () => {
    const input = {
      ...fields(),
      monthlyBudget: "0",
      groups: [
        { id: "monthly", name: "monthly", monthlyBudget: "0", yearlyBudget: "", capPolicy: defaultCustomerCapPolicy },
        { id: "yearly", name: "yearly", monthlyBudget: "", yearlyBudget: "120", capPolicy: defaultCustomerCapPolicy },
      ],
      defaultGroupId: "yearly",
    };
    const result = buildCustomerOrganizationBody(input);
    expect(result.ok).toBe(true);
    if (!result.ok) return;
    expect(result.body.budget?.max_budget).toBe(0);
    expect(result.body.default_budgets.map((group) => group.budget_limits)).toEqual([
      [{ budget_duration: "1mo", max_budget: 0 }],
      [{ budget_duration: "12mo", max_budget: 120 }],
    ]);
    expect(result.body.default_budget_template).toBe("yearly");
  });

  it.each(["-1", "Infinity", "NaN", "invalid"])("rejects invalid amounts %s at either level", (value) => {
    const input = fields();
    expect(buildCustomerOrganizationBody({ ...input, monthlyBudget: value })).toEqual({
      ok: false,
      error: "invalidBudget",
    });
    for (const field of ["monthlyBudget", "yearlyBudget"] as const) {
      expect(
        buildCustomerOrganizationBody({
          ...input,
          groups: input.groups.map((group) => ({ ...group, [field]: value })),
        }),
      ).toEqual({ ok: false, error: "invalidBudget" });
    }
  });

  it.each(["-1", "1.5", "Infinity", "invalid", "9007199254740992"])("rejects invalid rate limits %s", (value) => {
    for (const field of ["tpm", "rpm"] as const) {
      expect(buildCustomerOrganizationBody({ ...fields(), [field]: value })).toEqual({
        ok: false,
        error: "invalidRateLimit",
      });
    }
  });

  it("rejects missing names, duplicate trimmed group names, and groups without any cap", () => {
    const input = fields();
    expect(buildCustomerOrganizationBody({ ...input, name: " " })).toEqual({ ok: false, error: "nameRequired" });
    expect(
      buildCustomerOrganizationBody({ ...input, groups: input.groups.map((group) => ({ ...group, name: " " })) }),
    ).toEqual({ ok: false, error: "groupNameRequired" });
    expect(
      buildCustomerOrganizationBody({
        ...input,
        groups: input.groups.map((group) => ({ ...group, name: " repeated " })),
      }),
    ).toEqual({ ok: false, error: "duplicateGroups" });
    expect(
      buildCustomerOrganizationBody({
        ...input,
        groups: input.groups.map((group) => ({ ...group, monthlyBudget: " ", yearlyBudget: "" })),
      }),
    ).toEqual({ ok: false, error: "groupLimitRequired" });
  });

  it("requires a default belonging to the submitted groups", () => {
    expect(buildCustomerOrganizationBody({ ...fields(), groups: [] })).toEqual({ ok: false, error: "defaultRequired" });
    expect(buildCustomerOrganizationBody({ ...fields(), defaultGroupId: "missing" })).toEqual({
      ok: false,
      error: "defaultRequired",
    });
  });
});
