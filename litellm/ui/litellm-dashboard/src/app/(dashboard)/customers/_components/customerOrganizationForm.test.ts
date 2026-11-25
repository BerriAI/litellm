import { describe, expect, it } from "vitest";
import {
  buildCustomerOrganizationBody,
  defaultCustomerCapPolicy,
  initialCustomerOrganizationFields,
  type CustomerOrganizationFields,
} from "./customerOrganizationForm";
import { budgetEditorFields } from "./customerBudgetForm";

const fields = () => ({
  ...initialCustomerOrganizationFields(),
  budgetMode: "custom" as const,
  name: " Example customer ",
  customerOrgId: " customer-001 ",
});

describe("buildCustomerOrganizationBody", () => {
  it.each(["5", ""])("keeps custom template windows when creating calendar cap %s", (amount) => {
    const input = fields();
    const custom = { budget_duration: "7d", max_budget: 3, reset_at: "2027-01-04T00:00:00Z" };
    expect(
      buildCustomerOrganizationBody({
        ...input,
        defaultGroupId: "group",
        groups: [
          {
            ...budgetEditorFields({
              budget_limits: [...(amount ? [{ budget_duration: "1mo", max_budget: Number(amount) }] : []), custom],
            }),
            id: "group",
            name: "Group",
            capPolicy: defaultCustomerCapPolicy,
          },
        ],
      }),
    ).toMatchObject({
      body: {
        default_budgets: [{ budget_limits: [...(amount ? [{ budget_duration: "1mo", max_budget: 5 }] : []), custom] }],
      },
    });
  });

  it("leaves the server default in control until limits are explicitly selected", () => {
    const input = {
      ...fields(),
      budgetMode: "inherit",
      budgetId: "stale-profile",
      budgetAmount: "-1",
      tpm: "invalid",
    } satisfies CustomerOrganizationFields;
    const result = buildCustomerOrganizationBody(input);
    expect(result.ok).toBe(true);
    if (!result.ok) return;
    expect(result.body).not.toHaveProperty("budget_id");
    expect(result.body).not.toHaveProperty("budget");
  });

  it("creates model-only member groups with trimmed deduplicated restrictions", () => {
    const group = {
      ...fields().groups[1],
      maxBudget: "",
      allowedModels: [" model-a ", "model-a", " "],
    };
    expect(buildCustomerOrganizationBody({ ...fields(), groups: [group] })).toMatchObject({
      ok: true,
      body: {
        default_budgets: [{ name: group.name, cap_policy: group.capPolicy, allowed_models: ["model-a"] }],
      },
    });
  });

  it("creates the organization, overall calendar budget, and independent member caps together", () => {
    const input = {
      ...fields(),
      budgetAmount: "1000.25",
      tpm: "3000",
      rpm: "25",
      groups: [
        {
          ...budgetEditorFields({
            max_budget: 200.5,
            budget_duration: "1mo",
            budget_limits: [{ budget_duration: "12mo", max_budget: 1500 }],
          }),
          id: "custom",
          name: " power ",
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
            max_budget: 200.5,
            budget_duration: "1mo",
            budget_limits: [{ budget_duration: "12mo", max_budget: 1500 }],
          },
        ],
        default_budget_template: "power",
      },
    });
  });

  it.each(["", "1mo", "12mo", "7d"])(
    "opts an unlimited custom budget out of the default despite period %j",
    (budgetDuration) => {
      const result = buildCustomerOrganizationBody({ ...fields(), customerOrgId: "  ", budgetDuration });
      expect(result.ok).toBe(true);
      if (!result.ok) return;
      expect(result.body).not.toHaveProperty("customer_org_id");
      expect(result.body).not.toHaveProperty("budget");
      expect(result.body.budget_id).toBeNull();
      expect(result.body.default_budget_template).toBe("default_user");
    },
  );

  it("uses the profile without sending or validating inactive custom limits", () => {
    const input = {
      ...fields(),
      budgetMode: "profile",
      budgetId: "saved-profile",
      budgetAmount: "-1",
      tpm: "invalid",
      rpm: "1.5",
    } satisfies CustomerOrganizationFields;
    const result = buildCustomerOrganizationBody(input);
    expect(result.ok).toBe(true);
    if (!result.ok) return;
    expect(result.body.budget_id).toBe("saved-profile");
    expect(result.body).not.toHaveProperty("budget");
  });

  it("requires a profile selection only in profile mode", () => {
    expect(buildCustomerOrganizationBody({ ...fields(), budgetMode: "profile" })).toEqual({
      ok: false,
      error: "profileRequired",
    });
    expect(buildCustomerOrganizationBody(fields()).ok).toBe(true);
  });

  describe.each([
    ["", "0"],
    ["", "50"],
    ["0", "0"],
  ])("spend %j and soft budget %j", (budgetAmount, softBudget) => {
    const hasHardLimit = budgetAmount !== "";
    it.each(["", "1mo", "12mo", "7d"])("creates organization and group limits with period %j", (budgetDuration) => {
      const limits = {
        budgetDuration,
        softBudget,
        ...(hasHardLimit ? { allowedModels: [" model-a ", "model-a"], tpm: "0", rpm: "0" } : {}),
      };
      const group = { ...fields().groups[1], ...limits, maxBudget: budgetAmount };
      const input = { ...fields(), ...limits, budgetAmount, budgetId: "saved-profile", groups: [group] };
      const result = buildCustomerOrganizationBody(input);
      expect(result.ok).toBe(true);
      if (!result.ok) return;
      const expected = {
        soft_budget: Number(softBudget),
        ...(hasHardLimit ? { max_budget: 0, allowed_models: ["model-a"], tpm_limit: 0, rpm_limit: 0 } : {}),
        ...(budgetDuration || hasHardLimit ? { budget_duration: budgetDuration || null } : {}),
      };
      expect(result.body.budget).toEqual(expected);
      expect(result.body.default_budgets).toEqual([{ name: group.name, cap_policy: group.capPolicy, ...expected }]);
      expect(result.body).not.toHaveProperty("budget_id");
    });
  });

  it("keeps zero caps and permits either calendar limit to be omitted", () => {
    const input = {
      ...fields(),
      budgetAmount: "0",
      groups: [
        {
          ...budgetEditorFields({ max_budget: 0, budget_duration: "1mo" }),
          id: "monthly",
          name: "monthly",
          capPolicy: defaultCustomerCapPolicy,
        },
        {
          ...budgetEditorFields({ max_budget: 120, budget_duration: "12mo" }),
          id: "yearly",
          name: "yearly",
          capPolicy: defaultCustomerCapPolicy,
        },
      ],
      defaultGroupId: "yearly",
    };
    const result = buildCustomerOrganizationBody(input);
    expect(result.ok).toBe(true);
    if (!result.ok) return;
    expect(result.body.budget?.max_budget).toBe(0);
    expect(
      result.body.default_budgets.map(({ budget_duration, max_budget }) => ({ budget_duration, max_budget })),
    ).toEqual([
      { budget_duration: "1mo", max_budget: 0 },
      { budget_duration: "12mo", max_budget: 120 },
    ]);
    expect(result.body.default_budget_template).toBe("yearly");
  });

  it.each(["-1", "Infinity", "NaN", "invalid"])("rejects invalid amounts %s at either level", (value) => {
    const input = fields();
    expect(buildCustomerOrganizationBody({ ...input, budgetAmount: value })).toEqual({
      ok: false,
      error: "invalidBudget",
    });
    for (const field of ["maxBudget", "softBudget"] as const) {
      expect(
        buildCustomerOrganizationBody({
          ...input,
          groups: input.groups.map((group) => ({ ...group, [field]: value })),
        }),
      ).toEqual({ ok: false, error: "invalidBudget" });
    }
  });

  it("rejects malformed custom periods before creating an organization", () => {
    expect(buildCustomerOrganizationBody({ ...fields(), budgetDuration: "6days" })).toEqual({
      ok: false,
      error: "invalidBudget",
    });
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
        groups: input.groups.map((group) => ({ ...group, maxBudget: " " })),
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

  it("creates organization and member rate limits without dollar budgets and preserves zero", () => {
    const input = fields();
    const result = buildCustomerOrganizationBody({
      ...input,
      rpm: "120",
      groups: input.groups.map((group) => ({ ...group, maxBudget: "", tpm: "50000", rpm: "0" })),
    });
    expect(result.ok).toBe(true);
    if (!result.ok) return;
    expect(result.body.budget).toEqual({ rpm_limit: 120 });
    expect(result.body.default_budgets).toEqual(
      input.groups.map((group) => ({
        name: group.name,
        cap_policy: group.capPolicy,
        tpm_limit: 50000,
        rpm_limit: 0,
      })),
    );
  });
});
