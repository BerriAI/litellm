import { describe, expect, it } from "vitest";
import {
  buildCustomerHierarchyPatch as buildPatch,
  customerHierarchyFields as initialFields,
  type CustomerHierarchyEntity,
  type Organization,
  type Member,
  buildCustomerHierarchyCreate,
  newCustomerHierarchyFields,
  type CustomerHierarchyFields,
  type CustomerHierarchyCreateTarget,
} from "./customerHierarchyForms";

const audit = {
  created_at: "2026-11-01T00:00:00Z",
  updated_at: "2026-11-01T00:00:00Z",
  created_by: "admin",
  updated_by: "admin",
  spend: 5,
};
const organization: Organization = {
  budget_groups_revision: "a".repeat(64),
  member_count: 1,
  ...audit,
  customer_org_id: "org-1",
  alias: "Example",
  default_budgets: [],
  budget: {
    max_budget: 100,
    soft_budget: 80,
    budget_duration: "1mo",
    tpm_limit: 200,
    rpm_limit: 10,
    allowed_models: ["openai/gpt-4.1", "anthropic/claude-sonnet"],
  },
};
const memberBase = { max_budget: 500, budget_duration: "1mo", rpm_limit: 10, tpm_limit: 5000 };
const memberOrganization = {
  ...organization,
  default_budgets: [{ budget_id: "template-1", name: "Standard", ...memberBase }],
};
const member: Member = {
  ...audit,
  customer_member_id: "member-1",
  parent_org_id: organization.customer_org_id,
  parent_team_ids: ["team-1"],
  alias: "Member One",
  budget_id: "template-1",
  budget: memberBase,
  budget_override_id: "override-1",
  budget_override: { max_budget: 20, rpm_limit: 4 },
  effective_budget: { ...memberBase, max_budget: 1000, rpm_limit: 40 },
  budget_override_expires_at: "2026-12-01T12:00:00Z",
};

const customerHierarchyFields = (entity: CustomerHierarchyEntity, org = memberOrganization) =>
  initialFields(entity, org);
const buildCustomerHierarchyPatch = (entity: CustomerHierarchyEntity, fields: CustomerHierarchyFields) =>
  buildPatch(entity, fields, memberOrganization);

describe("customer hierarchy settings payloads", () => {
  it.each(["team", "member"] as const)("removes or replaces a shown custom window in the %s editor", (kind) => {
    const custom = { budget_duration: "7d", max_budget: 3, reset_at: "2027-01-04T00:00:00Z" };
    const budget = { budget_limits: [custom] };
    const entities: Record<typeof kind, CustomerHierarchyEntity> = {
      team: { kind: "team", record: { ...audit, customer_team_id: "team", parent_org_id: "org-1", budget } },
      member: {
        kind: "member",
        record: { ...member, budget_id: "private", budget, budget_override: null, budget_override_expires_at: null },
      },
    };
    const entity = entities[kind];
    const fields = customerHierarchyFields(entity);
    expect(buildCustomerHierarchyPatch(entity, { ...fields, budgetWindows: [] })).toMatchObject({
      patch: { body: { budget: { budget_limits: null } } },
    });
    expect(
      buildCustomerHierarchyPatch(entity, {
        ...fields,
        budgetWindows: [{ budget: custom, maxBudget: "" }],
      }),
    ).toMatchObject({ patch: { body: { budget: { budget_limits: null } } } });
    expect(
      buildCustomerHierarchyPatch(entity, {
        ...fields,
        budgetWindows: [{ budget: { budget_duration: "2d", max_budget: 3 }, maxBudget: "3" }],
      }),
    ).toMatchObject({ patch: { body: { budget: { budget_limits: [{ budget_duration: "2d", max_budget: 3 }] } } } });
  });

  it.each(["organization", "team", "member"] as const)(
    "preserves custom caps while editing or removing calendar caps in the %s editor",
    (kind) => {
      const custom = { budget_duration: "7d", max_budget: 3, reset_at: "2027-01-04T00:00:00Z" };
      const monthly = { budget_duration: "1mo", max_budget: 50, reset_at: "2027-02-01T00:00:00Z" };
      const yearly = { budget_duration: "12mo", max_budget: 500 };
      const budget = { max_budget: null, budget_duration: null, budget_limits: [monthly, yearly, custom] };
      const entities: Record<typeof kind, CustomerHierarchyEntity> = {
        organization: { kind: "organization", record: { ...organization, budget } },
        team: { kind: "team", record: { ...audit, customer_team_id: "team", parent_org_id: "org-1", budget } },
        member: {
          kind: "member",
          record: { ...member, budget_id: "private", budget, budget_override: null, budget_override_expires_at: null },
        },
      };
      const entity = entities[kind];
      const fields = customerHierarchyFields(entity);
      expect(
        buildCustomerHierarchyPatch(entity, {
          ...fields,
          budgetWindows: fields.budgetWindows.map((window) => ({
            ...window,
            maxBudget: window.budget.budget_duration === "1mo" ? "0" : window.maxBudget,
          })),
        }),
      ).toMatchObject({
        patch: { body: { budget: { budget_limits: [{ ...monthly, max_budget: 0 }, yearly, custom] } } },
      });
      expect(
        buildCustomerHierarchyPatch(entity, {
          ...fields,
          budgetWindows: fields.budgetWindows.filter((window) => window.budget.budget_duration !== "1mo"),
        }),
      ).toMatchObject({
        patch: { body: { budget: { budget_limits: [yearly, custom] } } },
      });
    },
  );

  it("should load the member template from the API response", () => {
    expect(customerHierarchyFields({ kind: "member", record: member }, memberOrganization).templateId).toBe(
      member.budget_id,
    );
  });

  it("should allow an alias-only edit for a member using a template without an override", () => {
    const entity: CustomerHierarchyEntity = {
      kind: "member",
      record: { ...member, budget_override: null, budget_override_expires_at: null },
    };
    expect(buildCustomerHierarchyPatch(entity, { ...customerHierarchyFields(entity), alias: "Updated" })).toEqual({
      ok: true,
      patch: { kind: "member", body: { alias: "Updated" } },
    });
  });

  it("switches a preset without changing an active temporary grant", () => {
    const entity: CustomerHierarchyEntity = { kind: "member", record: member };
    expect(
      buildCustomerHierarchyPatch(entity, { ...customerHierarchyFields(entity), templateId: "template-2" }),
    ).toEqual({
      ok: true,
      patch: { kind: "member", body: { budget_id: "template-2" } },
    });
  });

  it("should leave untouched settings out of the patch", () => {
    const entity: CustomerHierarchyEntity = { kind: "organization", record: organization };
    expect(buildCustomerHierarchyPatch(entity, customerHierarchyFields(entity))).toEqual({
      ok: true,
      patch: { kind: "organization", body: {} },
    });
  });

  it("should update only the alias without overwriting the existing budget", () => {
    const entity: CustomerHierarchyEntity = { kind: "organization", record: organization };
    expect(buildCustomerHierarchyPatch(entity, { ...customerHierarchyFields(entity), alias: " Updated " })).toEqual({
      ok: true,
      patch: { kind: "organization", body: { alias: "Updated" } },
    });
  });

  it("should preserve the other limits when changing a spend limit to zero", () => {
    const entity: CustomerHierarchyEntity = { kind: "organization", record: organization };
    expect(buildCustomerHierarchyPatch(entity, { ...customerHierarchyFields(entity), maxBudget: "0" })).toEqual({
      ok: true,
      patch: { kind: "organization", body: { budget: { max_budget: 0, budget_duration: "1mo" } } },
    });
  });

  it.each([null, 100])("edits a monthly window while retaining the yearly and scalar %s limits", (max_budget) => {
    const windows = [
      { budget_duration: "1mo" as const, max_budget: 50 },
      { budget_duration: "12mo" as const, max_budget: 500 },
    ];
    const entity: CustomerHierarchyEntity = {
      kind: "organization",
      record: {
        ...organization,
        budget_id: "custom",
        budget: { ...organization.budget, max_budget, budget_duration: null, budget_limits: windows },
      },
    };
    const fields = customerHierarchyFields(entity);
    expect(fields.maxBudget).toBe(max_budget == null ? "" : String(max_budget));
    expect(
      buildCustomerHierarchyPatch(entity, {
        ...fields,
        budgetWindows: fields.budgetWindows.map((window, index) =>
          index === 0 ? { ...window, maxBudget: "0" } : window,
        ),
      }),
    ).toEqual({
      ok: true,
      patch: {
        kind: "organization",
        body: {
          budget: {
            budget_limits: [{ budget_duration: "1mo", max_budget: 0 }, windows[1]],
          },
        },
      },
    });
  });

  it("should clear an explicit budget when disabled", () => {
    const entity: CustomerHierarchyEntity = {
      kind: "team",
      record: {
        ...audit,
        customer_team_id: "team-1",
        parent_org_id: "org-1",
        budget: { max_budget: 100 },
        effective_budget: { max_budget: 100 },
      },
    };
    expect(buildCustomerHierarchyPatch(entity, { ...customerHierarchyFields(entity), budgetEnabled: false })).toEqual({
      ok: true,
      patch: { kind: "team", body: { budget: null } },
    });
  });

  it("selects a preset from a custom base without clearing temporary grants", () => {
    const entity: CustomerHierarchyEntity = { kind: "member", record: { ...member, budget_id: "owned" } };
    expect(
      buildCustomerHierarchyPatch(entity, {
        ...customerHierarchyFields(entity),
        budgetEnabled: false,
        templateId: "template-1",
      }),
    ).toEqual({
      ok: true,
      patch: { kind: "member", body: { budget_id: "template-1" } },
    });
  });

  it("shows the base cap and rates instead of a temporary increase", () => {
    const entity: CustomerHierarchyEntity = { kind: "member", record: member };
    const expected = {
      templateId: "template-1",
      budgetEnabled: false,
      maxBudget: "500",
      rpm: "10",
      tpm: "5000",
    };
    expect(customerHierarchyFields(entity)).toMatchObject(expected);
  });

  it("should require a template or an override for a member", () => {
    const entity: CustomerHierarchyEntity = { kind: "member", record: member };
    expect(
      buildCustomerHierarchyPatch(entity, { ...customerHierarchyFields(entity), budgetEnabled: false, templateId: "" }),
    ).toEqual({ ok: false, error: "budgetRequired" });
  });

  it.each(["-1", "NaN", "Infinity", "1e999", "not a number"])(
    "should reject an invalid spend limit %s",
    (maxBudget) => {
      const entity: CustomerHierarchyEntity = { kind: "organization", record: organization };
      expect(buildCustomerHierarchyPatch(entity, { ...customerHierarchyFields(entity), maxBudget })).toEqual({
        ok: false,
        error: "invalidNumber",
      });
    },
  );

  it.each(["1.5", "9007199254740992"])("should reject a non-safe-integer rate limit %s", (rpm) => {
    const entity: CustomerHierarchyEntity = { kind: "organization", record: organization };
    expect(buildCustomerHierarchyPatch(entity, { ...customerHierarchyFields(entity), rpm })).toEqual({
      ok: false,
      error: "invalidNumber",
    });
  });

  it("should clear a blank limit and allow a sub-cent spend limit", () => {
    const entity: CustomerHierarchyEntity = { kind: "organization", record: organization };
    expect(
      buildCustomerHierarchyPatch(entity, { ...customerHierarchyFields(entity), maxBudget: "0.00001", softBudget: "" }),
    ).toEqual({
      ok: true,
      patch: {
        kind: "organization",
        body: { budget: { max_budget: 0.00001, budget_duration: "1mo", soft_budget: null } },
      },
    });
  });

  it("should retain an unchanged expiry exactly when editing the member name", () => {
    const entity: CustomerHierarchyEntity = { kind: "member", record: member };
    expect(buildCustomerHierarchyPatch(entity, { ...customerHierarchyFields(entity), alias: "New name" })).toEqual({
      ok: true,
      patch: { kind: "member", body: { alias: "New name" } },
    });
  });

  it("replaces a preset cap with a permanent custom budget while retaining its other base limits", () => {
    const entity: CustomerHierarchyEntity = { kind: "member", record: member };
    const fields = { ...customerHierarchyFields(entity), budgetEnabled: true, maxBudget: "750" };
    expect(buildCustomerHierarchyPatch(entity, fields)).toEqual({
      ok: true,
      patch: {
        kind: "member",
        body: {
          budget: { max_budget: 750, budget_duration: "1mo" },
        },
      },
    });
  });

  it("converts legacy non-expiring custom settings into a base budget", () => {
    const entity: CustomerHierarchyEntity = {
      kind: "member",
      record: {
        ...member,
        budget: { ...memberBase, budget_limits: [{ budget_duration: "12mo", max_budget: 5000 }] },
        budget_override: { max_budget: 750 },
        budget_override_expires_at: null,
        effective_budget: { ...memberBase, max_budget: 1000 },
      },
    };
    const fields = customerHierarchyFields(entity);
    expect(fields).toMatchObject({ budgetEnabled: true, maxBudget: "750", rpm: "10" });
    expect(buildCustomerHierarchyPatch(entity, { ...fields, maxBudget: "800" })).toMatchObject({
      patch: { body: { budget: { ...memberBase, max_budget: 800, budget_limits: null }, budget_override: null } },
    });
    expect(buildCustomerHierarchyPatch(entity, { ...fields, budgetEnabled: false })).toEqual({
      ok: true,
      patch: { kind: "member", body: { budget_id: "template-1", budget_override: null } },
    });
    expect(buildCustomerHierarchyPatch(entity, { ...fields, alias: "Renamed" })).toEqual({
      ok: true,
      patch: { kind: "member", body: { alias: "Renamed" } },
    });
  });

  it("clears visible legacy limits while preserving hidden policy during permanent conversion", () => {
    const yearly = { budget_duration: "12mo" as const, max_budget: 5000, reset_at: "2027-01-01T00:00:00Z" };
    const entity: CustomerHierarchyEntity = {
      kind: "member",
      record: {
        ...member,
        budget: { ...memberBase, model_tpm_limit: { "test-model": 1000 }, cap_policy: "pause_allow_increase" },
        budget_override: { max_budget: 750, budget_limits: [yearly] },
        budget_override_expires_at: null,
      },
    };
    const fields = customerHierarchyFields(entity);
    expect(buildCustomerHierarchyPatch(entity, { ...fields, maxBudget: "", rpm: "" })).toMatchObject({
      patch: {
        body: {
          budget: {
            max_budget: null,
            rpm_limit: null,
            tpm_limit: memberBase.tpm_limit,
            budget_limits: [yearly],
            model_tpm_limit: { "test-model": 1000 },
            cap_policy: "pause_allow_increase",
          },
          budget_override: null,
        },
      },
    });
  });
});

it("rejects mixing a new preset with custom limits, including zero", () => {
  const entity: CustomerHierarchyEntity = { kind: "member", record: member };
  const fields = { ...customerHierarchyFields(entity), templateId: "template-2", budgetEnabled: true, maxBudget: "0" };
  expect(buildCustomerHierarchyPatch(entity, fields)).toEqual({ ok: false, error: "conflictingBudgetSources" });
});

it("requires a preset when disabling a private member budget", () => {
  const entity: CustomerHierarchyEntity = { kind: "member", record: { ...member, budget_id: "owned" } };
  const fields = customerHierarchyFields(entity);
  expect(fields.templateId).toBe("");
  expect(buildCustomerHierarchyPatch(entity, { ...fields, budgetEnabled: false })).toEqual({
    ok: false,
    error: "budgetRequired",
  });
});

it("preserves explicit clears and zero when first customizing a preset with a temporary grant", () => {
  const entity: CustomerHierarchyEntity = { kind: "member", record: member };
  const fields = { ...customerHierarchyFields(entity), budgetEnabled: true, maxBudget: "0", rpm: "" };
  expect(buildCustomerHierarchyPatch(entity, fields)).toEqual({
    ok: true,
    patch: { kind: "member", body: { budget: { max_budget: 0, budget_duration: "1mo", rpm_limit: null } } },
  });
});

describe("customer hierarchy creation payloads", () => {
  const teamTarget: CustomerHierarchyCreateTarget = { kind: "team", organization };
  const organizationWithTemplate: Organization = {
    ...organization,
    default_budget_template_id: "standard-template",
    default_budgets: [{ budget_id: "standard-template", name: "Standard", max_budget: 50 }],
  };
  const memberTarget: CustomerHierarchyCreateTarget = {
    kind: "member",
    organization: organizationWithTemplate,
    team: { ...audit, customer_team_id: "team-1", parent_org_id: organization.customer_org_id, effective_budget: {} },
  };

  it("should create a team under its organization with an automatic identifier", () => {
    const fields = { ...newCustomerHierarchyFields(teamTarget), alias: " Support " };
    expect(buildCustomerHierarchyCreate(teamTarget, fields, "")).toEqual({
      ok: true,
      create: { kind: "team", body: { alias: "Support", parent_org_id: "org-1", budget: null } },
    });
  });

  it("should preserve an explicit zero budget and custom team identifier", () => {
    const fields = { ...newCustomerHierarchyFields(teamTarget), budgetEnabled: true, maxBudget: "0" };
    const result = buildCustomerHierarchyCreate(teamTarget, fields, " support-team ");
    expect(result).toMatchObject({
      ok: true,
      create: { kind: "team", body: { customer_team_id: "support-team", budget: { max_budget: 0 } } },
    });
  });

  it("should preselect the organization default template for a new member", () => {
    const fields = newCustomerHierarchyFields(memberTarget);
    expect(fields.templateId).toBe("standard-template");
    expect(fields.budgetEnabled).toBe(false);
    expect(fields.maxBudget).toBe("");
    expect(fields.allowedModels).toEqual([]);
    expect(buildCustomerHierarchyCreate(memberTarget, fields, "")).toEqual({
      ok: true,
      create: {
        kind: "member",
        body: {
          parent_org_id: "org-1",
          parent_team_ids: ["team-1"],
          alias: null,
          budget_id: "standard-template",
        },
      },
    });
  });

  it("creates a custom base when the organization has no templates", () => {
    const target: CustomerHierarchyCreateTarget = { ...memberTarget, organization };
    const fields = newCustomerHierarchyFields(target);
    expect(fields.budgetEnabled).toBe(true);
    expect(
      buildCustomerHierarchyCreate(target, { ...fields, maxBudget: "0", allowedModels: ["gpt-4.1"] }, "new-member"),
    ).toMatchObject({
      ok: true,
      create: {
        kind: "member",
        body: {
          customer_member_id: "new-member",
          budget_id: null,
          budget: { max_budget: 0, allowed_models: ["gpt-4.1"] },
        },
      },
    });
  });

  it("creates organization members in the canonical default team", () => {
    const target: CustomerHierarchyCreateTarget = { kind: "member", organization: organizationWithTemplate };
    expect(buildCustomerHierarchyCreate(target, newCustomerHierarchyFields(target), "new-member")).toMatchObject({
      ok: true,
      create: {
        defaultTeamOrganizationId: "org-1",
        body: { parent_org_id: "org-1", parent_team_ids: ["org-1:default"], customer_member_id: "new-member" },
      },
    });
  });

  it("creates rate-only custom limits without inheriting a preset cap", () => {
    const fields = { ...newCustomerHierarchyFields(memberTarget), budgetEnabled: true, tpm: "20000" };
    expect(buildCustomerHierarchyCreate(memberTarget, fields, "")).toMatchObject({
      ok: true,
      create: {
        body: {
          budget_id: null,
          budget: { max_budget: null, budget_duration: "1mo", tpm_limit: 20000, rpm_limit: null },
        },
      },
    });
  });

  it("should reject a member without either budget source", () => {
    const target: CustomerHierarchyCreateTarget = { ...memberTarget, organization };
    expect(
      buildCustomerHierarchyCreate(target, { ...newCustomerHierarchyFields(target), budgetEnabled: false }, ""),
    ).toEqual({ ok: false, error: "budgetRequired" });
  });

  it("should reject templates from another organization", () => {
    expect(
      buildCustomerHierarchyCreate(
        memberTarget,
        { ...newCustomerHierarchyFields(memberTarget), templateId: "foreign-template" },
        "",
      ),
    ).toEqual({ ok: false, error: "invalidTemplate" });
  });

  it("should reject invalid numeric limits before creating a record", () => {
    expect(
      buildCustomerHierarchyCreate(
        teamTarget,
        { ...newCustomerHierarchyFields(teamTarget), budgetEnabled: true, maxBudget: "-1" },
        "",
      ),
    ).toEqual({ ok: false, error: "invalidNumber" });
  });

  it.each(["team,another", "support/eu", ".", ".."])(
    "should reject a custom identifier that cannot be addressed: %s",
    (id) => {
      expect(buildCustomerHierarchyCreate(teamTarget, newCustomerHierarchyFields(teamTarget), id)).toEqual({
        ok: false,
        error: "invalidIdentifier",
      });
    },
  );
});
