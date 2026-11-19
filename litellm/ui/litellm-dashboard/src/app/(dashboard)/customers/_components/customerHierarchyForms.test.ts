import { describe, expect, it } from "vitest";
import {
  buildCustomerHierarchyPatch,
  customerHierarchyFields,
  type CustomerHierarchyEntity,
  type Organization,
  type Member,
  buildCustomerHierarchyCreate,
  newCustomerHierarchyFields,
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
const member: Member = {
  ...audit,
  customer_member_id: "member-1",
  parent_team_ids: ["team-1"],
  alias: "Member One",
  budget_id: "template-1",
  budget_override_id: "override-1",
  budget_override: { max_budget: 20, rpm_limit: 4 },
  effective_budget: { max_budget: 20, rpm_limit: 4 },
  budget_override_expires_at: "2026-12-01T12:00:00Z",
};

describe("customer hierarchy settings payloads", () => {
  it("should load the member template from the API response", () => {
    expect(customerHierarchyFields({ kind: "member", record: member }).templateId).toBe(member.budget_id);
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

  it.each([
    ["template-2", "template-2"],
    ["", null],
    ["template-2", "template-2", true],
  ] as const)("submits template %s with budget %s and legacy override %s", (templateId, budgetId, legacy) => {
    const entity: CustomerHierarchyEntity = {
      kind: "member",
      record: { ...member, budget_override_expires_at: legacy ? null : member.budget_override_expires_at },
    };
    expect(buildCustomerHierarchyPatch(entity, { ...customerHierarchyFields(entity), templateId })).toEqual({
      ok: true,
      patch: { kind: "member", body: { budget_id: budgetId, ...(legacy ? { budget_override: null } : {}) } },
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
      patch: { kind: "organization", body: { budget: { ...organization.budget, max_budget: 0 } } },
    });
  });

  it.each([null, 100])("edits a monthly window while retaining the yearly and scalar %s limits", (max_budget) => {
    const windows = [
      { budget_duration: "1mo", max_budget: 50 },
      { budget_duration: "12mo", max_budget: 500 },
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
    expect(fields.maxBudget).toBe("50");
    expect(buildCustomerHierarchyPatch(entity, { ...fields, maxBudget: "0" })).toMatchObject({
      patch: {
        body: {
          budget: {
            max_budget,
            budget_duration: null,
            budget_limits: [{ budget_duration: "1mo", max_budget: 0 }, windows[1]],
          },
        },
      },
    });
  });

  it("should clear an explicit budget when disabled", () => {
    const entity: CustomerHierarchyEntity = {
      kind: "team",
      record: { ...audit, customer_team_id: "team-1", parent_org_id: "org-1", budget: { max_budget: 100 } },
    };
    expect(buildCustomerHierarchyPatch(entity, { ...customerHierarchyFields(entity), budgetEnabled: false })).toEqual({
      ok: true,
      patch: { kind: "team", body: { budget: null } },
    });
  });

  it("should clear a member override and expiry while keeping its template", () => {
    const entity: CustomerHierarchyEntity = { kind: "member", record: member };
    expect(buildCustomerHierarchyPatch(entity, { ...customerHierarchyFields(entity), budgetEnabled: false })).toEqual({
      ok: true,
      patch: { kind: "member", body: { budget_override: null, budget_override_expires_at: null } },
    });
  });

  it.each([
    { changes: {}, override: {} },
    { changes: { budgetEnabled: false }, override: { budget_override: null, budget_override_expires_at: null } },
    { changes: { overrideExpiry: "" }, override: { budget_override: null, budget_override_expires_at: null } },
    {
      changes: { maxBudget: "90", overrideExpiry: "2026-12-15T12:00" },
      override: {
        budget_override: expect.objectContaining({ max_budget: 90, rpm_limit: 4 }),
        budget_override_expires_at: new Date("2026-12-15T12:00").toISOString(),
      },
    },
  ])("saves a member preset selection with independent changes $changes", ({ changes, override }) => {
    const entity: CustomerHierarchyEntity = { kind: "member", record: member };
    const fields = customerHierarchyFields(entity);
    expect(fields.templateId).toBe("template-1");
    expect(fields.maxBudget).toBe("20");
    expect(buildCustomerHierarchyPatch(entity, { ...fields, templateId: "template-2", ...changes })).toEqual({
      ok: true,
      patch: { kind: "member", body: { budget_id: "template-2", ...override } },
    });
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
        body: { budget: { ...organization.budget, max_budget: 0.00001, soft_budget: null } },
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

  it.each([50, 0])("omits untouched inherited limits on the first permanent custom edit %s", (limit) => {
    const entity: CustomerHierarchyEntity = {
      kind: "member",
      record: {
        ...member,
        budget_override: null,
        effective_budget: { max_budget: 100, rpm_limit: 10, allowed_models: ["model-a"] },
      },
    };
    const initial = customerHierarchyFields(entity);
    const expectedInitial = { budgetEnabled: false, maxBudget: "", rpm: "", allowedModels: [] };
    expect(initial).toMatchObject(expectedInitial);
    const fields = { ...initial, budgetEnabled: true, maxBudget: String(limit), rpm: limit === 0 ? "0" : "" };
    const budget = { max_budget: limit, ...(limit === 0 ? { rpm_limit: 0 } : {}) };
    expect(buildCustomerHierarchyPatch(entity, fields)).toEqual({
      ok: true,
      patch: { kind: "member", body: { budget } },
    });
  });

  it.each([
    { changes: { overrideExpiry: "invalid" }, templateId: "template-1", error: "invalidExpiry" },
    { changes: { overrideExpiry: "invalid" }, templateId: "template-2", error: "invalidExpiry" },
    { changes: { maxBudget: "invalid" }, templateId: "template-2", error: "invalidNumber" },
  ])("rejects invalid temporary edits with preset $templateId: $changes", ({ changes, templateId, error }) => {
    const entity: CustomerHierarchyEntity = { kind: "member", record: member };
    const fields = { ...customerHierarchyFields(entity), templateId, ...changes };
    expect(buildCustomerHierarchyPatch(entity, fields)).toEqual({ ok: false, error });
  });
});

describe("customer hierarchy creation payloads", () => {
  const teamTarget: CustomerHierarchyCreateTarget = { kind: "team", organization };
  const preset = {
    is_default: true,
    max_budget: 50,
    tpm_limit: 1700,
    rpm_limit: 17,
    allowed_models: ["model-a"],
    budget_limits: [{ budget_duration: "12mo", max_budget: 600, reset_at: "2027-01-01T00:00:00Z" }],
  };
  const organizationWithTemplate: Organization = {
    ...organization,
    default_budget_template_id: "standard-template",
    default_budgets: [{ budget_id: "standard-template", name: "Standard", ...preset }],
  };
  const memberTarget: CustomerHierarchyCreateTarget = {
    kind: "member",
    organization: organizationWithTemplate,
    team: { ...audit, customer_team_id: "team-1", parent_org_id: organization.customer_org_id },
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

  it.each([false, true])("selects the default template for creation, copying it for custom limits %s", (custom) => {
    const fields = newCustomerHierarchyFields(memberTarget);
    expect(fields.templateId).toBe("standard-template");
    expect(fields.budgetEnabled).toBe(false);
    expect(fields.maxBudget).toBe("");
    expect(fields.allowedModels).toEqual([]);
    const edited = { ...fields, budgetEnabled: custom, maxBudget: custom ? "75" : "" };
    expect(buildCustomerHierarchyCreate(memberTarget, edited, "")).toEqual({
      ok: true,
      create: {
        kind: "member",
        body: {
          parent_team_ids: ["team-1"],
          alias: null,
          ...(custom ? { budget: { ...preset, max_budget: 75 } } : { budget_id: "standard-template" }),
        },
      },
    });
  });

  it("should enable an override when the organization has no templates", () => {
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
          budget: { max_budget: 0, allowed_models: ["gpt-4.1"] },
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
