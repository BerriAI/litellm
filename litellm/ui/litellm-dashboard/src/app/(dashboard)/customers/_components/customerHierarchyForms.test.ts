import { describe, expect, it } from "vitest";
import {
  buildCustomerHierarchyPatch,
  customerHierarchyFields,
  type CustomerHierarchyEntity,
  type Organization,
  type Member,
} from "./customerHierarchyForms";

const audit = {
  created_at: "2026-11-01T00:00:00Z",
  updated_at: "2026-11-01T00:00:00Z",
  created_by: "admin",
  updated_by: "admin",
  spend: 5,
};
const organization: Organization = {
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
  parent_team_id: "team-1",
  alias: "Member One",
  budget_id: "template-1",
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
  ])("should submit template selection %s without changing the override", (templateId, budgetId) => {
    const entity: CustomerHierarchyEntity = { kind: "member", record: member };
    expect(buildCustomerHierarchyPatch(entity, { ...customerHierarchyFields(entity), templateId })).toEqual({
      ok: true,
      patch: { kind: "member", body: { budget_id: budgetId } },
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

  it("should reject an invalid override expiry", () => {
    const entity: CustomerHierarchyEntity = { kind: "member", record: member };
    expect(
      buildCustomerHierarchyPatch(entity, { ...customerHierarchyFields(entity), overrideExpiry: "invalid" }),
    ).toEqual({ ok: false, error: "invalidExpiry" });
  });
});
