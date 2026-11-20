import type { components } from "@/lib/http/schema";

import {
  budgetEditorFields,
  budgetChanges,
  compactBudgetEditorFields,
  compactBudgetEditorValue,
  customerBaseBudget,
  customerBudgetPatch,
  hasLegacyCustomerBudget,
  parseBudget,
  type Budget,
  type BudgetEditorFields,
} from "./customerBudgetForm";
export { budgetFields, parseBudget, type BudgetFields } from "./customerBudgetForm";

export type Organization = components["schemas"]["Organization"];
export type Team = components["schemas"]["Team"];
export type Member = components["schemas"]["CustomerMember"];

export type CustomerHierarchyEntity =
  | { kind: "organization"; record: Organization }
  | { kind: "team"; record: Team }
  | { kind: "member"; record: Member };

export type CreatedCustomerHierarchyEntity = Extract<CustomerHierarchyEntity, { kind: "team" | "member" }>;
export type CustomerHierarchyCreateTarget =
  | { kind: "team"; organization: Organization }
  | { kind: "member"; organization: Organization; team?: Team };

export type CustomerHierarchyCreate =
  | { kind: "team"; body: components["schemas"]["TeamCreate"] }
  | { kind: "member"; body: components["schemas"]["MemberCreate"]; defaultTeamOrganizationId?: string };

export type CustomerHierarchyPatch =
  | { kind: "organization"; body: components["schemas"]["OrganizationPatch"] }
  | { kind: "team"; body: components["schemas"]["TeamPatch"] }
  | { kind: "member"; body: components["schemas"]["MemberPatch"] };

export interface CustomerHierarchyFields extends BudgetEditorFields {
  alias: string;
  templateId: string;
}

export const CUSTOMER_HIERARCHY_FORM_ERRORS = {
  invalidNumber: "customerPortal.hierarchy.invalidNumber",
  budgetRequired: "customerPortal.hierarchy.budgetRequired",
  invalidIdentifier: "customerPortal.hierarchy.invalidIdentifier",
  invalidTemplate: "customerPortal.hierarchy.invalidTemplate",
  conflictingBudgetSources: "customerPortal.hierarchy.conflictingBudgetSources",
} as const;

type FormError = keyof typeof CUSTOMER_HIERARCHY_FORM_ERRORS;
type PatchResult = { ok: true; patch: CustomerHierarchyPatch } | { ok: false; error: FormError };
type CreateResult = { ok: true; create: CustomerHierarchyCreate } | { ok: false; error: FormError };

export const customerEntityId = (entity: CustomerHierarchyEntity): string => {
  switch (entity.kind) {
    case "organization":
      return entity.record.customer_org_id;
    case "team":
      return entity.record.customer_team_id;
    case "member":
      return entity.record.customer_member_id;
  }
};

export const customerEntityLabel = (entity: CustomerHierarchyEntity): string =>
  entity.record.alias || customerEntityId(entity);

export const customerHierarchyBudget = (entity: CustomerHierarchyEntity): Budget | null | undefined =>
  entity.kind === "member" ? customerBaseBudget(entity.record) : entity.record.budget;

export const customerHierarchyFields = (
  entity: CustomerHierarchyEntity,
  organization?: Organization,
): CustomerHierarchyFields => {
  const storedBudget = customerHierarchyBudget(entity);
  const memberPreset =
    entity.kind === "member" &&
    !hasLegacyCustomerBudget(entity.record) &&
    organization?.default_budgets.some((preset) => preset.budget_id === entity.record.budget_id);
  const monthly =
    entity.kind === "organization"
      ? storedBudget?.budget_limits?.find((window) => window.budget_duration === "1mo")
      : undefined;
  const privateMember =
    entity.kind === "member" &&
    organization != null &&
    !organization.default_budgets.some((preset) => preset.budget_id === entity.record.budget_id);
  return {
    alias: entity.record.alias ?? "",
    ...(entity.kind === "member" ? compactBudgetEditorFields(storedBudget) : budgetEditorFields(storedBudget)),
    ...(entity.kind === "member" ? { budgetEnabled: !memberPreset } : {}),
    ...(monthly ? { maxBudget: String(monthly.max_budget) } : {}),
    templateId: entity.kind === "team" || privateMember ? "" : entity.record.budget_id ?? "",
  };
};

export const newCustomerHierarchyFields = (target: CustomerHierarchyCreateTarget): CustomerHierarchyFields => {
  const template = target.organization.default_budgets.find(
    (budget) => budget.budget_id === target.organization.default_budget_template_id,
  );
  const isMember = target.kind === "member";
  return {
    alias: "",
    ...budgetEditorFields(null),
    budgetEnabled: isMember && !template,
    budgetDuration: "1mo",
    templateId: isMember ? template?.budget_id ?? "" : "",
  };
};

export const BUDGET_FIELDS = ["budgetEnabled", "maxBudget", "softBudget", "budgetDuration", "tpm", "rpm"] as const;

const budgetChanged = (fields: CustomerHierarchyFields, initial: CustomerHierarchyFields): boolean =>
  BUDGET_FIELDS.some((key) => fields[key] !== initial[key]) ||
  JSON.stringify(fields.allowedModels) !== JSON.stringify(initial.allowedModels) ||
  JSON.stringify(fields.budgetWindows) !== JSON.stringify(initial.budgetWindows);

const aliasChanges = (initial: string | null | undefined, value: string): { alias?: string | null } =>
  value === (initial ?? "") ? {} : { alias: value.trim() || null };

const memberPatch = (
  member: Member,
  fields: CustomerHierarchyFields,
  initial: CustomerHierarchyFields,
): PatchResult => {
  if (!fields.templateId && !fields.budgetEnabled) return { ok: false, error: "budgetRequired" };
  const alias = aliasChanges(member.alias, fields.alias);
  const changed = budgetChanged(fields, initial) || fields.templateId !== initial.templateId;
  if (!changed) return { ok: true, patch: { kind: "member", body: alias } };
  if (fields.budgetEnabled && fields.templateId && fields.templateId !== initial.templateId)
    return { ok: false, error: "conflictingBudgetSources" };
  const parsed = customerBudgetPatch(fields, customerBaseBudget(member), member);
  if (!parsed.ok) return { ok: false, error: "invalidNumber" };
  const body = {
    ...alias,
    ...(fields.budgetEnabled ? parsed.value : { budget_id: fields.templateId }),
    ...(hasLegacyCustomerBudget(member) ? { budget_override: null } : {}),
  };
  return { ok: true, patch: { kind: "member", body } };
};

const organizationBudget = (original: Budget | null | undefined, parsed: Budget): Budget => {
  const windows = original?.budget_limits;
  return {
    ...original,
    ...parsed,
    max_budget: windows?.some((window) => window.budget_duration === "1mo") ? original?.max_budget : parsed.max_budget,
    budget_limits:
      windows?.flatMap((window) => {
        if (window.budget_duration !== "1mo") return [window];
        return parsed.max_budget != null ? [{ ...window, max_budget: parsed.max_budget }] : [];
      }) ?? windows,
  };
};

export const buildCustomerHierarchyPatch = (
  entity: CustomerHierarchyEntity,
  fields: CustomerHierarchyFields,
  organization?: Organization,
): PatchResult => {
  const initial = customerHierarchyFields(entity, organization);
  if (entity.kind === "member") return memberPatch(entity.record, fields, initial);
  const aliasPatch = aliasChanges(entity.record.alias, fields.alias);
  const groupChanged = entity.kind === "organization" && fields.templateId !== initial.templateId;
  if (groupChanged && fields.templateId) {
    return { ok: true, patch: { kind: "organization", body: { ...aliasPatch, budget_id: fields.templateId } } };
  }
  const changed = budgetChanged(fields, initial) || groupChanged;
  const original = entity.record.budget;
  const parsed = compactBudgetEditorValue({ ...fields, budgetEnabled: fields.budgetEnabled && changed }, original);
  if (!parsed.ok) return { ok: false, error: "invalidNumber" };
  const budget =
    parsed.value && entity.kind === "organization" ? organizationBudget(original, parsed.value) : parsed.value;
  const baseline = compactBudgetEditorValue(initial, original);
  const previous =
    baseline.ok && baseline.value && entity.kind === "organization"
      ? organizationBudget(original, baseline.value)
      : original;
  return {
    ok: true,
    patch: {
      kind: entity.kind,
      body: {
        ...aliasPatch,
        ...(changed ? { budget: budget && budgetChanges(budget, previous) } : {}),
      },
    },
  };
};

const createMember = (
  target: Extract<CustomerHierarchyCreateTarget, { kind: "member" }>,
  fields: CustomerHierarchyFields,
  common: { alias: string | null; budget: Budget | null; identifier: string },
): CreateResult => {
  const templateId = fields.budgetEnabled ? null : fields.templateId || null;
  if (templateId && !target.organization.default_budgets.some((template) => template.budget_id === templateId))
    return { ok: false, error: "invalidTemplate" };
  if (!templateId && !fields.budgetEnabled) return { ok: false, error: "budgetRequired" };
  return {
    ok: true,
    create: {
      kind: "member",
      ...(!target.team ? { defaultTeamOrganizationId: target.organization.customer_org_id } : {}),
      body: {
        parent_org_id: target.organization.customer_org_id,
        parent_team_ids: [target.team?.customer_team_id ?? `${target.organization.customer_org_id}:default`],
        alias: common.alias,
        ...(common.identifier ? { customer_member_id: common.identifier } : {}),
        budget_id: templateId,
        ...(fields.budgetEnabled ? { budget: common.budget } : {}),
      },
    },
  };
};

export const buildCustomerHierarchyCreate = (
  target: CustomerHierarchyCreateTarget,
  fields: CustomerHierarchyFields,
  identifier: string,
): CreateResult => {
  const id = identifier.trim();
  const unusablePath = id.includes("/") || id === "." || id === "..";
  if (unusablePath || id.includes(",") || id.length > 256) return { ok: false, error: "invalidIdentifier" };
  const budget = fields.budgetEnabled ? parseBudget(fields) : null;
  if (fields.budgetEnabled && budget === null) return { ok: false, error: "invalidNumber" };
  const common = { alias: fields.alias.trim() || null, budget, identifier: id };
  if (target.kind === "member") return createMember(target, fields, common);
  return {
    ok: true,
    create: {
      kind: "team",
      body: {
        parent_org_id: target.organization.customer_org_id,
        alias: common.alias,
        budget,
        ...(id ? { customer_team_id: id } : {}),
      },
    },
  };
};
