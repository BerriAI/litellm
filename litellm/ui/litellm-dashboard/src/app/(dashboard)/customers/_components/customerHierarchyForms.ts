import type { components } from "@/lib/http/schema";

export type Organization = components["schemas"]["Organization"];
export type Team = components["schemas"]["Team"];
export type Member = components["schemas"]["CustomerMember"];
type Budget = components["schemas"]["Budget"];

export type CustomerHierarchyEntity =
  | { kind: "organization"; record: Organization }
  | { kind: "team"; record: Team }
  | { kind: "member"; record: Member };

export type CreatedCustomerHierarchyEntity = Extract<CustomerHierarchyEntity, { kind: "team" | "member" }>;
export type CustomerHierarchyCreateTarget =
  | { kind: "team"; organization: Organization }
  | { kind: "member"; organization: Organization; team: Team };

export type CustomerHierarchyCreate =
  | { kind: "team"; body: components["schemas"]["TeamCreate"] }
  | { kind: "member"; body: components["schemas"]["MemberCreate"] };

export type CustomerHierarchyPatch =
  | { kind: "organization"; body: components["schemas"]["OrganizationPatch"] }
  | { kind: "team"; body: components["schemas"]["TeamPatch"] }
  | { kind: "member"; body: components["schemas"]["MemberPatch"] };

export interface CustomerHierarchyFields {
  alias: string;
  budgetEnabled: boolean;
  maxBudget: string;
  softBudget: string;
  budgetDuration: string;
  tpm: string;
  rpm: string;
  allowedModels: readonly string[];
  templateId: string;
  overrideExpiry: string;
}

export const CUSTOMER_HIERARCHY_FORM_ERRORS = {
  invalidNumber: "customerPortal.hierarchy.invalidNumber",
  invalidExpiry: "customerPortal.hierarchy.invalidExpiry",
  budgetRequired: "customerPortal.hierarchy.budgetRequired",
  invalidIdentifier: "customerPortal.hierarchy.invalidIdentifier",
  invalidTemplate: "customerPortal.hierarchy.invalidTemplate",
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

const asText = (value: number | null | undefined): string => (value == null ? "" : String(value));

const localDateTime = (value: string | null | undefined): string => {
  if (!value) return "";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "";
  return new Date(date.getTime() - date.getTimezoneOffset() * 60_000).toISOString().slice(0, 16);
};

const budgetFields = (budget: Budget | null | undefined) => ({
  maxBudget: asText(budget?.max_budget),
  softBudget: asText(budget?.soft_budget),
  budgetDuration: budget?.budget_duration ?? "",
  tpm: asText(budget?.tpm_limit),
  rpm: asText(budget?.rpm_limit),
  allowedModels: budget?.allowed_models ?? [],
});

export const customerHierarchyFields = (entity: CustomerHierarchyEntity): CustomerHierarchyFields => {
  const storedBudget = entity.kind === "member" ? entity.record.budget_override : entity.record.budget;
  return {
    alias: entity.record.alias ?? "",
    budgetEnabled: storedBudget != null,
    ...budgetFields(storedBudget),
    templateId: entity.kind === "member" ? entity.record.budget_id ?? "" : "",
    overrideExpiry: entity.kind === "member" ? localDateTime(entity.record.budget_override_expires_at) : "",
  };
};

export const newCustomerHierarchyFields = (target: CustomerHierarchyCreateTarget): CustomerHierarchyFields => {
  const template = target.organization.default_budgets.find(
    (budget) => budget.budget_id === target.organization.default_budget_template_id,
  );
  const isMember = target.kind === "member";
  return {
    alias: "",
    budgetEnabled: isMember && !template,
    ...budgetFields(null),
    templateId: isMember ? template?.budget_id ?? "" : "",
    overrideExpiry: "",
  };
};

const numericLimit = (raw: string, integer = false): number | null | undefined => {
  const value = raw.trim();
  if (!value) return null;
  if (!(integer ? /^\d+$/ : /^(?:\d+(?:\.\d*)?|\.\d+)$/).test(value)) return undefined;
  const number = Number(value);
  if (!Number.isFinite(number) || (integer && !Number.isSafeInteger(number))) return undefined;
  return number;
};

const parseBudget = (fields: CustomerHierarchyFields): Budget | null => {
  const max = numericLimit(fields.maxBudget);
  const soft = numericLimit(fields.softBudget);
  const tpm = numericLimit(fields.tpm, true);
  const rpm = numericLimit(fields.rpm, true);
  if ([max, soft, tpm, rpm].includes(undefined)) return null;
  const models = fields.allowedModels.map((model) => model.trim()).filter(Boolean);
  return {
    max_budget: max,
    soft_budget: soft,
    tpm_limit: tpm,
    rpm_limit: rpm,
    budget_duration: fields.budgetDuration.trim() || null,
    allowed_models: models.length ? [...new Set(models)] : null,
  };
};

const BUDGET_FIELDS = ["budgetEnabled", "maxBudget", "softBudget", "budgetDuration", "tpm", "rpm"] as const;

const budgetChanged = (fields: CustomerHierarchyFields, initial: CustomerHierarchyFields): boolean =>
  BUDGET_FIELDS.some((key) => fields[key] !== initial[key]) ||
  JSON.stringify(fields.allowedModels) !== JSON.stringify(initial.allowedModels);

const aliasChanges = (initial: string | null | undefined, value: string): { alias?: string | null } =>
  value === (initial ?? "") ? {} : { alias: value.trim() || null };

const memberPatch = (
  member: Member,
  fields: CustomerHierarchyFields,
  budget: Budget | null,
  changed: boolean,
): PatchResult => {
  const initial = customerHierarchyFields({ kind: "member", record: member });
  if (!fields.templateId && !fields.budgetEnabled) return { ok: false, error: "budgetRequired" };
  const body: components["schemas"]["MemberPatch"] = aliasChanges(member.alias, fields.alias);
  if (fields.templateId !== initial.templateId) body.budget_id = fields.templateId || null;
  if (changed) body.budget_override = budget;
  if (!fields.budgetEnabled && initial.budgetEnabled) body.budget_override_expires_at = null;
  if (fields.budgetEnabled && fields.overrideExpiry !== initial.overrideExpiry) {
    const expiry = fields.overrideExpiry ? new Date(fields.overrideExpiry) : null;
    if (expiry && Number.isNaN(expiry.getTime())) return { ok: false, error: "invalidExpiry" };
    body.budget_override_expires_at = expiry?.toISOString() ?? null;
  }
  return { ok: true, patch: { kind: "member", body } };
};

export const buildCustomerHierarchyPatch = (
  entity: CustomerHierarchyEntity,
  fields: CustomerHierarchyFields,
): PatchResult => {
  const initial = customerHierarchyFields(entity);
  const aliasPatch = aliasChanges(entity.record.alias, fields.alias);
  const changed = budgetChanged(fields, initial);
  const budget = fields.budgetEnabled && changed ? parseBudget(fields) : null;
  if (fields.budgetEnabled && changed && budget === null) return { ok: false, error: "invalidNumber" };
  if (entity.kind === "organization" || entity.kind === "team") {
    return {
      ok: true,
      patch: { kind: entity.kind, body: { ...aliasPatch, ...(changed ? { budget } : {}) } },
    };
  }
  return memberPatch(entity.record, fields, budget, changed);
};

const createMember = (
  target: Extract<CustomerHierarchyCreateTarget, { kind: "member" }>,
  fields: CustomerHierarchyFields,
  common: { alias: string | null; budget: Budget | null; identifier: string },
): CreateResult => {
  const templateId = fields.templateId || target.organization.default_budget_template_id || null;
  if (templateId && !target.organization.default_budgets.some((template) => template.budget_id === templateId))
    return { ok: false, error: "invalidTemplate" };
  if (!templateId && !fields.budgetEnabled) return { ok: false, error: "budgetRequired" };
  const expiry = fields.budgetEnabled && fields.overrideExpiry ? new Date(fields.overrideExpiry) : null;
  if (expiry && Number.isNaN(expiry.getTime())) return { ok: false, error: "invalidExpiry" };
  return {
    ok: true,
    create: {
      kind: "member",
      body: {
        parent_team_ids: [target.team.customer_team_id],
        alias: common.alias,
        ...(common.identifier ? { customer_member_id: common.identifier } : {}),
        budget_id: templateId,
        budget_override: common.budget,
        budget_override_expires_at: expiry?.toISOString() ?? null,
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
