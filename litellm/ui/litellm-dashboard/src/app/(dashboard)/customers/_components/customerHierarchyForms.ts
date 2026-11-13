import type { components } from "@/lib/http/schema";

export type Organization = components["schemas"]["Organization"];
export type Team = components["schemas"]["Team"];
export type Member = components["schemas"]["CustomerMember"];
type Budget = components["schemas"]["Budget"];

export type CustomerHierarchyEntity =
  | { kind: "organization"; record: Organization }
  | { kind: "team"; record: Team }
  | { kind: "member"; record: Member };

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
  allowedModels: string;
  templateId: string;
  overrideExpiry: string;
}

type FormError = "invalidNumber" | "invalidExpiry" | "budgetRequired";
type PatchResult = { ok: true; patch: CustomerHierarchyPatch } | { ok: false; error: FormError };

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

export const customerHierarchyFields = (entity: CustomerHierarchyEntity): CustomerHierarchyFields => {
  const storedBudget = entity.kind === "member" ? entity.record.budget_override : entity.record.budget;
  const budget = storedBudget ?? (entity.kind === "member" ? entity.record.effective_budget : null);
  return {
    alias: entity.record.alias ?? "",
    budgetEnabled: storedBudget != null,
    maxBudget: asText(budget?.max_budget),
    softBudget: asText(budget?.soft_budget),
    budgetDuration: budget?.budget_duration ?? "",
    tpm: asText(budget?.tpm_limit),
    rpm: asText(budget?.rpm_limit),
    allowedModels: budget?.allowed_models?.join("\n") ?? "",
    templateId: entity.kind === "member" ? entity.record.budget_id ?? "" : "",
    overrideExpiry: entity.kind === "member" ? localDateTime(entity.record.budget_override_expires_at) : "",
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
  const models = fields.allowedModels
    .split(/\r?\n/)
    .map((model) => model.trim())
    .filter(Boolean);
  return {
    max_budget: max,
    soft_budget: soft,
    tpm_limit: tpm,
    rpm_limit: rpm,
    budget_duration: fields.budgetDuration.trim() || null,
    allowed_models: models.length ? [...new Set(models)] : null,
  };
};

const BUDGET_FIELDS = [
  "budgetEnabled",
  "maxBudget",
  "softBudget",
  "budgetDuration",
  "tpm",
  "rpm",
  "allowedModels",
] as const;

const budgetChanged = (fields: CustomerHierarchyFields, initial: CustomerHierarchyFields): boolean =>
  BUDGET_FIELDS.some((key) => fields[key] !== initial[key]);

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
