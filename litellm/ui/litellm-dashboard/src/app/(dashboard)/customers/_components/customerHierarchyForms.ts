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

export interface BudgetFields {
  maxBudget: string;
  softBudget: string;
  budgetDuration: string;
  tpm: string;
  rpm: string;
  allowedModels: readonly string[];
}

export interface CustomerHierarchyFields extends BudgetFields {
  alias: string;
  budgetEnabled: boolean;
  templateId: string;
  overrideExpiry: string;
}

export const CUSTOMER_HIERARCHY_FORM_ERRORS = {
  invalidNumber: "customerPortal.hierarchy.invalidNumber",
  invalidExpiry: "customerPortal.hierarchy.invalidExpiry",
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

const asText = (value: number | null | undefined): string => (value == null ? "" : String(value));

const localDateTime = (value: string | null | undefined): string => {
  if (!value) return "";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "";
  return new Date(date.getTime() - date.getTimezoneOffset() * 60_000).toISOString().slice(0, 16);
};

export const budgetFields = (budget: Budget | null | undefined): BudgetFields => ({
  maxBudget: asText(budget?.max_budget),
  softBudget: asText(budget?.soft_budget),
  budgetDuration: budget?.budget_duration ?? "",
  tpm: asText(budget?.tpm_limit),
  rpm: asText(budget?.rpm_limit),
  allowedModels: budget?.allowed_models ?? [],
});

const hierarchyEditorBudget = (
  entity: CustomerHierarchyEntity,
  organization?: Organization,
): Budget | null | undefined => {
  if (entity.kind !== "member") return entity.record.budget;
  if (entity.record.budget_override != null) return entity.record.budget_override;
  const preset = organization?.default_budgets.some((template) => template.budget_id === entity.record.budget_id);
  return preset ? null : entity.record.budget;
};

const memberTemplateId = (member: Member, organization?: Organization): string => {
  const privateBudget =
    member.budget != null &&
    organization != null &&
    !organization.default_budgets.some((template) => template.budget_id === member.budget_id);
  return privateBudget ? "" : member.budget_id ?? "";
};

export const customerHierarchyFields = (
  entity: CustomerHierarchyEntity,
  organization?: Organization,
): CustomerHierarchyFields => {
  const storedBudget = hierarchyEditorBudget(entity, organization);
  const monthly =
    entity.kind === "organization"
      ? storedBudget?.budget_limits?.find((window) => window.budget_duration === "1mo")
      : undefined;
  const templateId = entity.kind === "team" ? "" : entity.record.budget_id ?? "";
  return {
    alias: entity.record.alias ?? "",
    budgetEnabled: storedBudget != null,
    ...budgetFields(storedBudget),
    ...(monthly ? { maxBudget: asText(monthly.max_budget) } : {}),
    templateId: entity.kind === "member" ? memberTemplateId(entity.record, organization) : templateId,
    overrideExpiry:
      entity.kind === "member" && entity.record.budget_override != null
        ? localDateTime(entity.record.budget_override_expires_at)
        : "",
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

export const parseBudget = (fields: BudgetFields): Budget | null => {
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

const enteredBudget = (budget: Budget): Budget => ({
  ...(budget.max_budget != null ? { max_budget: budget.max_budget } : {}),
  ...(budget.soft_budget != null ? { soft_budget: budget.soft_budget } : {}),
  ...(budget.tpm_limit != null ? { tpm_limit: budget.tpm_limit } : {}),
  ...(budget.rpm_limit != null ? { rpm_limit: budget.rpm_limit } : {}),
  ...(budget.budget_duration != null ? { budget_duration: budget.budget_duration } : {}),
  ...(budget.allowed_models != null ? { allowed_models: budget.allowed_models } : {}),
});

const permanentMemberBudget = (budget: Budget | null, inheritBlank: boolean): Budget | null =>
  budget && inheritBlank ? enteredBudget(budget) : budget;

export const BUDGET_FIELDS = ["budgetEnabled", "maxBudget", "softBudget", "budgetDuration", "tpm", "rpm"] as const;

const budgetChanged = (fields: CustomerHierarchyFields, initial: CustomerHierarchyFields): boolean =>
  BUDGET_FIELDS.some((key) => fields[key] !== initial[key]) ||
  JSON.stringify(fields.allowedModels) !== JSON.stringify(initial.allowedModels);

const conflictingMemberBudgetSources = (fields: CustomerHierarchyFields, initial: CustomerHierarchyFields): boolean => {
  const selectingPreset = fields.templateId !== "" && fields.templateId !== initial.templateId;
  return selectingPreset && fields.budgetEnabled && budgetChanged(fields, initial);
};

const aliasChanges = (initial: string | null | undefined, value: string): { alias?: string | null } =>
  value === (initial ?? "") ? {} : { alias: value.trim() || null };

const legacyMemberBudgetClear = (member: Member): { budget_override?: null } =>
  member.budget_override != null && member.budget_override_expires_at == null ? { budget_override: null } : {};

const temporaryMemberPatch = (
  fields: CustomerHierarchyFields,
  initial: CustomerHierarchyFields,
  selection: components["schemas"]["MemberPatch"],
): PatchResult => {
  if (!fields.budgetEnabled) {
    return {
      ok: true,
      patch: { kind: "member", body: { ...selection, budget_override: null, budget_override_expires_at: null } },
    };
  }
  const budget = parseBudget(fields);
  if (budget == null) return { ok: false, error: "invalidNumber" };
  const expiry = new Date(fields.overrideExpiry);
  if (Number.isNaN(expiry.getTime())) return { ok: false, error: "invalidExpiry" };
  return {
    ok: true,
    patch: {
      kind: "member",
      body: {
        ...selection,
        budget_override: budget,
        ...(fields.overrideExpiry !== initial.overrideExpiry
          ? { budget_override_expires_at: expiry.toISOString() }
          : {}),
      },
    },
  };
};

const permanentMemberPatch = (
  fields: CustomerHierarchyFields,
  inheritBlank: boolean,
  changes: components["schemas"]["MemberPatch"],
): PatchResult => {
  const budget = fields.budgetEnabled ? parseBudget(fields) : null;
  if (fields.budgetEnabled && budget == null) return { ok: false, error: "invalidNumber" };
  const selection = fields.budgetEnabled
    ? { budget: permanentMemberBudget(budget, inheritBlank) }
    : { budget_id: fields.templateId };
  return { ok: true, patch: { kind: "member", body: { ...changes, ...selection } } };
};

const memberPatch = (
  member: Member,
  fields: CustomerHierarchyFields,
  initial: CustomerHierarchyFields,
): PatchResult => {
  const alias = aliasChanges(member.alias, fields.alias);
  const clearsExpiry = initial.overrideExpiry !== "" && fields.overrideExpiry === "";
  const clearOverride = clearsExpiry
    ? { budget_override: null, budget_override_expires_at: null }
    : legacyMemberBudgetClear(member);
  const templateChanged = fields.templateId !== initial.templateId;
  const templateSelection = templateChanged ? { budget_id: fields.templateId || null } : {};
  const changed = budgetChanged(fields, initial) || fields.overrideExpiry !== initial.overrideExpiry;
  const removingTemporary = !fields.budgetEnabled && initial.overrideExpiry !== "";
  const temporaryEdit = changed && (fields.overrideExpiry !== "" || removingTemporary);
  const retainedBase = temporaryEdit && !templateChanged && member.budget_id;
  if (!fields.templateId && !fields.budgetEnabled && !retainedBase) return { ok: false, error: "budgetRequired" };
  if (temporaryEdit) return temporaryMemberPatch(fields, initial, { ...alias, ...templateSelection });
  if (conflictingMemberBudgetSources(fields, initial)) return { ok: false, error: "conflictingBudgetSources" };
  if (templateChanged && fields.templateId)
    return { ok: true, patch: { kind: "member", body: { ...alias, budget_id: fields.templateId, ...clearOverride } } };
  if (!changed) return { ok: true, patch: { kind: "member", body: { ...alias, ...templateSelection } } };
  return permanentMemberPatch(fields, !initial.budgetEnabled || member.budget_override != null, {
    ...alias,
    ...clearOverride,
  });
};

const organizationBudget = (original: Budget | null | undefined, parsed: Budget, groupChanged: boolean): Budget => {
  const windows = groupChanged ? null : original?.budget_limits;
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
  const parsed = fields.budgetEnabled && changed ? parseBudget(fields) : null;
  const budget =
    parsed && entity.kind === "organization" ? organizationBudget(entity.record.budget, parsed, groupChanged) : parsed;
  if (fields.budgetEnabled && changed && budget === null) return { ok: false, error: "invalidNumber" };
  return {
    ok: true,
    patch: {
      kind: entity.kind,
      body: {
        ...aliasPatch,
        ...(changed ? { budget } : {}),
      },
    },
  };
};

const memberCreateBudget = (
  organization: Organization,
  templateId: string | null,
  budget: Budget | null,
): Budget | null => {
  const template = organization.default_budgets.find((preset) => preset.budget_id === templateId);
  if (!template || !budget) return budget;
  const { name, budget_id, ...base } = template;
  return { ...base, ...enteredBudget(budget) };
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
  const permanent = fields.budgetEnabled
    ? { budget: memberCreateBudget(target.organization, templateId, common.budget) }
    : { budget_id: templateId };
  return {
    ok: true,
    create: {
      kind: "member",
      body: {
        parent_team_ids: [target.team.customer_team_id],
        alias: common.alias,
        ...(common.identifier ? { customer_member_id: common.identifier } : {}),
        ...(expiry
          ? { budget_id: templateId, budget_override: common.budget, budget_override_expires_at: expiry.toISOString() }
          : permanent),
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
