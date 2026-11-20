import type { components } from "@/lib/http/schema";
import {
  budgetChanges,
  budgetEditorFields,
  compactBudgetEditorValue,
  parseBudget,
  type Budget,
  type BudgetEditorFields,
} from "./customerBudgetForm";

export type CustomerCapPolicy = NonNullable<components["schemas"]["BudgetTemplate"]["cap_policy"]>;
export const defaultCustomerCapPolicy: CustomerCapPolicy = "pause_until_reset";

export interface CustomerBudgetGroupFields extends BudgetEditorFields {
  id: string;
  name: string;
  capPolicy: CustomerCapPolicy;
}

export interface CustomerOrganizationFields {
  name: string;
  customerOrgId: string;
  budgetMode: "inherit" | "custom" | "profile";
  budgetId: string;
  budgetAmount: string;
  budgetDuration: string;
  softBudget: string;
  allowedModels: readonly string[];
  tpm: string;
  rpm: string;
  groups: CustomerBudgetGroupFields[];
  defaultGroupId: string;
}

export const initialCustomerOrganizationFields = (): CustomerOrganizationFields => ({
  name: "",
  customerOrgId: "",
  budgetMode: "inherit",
  budgetId: "",
  budgetAmount: "",
  budgetDuration: "1mo",
  softBudget: "",
  allowedModels: [],
  tpm: "",
  rpm: "",
  groups: (
    [
      ["power", 200],
      ["default", 50],
      ["low", 10],
    ] as const
  ).map(([id, max_budget]) => ({
    ...budgetEditorFields({ max_budget, budget_duration: "1mo" }),
    id,
    name: `${id}_user`,
    capPolicy: defaultCustomerCapPolicy,
  })),
  defaultGroupId: "default",
});

const optionalAmount = (value: string): number | null => (value.trim() === "" ? null : Number(value));
const validRate = (value: string): boolean => {
  const rate = optionalAmount(value);
  return rate === null || (Number.isSafeInteger(rate) && rate >= 0);
};
export const normalizeGroupModels = (models: readonly string[] = []): string[] =>
  [...new Set(models.map((model) => model.trim()).filter(Boolean))].sort();

export const customerBudgetGroupValue = (group: CustomerBudgetGroupFields, original?: Budget) => {
  const parsed = compactBudgetEditorValue(
    { ...group, allowedModels: normalizeGroupModels(group.allowedModels) },
    original,
  );
  if (!parsed.ok || !parsed.value) return null;
  return {
    ...parsed.value,
    budget_duration:
      parsed.value.max_budget != null || parsed.value.soft_budget != null ? parsed.value.budget_duration : null,
  };
};

export type CustomerOrganizationFormError =
  | "nameRequired"
  | "groupNameRequired"
  | "duplicateGroups"
  | "groupLimitRequired"
  | "invalidBudget"
  | "invalidRateLimit"
  | "profileRequired"
  | "defaultRequired";

export function validateCustomerBudgetGroups(
  fields: Pick<CustomerOrganizationFields, "groups" | "defaultGroupId">,
  existingGroups: Readonly<components["schemas"]["BudgetGroups"]["default_budgets"]> = [],
  requireDefault = true,
) {
  const existingGroupIds = existingGroups.map((group) => group.budget_id);
  const missingDefault = !fields.defaultGroupId && !existingGroups.length;
  const invalidDefault = fields.defaultGroupId && !fields.groups.some((group) => group.id === fields.defaultGroupId);
  if (requireDefault && (missingDefault || invalidDefault)) {
    return { ok: false, error: "defaultRequired" } as const;
  }
  const names = fields.groups.map((group) => group.name.trim());
  if (names.some((name) => !name)) return { ok: false, error: "groupNameRequired" } as const;
  if (new Set(names).size !== names.length) return { ok: false, error: "duplicateGroups" } as const;
  if (
    fields.groups.some(
      (group) =>
        !existingGroupIds.includes(group.id) &&
        !normalizeGroupModels(group.allowedModels).length &&
        [
          group.maxBudget,
          group.softBudget,
          group.tpm,
          group.rpm,
          ...group.budgetWindows.map((window) => window.maxBudget),
        ].every((value) => !value.trim()),
    )
  ) {
    return { ok: false, error: "groupLimitRequired" } as const;
  }
  if (fields.groups.some((group) => !validRate(group.tpm) || !validRate(group.rpm))) {
    return { ok: false, error: "invalidRateLimit" } as const;
  }
  if (
    fields.groups.some(
      (group) =>
        !customerBudgetGroupValue(
          group,
          existingGroups.find((original) => original.budget_id === group.id),
        ),
    )
  ) {
    return { ok: false, error: "invalidBudget" } as const;
  }
  return { ok: true } as const;
}

export function buildCustomerOrganizationBody(fields: CustomerOrganizationFields) {
  if (!fields.name.trim()) return { ok: false, error: "nameRequired" } as const;
  const validation = validateCustomerBudgetGroups(fields);
  if (!validation.ok) return validation;
  const useProfile = fields.budgetMode === "profile";
  const useCustom = fields.budgetMode === "custom";
  if (useProfile && !fields.budgetId) return { ok: false, error: "profileRequired" } as const;
  if (useCustom && [fields.tpm, fields.rpm].some((value) => !validRate(value))) {
    return { ok: false, error: "invalidRateLimit" } as const;
  }
  const parsed = useCustom ? parseBudget({ ...fields, maxBudget: fields.budgetAmount }) : null;
  if (useCustom && !parsed) return { ok: false, error: "invalidBudget" } as const;
  const budget = parsed
    ? budgetChanges({
        ...parsed,
        budget_duration: parsed.max_budget != null || parsed.soft_budget != null ? parsed.budget_duration : null,
      })
    : {};
  const hasCustomBudget = useCustom && Object.keys(budget).length > 0;
  const body = {
    alias: fields.name.trim(),
    ...(fields.customerOrgId.trim() ? { customer_org_id: fields.customerOrgId.trim() } : {}),
    ...(useProfile ? { budget_id: fields.budgetId } : {}),
    ...(useCustom && !hasCustomBudget ? { budget_id: null } : {}),
    ...(hasCustomBudget ? { budget } : {}),
    default_budgets: fields.groups.map((group): components["schemas"]["BudgetTemplate"] => ({
      name: group.name.trim(),
      cap_policy: group.capPolicy,
      ...budgetChanges(customerBudgetGroupValue(group)!),
    })),
    default_budget_template: fields.groups.find((group) => group.id === fields.defaultGroupId)?.name.trim(),
  } satisfies components["schemas"]["OrganizationCreate"];
  return { ok: true, body } as const;
}
