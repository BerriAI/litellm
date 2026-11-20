import type { components } from "@/lib/http/schema";

export type CustomerCapPolicy = NonNullable<components["schemas"]["BudgetTemplate"]["cap_policy"]>;
export const defaultCustomerCapPolicy: CustomerCapPolicy = "pause_until_reset";

export interface CustomerBudgetGroupFields {
  id: string;
  name: string;
  monthlyBudget: string;
  yearlyBudget: string;
  tpm: string;
  rpm: string;
  capPolicy: CustomerCapPolicy;
}

export interface CustomerOrganizationFields {
  name: string;
  customerOrgId: string;
  budgetMode: "inherit" | "custom" | "profile";
  budgetId: string;
  budgetAmount: string;
  budgetDuration: "1mo" | "12mo";
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
  tpm: "",
  rpm: "",
  groups: [
    {
      id: "power",
      name: "power_user",
      monthlyBudget: "200",
      yearlyBudget: "2000",
      tpm: "",
      rpm: "",
      capPolicy: defaultCustomerCapPolicy,
    },
    {
      id: "default",
      name: "default_user",
      monthlyBudget: "50",
      yearlyBudget: "500",
      tpm: "",
      rpm: "",
      capPolicy: defaultCustomerCapPolicy,
    },
    {
      id: "low",
      name: "low_user",
      monthlyBudget: "10",
      yearlyBudget: "100",
      tpm: "",
      rpm: "",
      capPolicy: defaultCustomerCapPolicy,
    },
  ],
  defaultGroupId: "default",
});

const optionalAmount = (value: string): number | null => (value.trim() === "" ? null : Number(value));
const validAmount = (value: string): boolean => {
  const amount = optionalAmount(value);
  return amount === null || (Number.isFinite(amount) && amount >= 0);
};
const validRate = (value: string): boolean => {
  const rate = optionalAmount(value);
  return rate === null || (Number.isSafeInteger(rate) && rate >= 0);
};
export const limit = (value: string, duration: "1mo" | "12mo") => {
  const amount = optionalAmount(value);
  return amount === null ? [] : [{ budget_duration: duration, max_budget: amount }];
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
  existingGroupIds: readonly string[] = [],
  requireDefault = true,
) {
  const missingDefault = !fields.defaultGroupId && !existingGroupIds.length;
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
        [group.monthlyBudget, group.yearlyBudget, group.tpm, group.rpm].every((value) => !value.trim()),
    )
  ) {
    return { ok: false, error: "groupLimitRequired" } as const;
  }
  if (fields.groups.some((group) => !validAmount(group.monthlyBudget) || !validAmount(group.yearlyBudget))) {
    return { ok: false, error: "invalidBudget" } as const;
  }
  if (fields.groups.some((group) => !validRate(group.tpm) || !validRate(group.rpm))) {
    return { ok: false, error: "invalidRateLimit" } as const;
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
  if (useCustom && !validAmount(fields.budgetAmount)) return { ok: false, error: "invalidBudget" } as const;
  if (useCustom && [fields.tpm, fields.rpm].some((value) => !validRate(value))) {
    return { ok: false, error: "invalidRateLimit" } as const;
  }
  const budget = optionalAmount(fields.budgetAmount);
  const tpm = optionalAmount(fields.tpm);
  const rpm = optionalAmount(fields.rpm);
  const hasCustomBudget = useCustom && [budget, tpm, rpm].some((value) => value !== null);
  const body = {
    alias: fields.name.trim(),
    ...(fields.customerOrgId.trim() ? { customer_org_id: fields.customerOrgId.trim() } : {}),
    ...(useProfile ? { budget_id: fields.budgetId } : {}),
    ...(useCustom && !hasCustomBudget ? { budget_id: null } : {}),
    ...(hasCustomBudget
      ? {
          budget: {
            ...(budget !== null ? { max_budget: budget, budget_duration: fields.budgetDuration } : {}),
            ...(tpm !== null ? { tpm_limit: tpm } : {}),
            ...(rpm !== null ? { rpm_limit: rpm } : {}),
          },
        }
      : {}),
    default_budgets: fields.groups.map((group): components["schemas"]["BudgetTemplate"] => ({
      name: group.name.trim(),
      cap_policy: group.capPolicy,
      budget_limits: [...limit(group.monthlyBudget, "1mo"), ...limit(group.yearlyBudget, "12mo")],
      ...(group.tpm.trim() ? { tpm_limit: Number(group.tpm) } : {}),
      ...(group.rpm.trim() ? { rpm_limit: Number(group.rpm) } : {}),
    })),
    default_budget_template: fields.groups.find((group) => group.id === fields.defaultGroupId)?.name.trim(),
  } satisfies components["schemas"]["OrganizationCreate"];
  return { ok: true, body } as const;
}
