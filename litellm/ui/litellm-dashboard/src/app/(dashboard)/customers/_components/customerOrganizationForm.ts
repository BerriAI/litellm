import type { components } from "@/lib/http/schema";
import type { BudgetCapPolicy } from "@/components/shared/BudgetCapPolicySelect";

export interface CustomerBudgetGroupFields {
  id: string;
  name: string;
  monthlyBudget: string;
  yearlyBudget: string;
  capPolicy: BudgetCapPolicy;
}

export interface CustomerOrganizationFields {
  name: string;
  alias: string;
  monthlyBudget: string;
  tpm: string;
  rpm: string;
  groups: CustomerBudgetGroupFields[];
  defaultGroupId: string;
}

export const initialCustomerOrganizationFields = (): CustomerOrganizationFields => ({
  name: "",
  alias: "",
  monthlyBudget: "",
  tpm: "",
  rpm: "",
  groups: [
    { id: "power", name: "power_user", monthlyBudget: "200", yearlyBudget: "2000", capPolicy: "pause_until_reset" },
    { id: "default", name: "default_user", monthlyBudget: "50", yearlyBudget: "500", capPolicy: "pause_until_reset" },
    { id: "low", name: "low_user", monthlyBudget: "10", yearlyBudget: "100", capPolicy: "pause_until_reset" },
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
const limit = (value: string, duration: "1mo" | "12mo") => {
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
  | "defaultRequired";

export function buildCustomerOrganizationBody(fields: CustomerOrganizationFields) {
  if (!fields.name.trim()) return { ok: false, error: "nameRequired" } as const;
  if (!fields.groups.length || !fields.groups.some((group) => group.id === fields.defaultGroupId)) {
    return { ok: false, error: "defaultRequired" } as const;
  }
  const names = fields.groups.map((group) => group.name.trim());
  if (names.some((name) => !name)) return { ok: false, error: "groupNameRequired" } as const;
  if (new Set(names).size !== names.length) return { ok: false, error: "duplicateGroups" } as const;
  if (fields.groups.some((group) => !group.monthlyBudget.trim() && !group.yearlyBudget.trim())) {
    return { ok: false, error: "groupLimitRequired" } as const;
  }
  if (
    !validAmount(fields.monthlyBudget) ||
    fields.groups.some((group) => !validAmount(group.monthlyBudget) || !validAmount(group.yearlyBudget))
  ) {
    return { ok: false, error: "invalidBudget" } as const;
  }
  if (!validRate(fields.tpm) || !validRate(fields.rpm)) {
    return { ok: false, error: "invalidRateLimit" } as const;
  }
  const budget = optionalAmount(fields.monthlyBudget);
  const tpm = optionalAmount(fields.tpm);
  const rpm = optionalAmount(fields.rpm);
  const body = {
    alias: fields.name.trim(),
    ...(fields.alias.trim() ? { customer_org_id: fields.alias.trim() } : {}),
    ...(budget !== null || tpm !== null || rpm !== null
      ? {
          budget: {
            ...(budget !== null ? { max_budget: budget, budget_duration: "1mo" } : {}),
            ...(tpm !== null ? { tpm_limit: tpm } : {}),
            ...(rpm !== null ? { rpm_limit: rpm } : {}),
          },
        }
      : {}),
    default_budgets: fields.groups.map((group) => ({
      name: group.name.trim(),
      is_default: group.id === fields.defaultGroupId,
      budget_limits: [...limit(group.monthlyBudget, "1mo"), ...limit(group.yearlyBudget, "12mo")],
      cap_policy: group.capPolicy,
    })),
  } satisfies components["schemas"]["OrganizationCreate"];
  return { ok: true, body } as const;
}
