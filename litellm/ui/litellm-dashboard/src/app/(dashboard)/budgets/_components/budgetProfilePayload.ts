import type { BudgetLimit, BudgetWrite, CapPolicy } from "../../hooks/budgets/budgetProfileTypes";
import { applyBudgetPrecision } from "./budgetPrecision";

export interface ProfileFormValues {
  budget_id: string;
  is_profile?: boolean;
  revision?: string;
  monthly?: number | null;
  annual?: number | null;
  cap_policy?: CapPolicy | null;
  max_budget?: number | null;
  budget_duration?: string | null;
  tpm_limit?: number | null;
  rpm_limit?: number | null;
  tpd_limit?: number | null;
}

export const profileLimits = (monthly?: number | null, annual?: number | null): BudgetLimit[] => [
  ...(monthly == null ? [] : [{ budget_duration: "1mo" as const, max_budget: monthly }]),
  ...(annual == null ? [] : [{ budget_duration: "12mo" as const, max_budget: annual }]),
];

export function budgetProfilePayload(
  values: ProfileFormValues,
  dirty?: Partial<Record<keyof ProfileFormValues, boolean>>,
): BudgetWrite {
  const { monthly, annual, is_profile, revision, ...shared } = values;
  const changed = (name: keyof ProfileFormValues) => dirty === undefined || dirty[name] === true;
  if (!is_profile) {
    const { cap_policy: _policy, ...sharedValues } = shared;
    return applyBudgetPrecision({
      budget_id: values.budget_id,
      ...Object.fromEntries(Object.entries(sharedValues).filter(([name]) => changed(name as keyof ProfileFormValues))),
    });
  }
  return {
    budget_id: values.budget_id,
    ...(dirty === undefined ? { is_profile: true } : {}),
    ...(dirty !== undefined ? { revision } : {}),
    ...(changed("monthly") || changed("annual") ? { budget_limits: profileLimits(monthly, annual) } : {}),
    ...(changed("cap_policy") ? { cap_policy: values.cap_policy } : {}),
  };
}
