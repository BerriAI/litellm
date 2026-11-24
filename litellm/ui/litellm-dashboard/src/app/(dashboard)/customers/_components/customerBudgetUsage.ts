import type { components } from "@/lib/http/schema";
import type { TFunction } from "i18next";
import { parseAsStringLiteral } from "nuqs";
import type { CustomerUsageEntity } from "./customerUsageApi";
import { budgetLimitAmount, budgetPeriod, type BudgetPeriod } from "@/lib/forms/budgetWindows";

export type CustomerBudgetUsage = components["schemas"]["CustomerBudgetUsage"];
export type CustomerBudgetPeriod = BudgetPeriod;
export const customerBudgetSortParser = parseAsStringLiteral(["budget_asc", "budget_desc"] as const);

export const CUSTOMER_BUDGET_LABELS = {
  monthly: "customerPortal.analytics.monthlySpend",
  yearly: "customerPortal.analytics.yearlySpend",
  other: "customerPortal.analytics.budgetPeriod",
} as const;

export const customerBudgetPeriod = budgetPeriod;

export const customerBudgetWindows = (entity: CustomerUsageEntity): CustomerBudgetUsage[] => [
  ...(entity.budgetLimit === null
    ? []
    : [
        {
          source: "scalar" as const,
          budget_duration: entity.budgetDuration,
          max_budget: entity.budgetLimit,
          spend: entity.spend,
          reset_at: entity.budgetResetAt,
        },
      ]),
  ...entity.budgetLimits.map((limit) => ({
    source: "window" as const,
    budget_duration: limit.budget_duration,
    max_budget: limit.max_budget,
    spend: null,
    reset_at: limit.reset_at,
  })),
];

export const customerBudgetUsage = (entity: CustomerUsageEntity): CustomerBudgetUsage[] =>
  entity.budgetUsage ?? customerBudgetWindows(entity);

export function customerBudgetLimitLabel(amount: string, duration: string | null, t: TFunction): string {
  const period = customerBudgetPeriod(duration);
  if (period === "monthly") return t("customerPortal.createDialog.profileMonthlyBudget", { amount });
  if (period === "yearly") return t("customerPortal.createDialog.profileYearlyBudget", { amount });
  return duration
    ? t("customerPortal.createDialog.profilePeriodBudget", { amount, duration })
    : t("customerPortal.analytics.totalLimitAmount", { amount });
}

export const customerBudgetLimit = (entity: CustomerUsageEntity): number | null =>
  budgetLimitAmount(entity.budgetLimits, entity.budgetLimit);

export const compareCustomerBudgets = (left: CustomerUsageEntity, right: CustomerUsageEntity, descending: boolean) => {
  const a = customerBudgetLimit(left);
  const b = customerBudgetLimit(right);
  if (a === null && b === null) return 0;
  if (a === null) return 1;
  if (b === null) return -1;
  return (a - b) * (descending ? -1 : 1);
};

export const customerBudgetStatus = (usage: CustomerBudgetUsage[]) => {
  if (usage.some((budget) => budget.spend !== null && budget.spend >= budget.max_budget))
    return "customerPortal.analytics.reachedLimit" as const;
  if (usage.some((budget) => budget.spend === null)) return "customerPortal.analytics.budgetUsageUnavailable" as const;
  if (usage.length === 0) return "customerPortal.analytics.noLimit" as const;
  return "customerPortal.analytics.withinBudget" as const;
};
