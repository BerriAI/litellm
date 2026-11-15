import type { components } from "@/lib/http/schema";
import type { CustomerUsageEntity } from "./customerUsageApi";

export type CustomerBudgetUsage = components["schemas"]["CustomerBudgetUsage"];
export type CustomerBudgetPeriod = "monthly" | "yearly" | "other";

export const CUSTOMER_BUDGET_LABELS = {
  monthly: "customerPortal.analytics.monthlySpend",
  yearly: "customerPortal.analytics.yearlySpend",
  other: "customerPortal.analytics.budgetPeriod",
} as const;

export const customerBudgetPeriod = (duration: string | null): CustomerBudgetPeriod => {
  const normalized = duration?.trim().toLowerCase() ?? "";
  if (["1mo", "1month", "1months", "30d", "30days", "monthly"].includes(normalized)) return "monthly";
  if (["12mo", "12month", "12months"].includes(normalized)) return "yearly";
  return "other";
};

export const customerBudgetUsage = (entity: CustomerUsageEntity): CustomerBudgetUsage[] =>
  entity.budgetUsage ?? [
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

export const customerBudgetStatus = (usage: CustomerBudgetUsage[]) => {
  if (usage.some((budget) => budget.spend !== null && budget.spend >= budget.max_budget))
    return "customerPortal.analytics.reachedLimit" as const;
  if (usage.some((budget) => budget.spend === null)) return "customerPortal.analytics.budgetUsageUnavailable" as const;
  if (usage.length === 0) return "customerPortal.analytics.noLimit" as const;
  return "customerPortal.analytics.withinBudget" as const;
};
