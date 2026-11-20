import { formatUsd } from "@/app/(dashboard)/usage/_components/components/overview/overviewData";
import { customerBudgetPeriod, type CustomerBudgetUsage } from "../../customers/_components/customerBudgetUsage";
import type { AppMember, AppOrganization, Budget } from "./appForms";
import type { CustomerUsageEntity } from "../../customers/_components/customerUsageApi";

export function appUsageEntity(
  id: string,
  label: string,
  row: Pick<AppOrganization | AppMember, "spend" | "effective_budget" | "budget_usage">,
): CustomerUsageEntity {
  const budget = row.effective_budget;
  return {
    id,
    label,
    spend: row.spend,
    budgetLimit: budget?.max_budget ?? null,
    budgetDuration: budget?.budget_duration ?? null,
    budgetLimits: budget?.budget_limits ?? [],
    budgetResetAt: null,
    budgetUsage: appBudgetWindows(budget, row.budget_usage).windows,
    rateLimits: budget ?? undefined,
  };
}

export const appSpend = (spend: number): string => (spend > 0 && spend < 0.01 ? "<$0.01" : formatUsd(spend));

export const appBudgetReached = (window: Pick<CustomerBudgetUsage, "max_budget" | "spend">): boolean =>
  window.max_budget === 0 || (window.spend !== null && window.spend >= window.max_budget);

export function appBudgetWindows(budget?: Budget | null, usage?: readonly CustomerBudgetUsage[] | null) {
  const scalar =
    budget?.max_budget == null
      ? []
      : [
          {
            source: "scalar" as const,
            budget_duration: budget.budget_duration ?? null,
            max_budget: budget.max_budget,
            spend: null,
            reset_at: null,
          },
        ];
  const configured = (budget?.budget_limits ?? []).map((limit) => ({
    ...limit,
    source: "window" as const,
    spend: null,
    reset_at: null,
  }));
  const windows: CustomerBudgetUsage[] = usage?.length ? [...usage] : [...scalar, ...configured];
  const reached = windows.find(appBudgetReached);
  const monthly = windows.find((window) => customerBudgetPeriod(window.budget_duration) === "monthly");
  const primary = reached ?? monthly ?? windows[0];
  return { primary, windows };
}
