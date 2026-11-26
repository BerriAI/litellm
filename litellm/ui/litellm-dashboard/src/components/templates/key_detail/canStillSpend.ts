import type { AppliedBudget } from "@/components/shared/appliedBudgets";

export type CanStillSpend =
  | { readonly kind: "noBudget" }
  | { readonly kind: "unmeasured" }
  | {
      readonly kind: "measured";
      readonly budget: AppliedBudget;
      readonly spend: number;
      readonly remaining: number;
      readonly throttles: boolean;
    };

export const budgetThrottles = (budget: AppliedBudget, keyThrottlesOnBudget: boolean): boolean =>
  budget.scope === "Key" && keyThrottlesOnBudget;

export const canStillSpend = (budgets: readonly AppliedBudget[], keyThrottlesOnBudget: boolean): CanStillSpend => {
  if (budgets.length === 0) return { kind: "noBudget" };
  const measured = budgets.flatMap((budget) =>
    budget.spend == null
      ? []
      : [{ budget, spend: budget.spend, remaining: Math.max(0, budget.maxBudget - budget.spend) }],
  );
  if (measured.length === 0) return { kind: "unmeasured" };
  const tightest = measured.reduce((least, next) => (next.remaining < least.remaining ? next : least));
  return { kind: "measured", ...tightest, throttles: budgetThrottles(tightest.budget, keyThrottlesOnBudget) };
};

export const usedUpBudget = (budgets: readonly AppliedBudget[], keyThrottlesOnBudget: boolean): AppliedBudget | null =>
  budgets.find(
    (budget) =>
      budget.spend != null && budget.spend >= budget.maxBudget && !budgetThrottles(budget, keyThrottlesOnBudget),
  ) ?? null;
