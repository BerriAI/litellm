import type { BudgetScope, InheritedBudgetGate } from "@/components/shared/InheritedBudgetHint";

export interface AppliedBudget {
  scope: BudgetScope;
  name: string;
  maxBudget: number;
  spend: number | null;
  budgetDuration: string | null;
}

interface KeyBudgetSource {
  max_budget: number | null;
  spend: number;
  budget_duration?: string | null;
}

export const appliedBudgets = (
  key: KeyBudgetSource,
  keyName: string,
  inherited: readonly InheritedBudgetGate[],
): readonly AppliedBudget[] => [
  ...(key.max_budget != null
    ? [
        {
          scope: "Key" as const,
          name: keyName,
          maxBudget: key.max_budget,
          spend: key.spend,
          budgetDuration: key.budget_duration ?? null,
        },
      ]
    : []),
  ...inherited.map((gate) => ({
    scope: gate.scope,
    name: gate.alias,
    maxBudget: gate.maxBudget,
    spend: gate.spend ?? null,
    budgetDuration: gate.budgetDuration,
  })),
];

export const budgetUsagePercent = (budget: AppliedBudget): number | null =>
  budget.spend == null || budget.maxBudget <= 0 ? null : (budget.spend / budget.maxBudget) * 100;

export const closestToLimit = (budgets: readonly AppliedBudget[]): AppliedBudget | null => {
  if (budgets.length < 2) return null;
  const measured = budgets.flatMap((budget) => {
    const usage = budgetUsagePercent(budget);
    return usage === null ? [] : [{ budget, usage }];
  });
  if (measured.length === 0) return null;
  return measured.reduce((closest, next) => (next.usage > closest.usage ? next : closest)).budget;
};
