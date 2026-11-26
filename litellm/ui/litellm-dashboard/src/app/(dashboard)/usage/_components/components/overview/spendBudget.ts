import type { TFunction } from "i18next";
import { getBudgetDurationLabel } from "@/components/common_components/budget_duration_dropdown";
import type { UserInfoV2Response } from "@/components/networking";

export interface SpendBudget {
  readonly periodSpend: number;
  readonly maxBudget: number;
  readonly budgetDuration: string | null;
  readonly budgetResetAt: string | null;
}

export function spendBudgetFromUser(user: UserInfoV2Response | null | undefined): SpendBudget | null {
  const maxBudget = user?.effective_max_budget ?? user?.max_budget ?? null;
  if (!user || maxBudget === null) return null;
  return {
    periodSpend: user.spend ?? 0,
    maxBudget,
    budgetDuration: user.budget_duration || null,
    budgetResetAt: user.budget_reset_at ?? null,
  };
}

export function spendBudgetHint(budget: SpendBudget | null, t: TFunction): string {
  if (budget === null) return t("usage.statStrip.noBudget");
  const limit = t("format.usd", { val: budget.maxBudget });
  if (!budget.budgetDuration) return t("usage.statStrip.budgetHint", { budget: limit });
  return t("usage.statStrip.periodBudgetHint", {
    spend: t("format.usd", { val: budget.periodSpend }),
    budget: limit,
    duration: getBudgetDurationLabel(budget.budgetDuration, t),
  });
}
