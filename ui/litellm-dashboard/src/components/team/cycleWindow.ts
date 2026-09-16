import { NEVER_RESETS_BUDGET_DURATION } from "../common_components/budget_duration_dropdown";

export type CycleWindowBudget =
  | {
      budget_duration: string | null;
      budget_reset_at: string | null;
    }
  | null
  | undefined;

export const describeCycleWindow = (budget: CycleWindowBudget, formatDate: (iso: string) => string): string => {
  if (budget == null) {
    return "Never resets";
  }
  if (budget.budget_duration == null || budget.budget_duration === NEVER_RESETS_BUDGET_DURATION) {
    return "Never resets";
  }
  if (budget.budget_reset_at == null) {
    return `Resets every ${budget.budget_duration}`;
  }

  return `Resets ${formatDate(budget.budget_reset_at)} (every ${budget.budget_duration})`;
};
