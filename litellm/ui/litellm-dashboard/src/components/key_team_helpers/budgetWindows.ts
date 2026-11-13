import type { BudgetWindowEntry } from "./BudgetWindowsEditor";

export interface BudgetWindow {
  readonly budget_duration: string;
  readonly max_budget: number;
}

export const completeBudgetWindows = (windows: readonly BudgetWindowEntry[]): BudgetWindow[] =>
  windows.flatMap(({ budget_duration, max_budget }) =>
    budget_duration && max_budget !== null && max_budget !== undefined ? [{ budget_duration, max_budget }] : [],
  );

const windowSignature = (windows: readonly BudgetWindowEntry[]): string =>
  completeBudgetWindows(windows)
    .map(({ budget_duration, max_budget }) => `${budget_duration}:${max_budget}`)
    .sort()
    .join("|");

export const budgetWindowsUpdate = (
  edited: readonly BudgetWindowEntry[],
  stored: readonly BudgetWindowEntry[],
): BudgetWindow[] | undefined => {
  if (windowSignature(edited) === windowSignature(stored)) return undefined;
  const complete = completeBudgetWindows(edited);
  if (complete.length > 0) return complete;
  return edited.length === 0 ? [] : undefined;
};
