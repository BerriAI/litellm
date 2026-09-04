import type { ModelAccessGroupBudget } from "@/app/(dashboard)/hooks/modelAccessGroups/useModelAccessGroups";
import type { SetModelAccessGroupBudgetParams } from "@/app/(dashboard)/hooks/modelAccessGroups/useSetModelAccessGroupBudget";

export interface AccessGroupBudgetFormValues {
  max_budget?: string;
  soft_budget?: string;
  budget_duration?: string;
}

export const accessGroupBudgetFormValues = (
  budget: ModelAccessGroupBudget | null | undefined,
): Required<AccessGroupBudgetFormValues> => ({
  max_budget: budget?.max_budget != null ? String(budget.max_budget) : "",
  soft_budget: budget?.soft_budget != null ? String(budget.soft_budget) : "",
  budget_duration: budget?.budget_duration ?? "",
});

/**
 * Blank fields are left out rather than sent as null: the proxy drops nulls when merging a
 * budget update, so sending one would look like a clear and silently change nothing.
 */
export const buildAccessGroupBudgetBody = (values: AccessGroupBudgetFormValues): SetModelAccessGroupBudgetParams => ({
  ...(values.max_budget ? { max_budget: Number(values.max_budget) } : {}),
  ...(values.soft_budget ? { soft_budget: Number(values.soft_budget) } : {}),
  ...(values.budget_duration ? { budget_duration: values.budget_duration } : {}),
});

export const hasAnyBudgetValue = (values: AccessGroupBudgetFormValues): boolean =>
  Object.keys(buildAccessGroupBudgetBody(values)).length > 0;
