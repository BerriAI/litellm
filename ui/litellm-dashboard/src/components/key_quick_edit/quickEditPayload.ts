export type BudgetQuickEditPayload = {
  key: string;
  max_budget: number | null;
  budget_duration: string | null;
};

export type BudgetQuickEditResult =
  | { kind: "ok"; payload: BudgetQuickEditPayload }
  | { kind: "invalid"; message: string };

export type ModelsQuickEditPayload = {
  key: string;
  models: string[];
};

export const budgetQuickEditPayload = (
  token: string,
  input: { maxBudget: string; budgetDuration: string | null },
): BudgetQuickEditResult => {
  const rawBudget = input.maxBudget.trim();
  if (rawBudget === "") {
    return {
      kind: "ok",
      payload: { key: token, max_budget: null, budget_duration: input.budgetDuration },
    };
  }

  const maxBudget = Number(rawBudget);
  if (!Number.isFinite(maxBudget) || maxBudget < 0) {
    return { kind: "invalid", message: "Max budget must be a finite, non-negative number." };
  }

  return {
    kind: "ok",
    payload: { key: token, max_budget: maxBudget, budget_duration: input.budgetDuration },
  };
};

export const modelsQuickEditPayload = (token: string, models: string[]): ModelsQuickEditPayload => {
  if (models.includes("all-team-models")) return { key: token, models: ["all-team-models"] };
  if (models.includes("all-proxy-models")) return { key: token, models: ["all-proxy-models"] };
  return { key: token, models };
};
