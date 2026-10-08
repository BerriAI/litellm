import { canonicalBudgetDuration } from "@/components/templates/keyEditFieldNormalizers";

export type BudgetQuickEditPayload = {
  key: string;
  max_budget: number | null;
  budget_duration?: string | null;
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
  input: {
    maxBudget: string;
    budgetDuration: string | null;
    currentBudgetDuration: string | null | undefined;
  },
): BudgetQuickEditResult => {
  const rawBudget = input.maxBudget.trim();
  const durationUpdate =
    input.budgetDuration !== canonicalBudgetDuration(input.currentBudgetDuration)
      ? { budget_duration: input.budgetDuration }
      : {};

  if (rawBudget === "") {
    return {
      kind: "ok",
      payload: { key: token, max_budget: null, ...durationUpdate },
    };
  }

  const maxBudget = Number(rawBudget);
  if (!Number.isFinite(maxBudget) || maxBudget < 0) {
    return { kind: "invalid", message: "Max budget must be a finite, non-negative number." };
  }

  return {
    kind: "ok",
    payload: { key: token, max_budget: maxBudget, ...durationUpdate },
  };
};

export const modelsQuickEditPayload = (token: string, models: string[]): ModelsQuickEditPayload => {
  if (models.includes("all-team-models")) return { key: token, models: ["all-team-models"] };
  if (models.includes("all-proxy-models")) return { key: token, models: ["all-proxy-models"] };
  return { key: token, models };
};
