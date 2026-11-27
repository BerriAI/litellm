import { z } from "zod";
import type { BudgetEditorFields } from "./customerBudgetForm";

type BudgetWindow = BudgetEditorFields["budgetWindows"][number]["budget"];
const budgetWindowShape = {
  budget_duration: z.string(),
  max_budget: z.number(),
  reset_at: z.string().nullish(),
  window_start: z.string().nullish(),
};
const budgetWindowSchema = z.looseObject(budgetWindowShape);

const budgetLimitFieldsShape = {
  softBudget: z.string(),
  budgetDuration: z.string(),
  tpm: z.string(),
  rpm: z.string(),
  allowedModels: z.custom<readonly string[]>((value) => z.array(z.string()).safeParse(value).success),
};
export const budgetLimitFieldsSchema = z.object(budgetLimitFieldsShape);

export const budgetEditorFieldsSchema = budgetLimitFieldsSchema.extend({
  maxBudget: z.string(),
  budgetEnabled: z.boolean(),
  budgetWindows: z.array(
    z.object({
      // The PATCH mapper compares window JSON, including original key order.
      budget: z.custom<BudgetWindow>((value) => budgetWindowSchema.safeParse(value).success),
      maxBudget: z.string(),
    }),
  ),
});
