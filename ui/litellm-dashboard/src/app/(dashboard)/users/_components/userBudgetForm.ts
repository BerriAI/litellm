import { z } from "zod";

export const userBudgetSchema = z
  .object({
    unlimited: z.boolean(),
    max_budget: z.string(),
    budget_duration: z.string().nullable(),
  })
  .refine(
    ({ unlimited, max_budget }) => {
      const amount = Number(max_budget);
      const validAmount = max_budget.trim() !== "" && Number.isFinite(amount) && amount >= 0;
      return unlimited || validAmount;
    },
    { path: ["max_budget"], message: "Enter a non-negative budget" },
  );
