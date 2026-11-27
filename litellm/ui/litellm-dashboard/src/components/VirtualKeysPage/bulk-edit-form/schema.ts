import { z } from "zod";

export const bulkKeyEditSchema = z.object({
  teamId: z.string().nullable(),
  userId: z.string().nullable(),
  budgetInput: z.string(),
  duration: z.string().nullable(),
});

export const bulkKeyBudgetSchema = z.string().transform((input, ctx) => {
  const amount = Number(input);
  if (input.trim() !== "" && Number.isFinite(amount) && amount >= 0) return amount;
  ctx.addIssue({ code: "custom" });
  return z.NEVER;
});
