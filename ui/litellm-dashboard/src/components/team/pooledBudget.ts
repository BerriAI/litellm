import { z } from "zod/v4";

export const pooledBudgetSchema = z
  .union([z.number(), z.literal("")])
  .nullish()
  .refine((value) => value !== "", "Enter a pooled budget amount")
  .refine((value) => typeof value !== "number" || value >= 0, "Enter a pooled budget of 0 or more");

export type PooledBudgetValue = z.infer<typeof pooledBudgetSchema>;
