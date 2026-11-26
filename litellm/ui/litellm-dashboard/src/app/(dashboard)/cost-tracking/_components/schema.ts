import { z } from "zod";

export const addDiscountSchema = z.object({
  provider: z.string().nullable(),
  discount: z.string(),
  pattern: z.string(),
});

const addMarginFields = {
  provider: z.string().nullable(),
  marginType: z.enum(["percentage", "fixed"]),
  percentageValue: z.string(),
  fixedAmountValue: z.string(),
};

export const addMarginSchema = z.object(addMarginFields);

export type AddDiscountFields = z.infer<typeof addDiscountSchema>;
export type AddMarginFields = z.infer<typeof addMarginSchema>;
