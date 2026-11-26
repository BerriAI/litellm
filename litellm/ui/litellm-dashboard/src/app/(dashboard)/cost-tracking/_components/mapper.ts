import type { AddDiscountFields, AddMarginFields } from "./schema";

export const emptyDiscount: AddDiscountFields = { provider: null, discount: "", pattern: "" };

export const emptyMargin: AddMarginFields = {
  provider: null,
  marginType: "percentage",
  percentageValue: "",
  fixedAmountValue: "",
};

export const discountToAddArgs = ({ provider, discount, pattern }: AddDiscountFields) =>
  [provider ?? undefined, discount, pattern] as const;

export const marginToAddArgs = ({ provider, ...values }: AddMarginFields) => ({
  selectedProvider: provider ?? undefined,
  ...values,
});
