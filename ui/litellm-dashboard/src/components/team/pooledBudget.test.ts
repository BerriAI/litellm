import { describe, expect, it } from "vitest";
import { pooledBudgetSchema } from "./pooledBudget";

describe("pooled budget amount", () => {
  it.each([undefined, null, 0, 0.001, 250])("preserves a valid amount or unset state: %s", (value) => {
    expect(pooledBudgetSchema.parse(value)).toBe(value);
  });

  it.each(["", " ", "0", "100", -1, Number.NaN, Number.POSITIVE_INFINITY, Number.NEGATIVE_INFINITY])(
    "rejects an empty draft, unparsed value, or invalid amount: %s",
    (value) => {
      expect(pooledBudgetSchema.safeParse(value).success).toBe(false);
    },
  );
});
