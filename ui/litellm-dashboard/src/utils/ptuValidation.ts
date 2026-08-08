interface ValidatorRule {
  validator: (rule: unknown, value: unknown) => Promise<void>;
}

interface FormInstance {
  getFieldValue: (name: string) => unknown;
}

export const PTU_COUNT_FIELD = "ptu_count";
export const PTU_RATE_FIELD = "cost_per_ptu_per_hour";
export const PTU_START_FIELD = "ptu_effective_from";

const isFilled = (value: unknown): boolean => value !== undefined && value !== null && value !== "";

const isPositiveWholeNumber = (value: unknown): boolean => {
  if (!isFilled(value)) {
    return true;
  }
  const parsed = Number(value);
  return Number.isInteger(parsed) && parsed > 0;
};

/** Mirrors the backend contract, which rejects a ptu_count that is not a positive integer. */
export const ptuCountRules: ValidatorRule[] = [
  {
    validator: (_, value) =>
      isPositiveWholeNumber(value)
        ? Promise.resolve()
        : Promise.reject(new Error("PTU Count must be a positive whole number")),
  },
];

const isNonNegativeNumber = (value: unknown): boolean => {
  if (!isFilled(value)) {
    return true;
  }
  const parsed = Number(value);
  return Number.isFinite(parsed) && parsed >= 0;
};

/** Mirrors the backend contract, which rejects a negative cost_per_ptu_per_hour. */
export const ptuRateRules: ValidatorRule[] = [
  {
    validator: (_, value) =>
      isNonNegativeNumber(value)
        ? Promise.resolve()
        : Promise.reject(new Error("Cost per PTU / Hour must be a non-negative number")),
  },
];

/**
 * The backend rejects a half-set pair with "ptu_count and cost_per_ptu_per_hour must be set
 * together", so filling or clearing one field without the other is caught in the form. Pair
 * this with `dependencies` on the sibling field so its error clears when the pair is resolved.
 */
export const ptuPairRule =
  (siblingField: string) =>
  ({ getFieldValue }: FormInstance): ValidatorRule => ({
    validator: (_, value) =>
      isFilled(value) === isFilled(getFieldValue(siblingField))
        ? Promise.resolve()
        : Promise.reject(new Error("PTU Count and Cost per PTU / Hour must be set together")),
  });

/**
 * The backend requires an effective start whenever PTU is configured, since flat cost
 * accrues from that instant and an inferred start would bill days a deployment did not
 * exist. Pair this with `dependencies` on the count so the error clears when both resolve.
 */
export const ptuStartRequiredRule =
  (countField: string) =>
  ({ getFieldValue }: FormInstance): ValidatorRule => ({
    validator: (_, value) =>
      isFilled(value) || !isFilled(getFieldValue(countField))
        ? Promise.resolve()
        : Promise.reject(new Error("PTU Effective From is required when PTU Count is set")),
  });
