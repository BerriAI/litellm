interface ValidatorRule {
  validator: (rule: unknown, value: unknown) => Promise<void>;
}

interface FormInstance {
  getFieldValue: (name: string) => unknown;
  isFieldTouched?: (name: string) => boolean;
}

export const PTU_COUNT_FIELD = "ptu_count";
export const PTU_RATE_FIELD = "cost_per_ptu_per_hour";
export const PTU_START_FIELD = "ptu_effective_from";
export const PTU_END_FIELD = "ptu_effective_to";

// Mirrors ModelInfo.MAX_PTU_COUNT / MAX_COST_PER_PTU_PER_HOUR. Flat cost multiplies the
// count by a float, so the backend caps both; without the same ceiling here the form
// reports valid input and the save then fails with a 422 the operator cannot anticipate.
export const MAX_PTU_COUNT = 1_000_000;
export const MAX_COST_PER_PTU_PER_HOUR = 1_000_000;

export const isFilledPtuValue = (value: unknown): boolean => value !== undefined && value !== null && value !== "";

export const isPositiveWholePtuCount = (value: unknown): boolean => {
  if (!isFilledPtuValue(value)) {
    return true;
  }
  const parsed = Number(value);
  return Number.isInteger(parsed) && parsed > 0 && parsed <= MAX_PTU_COUNT;
};

/** Mirrors the backend contract, which rejects a ptu_count that is not a positive integer. */
export const ptuCountRules: ValidatorRule[] = [
  {
    validator: (_, value) =>
      isPositiveWholePtuCount(value)
        ? Promise.resolve()
        : Promise.reject(new Error(`PTU Count must be a whole number between 1 and ${MAX_PTU_COUNT.toLocaleString()}`)),
  },
];

export const isNonNegativePtuRate = (value: unknown): boolean => {
  if (!isFilledPtuValue(value)) {
    return true;
  }
  const parsed = Number(value);
  return Number.isFinite(parsed) && parsed >= 0 && parsed <= MAX_COST_PER_PTU_PER_HOUR;
};

/** Mirrors the backend contract, which rejects a negative cost_per_ptu_per_hour. */
export const ptuRateRules: ValidatorRule[] = [
  {
    validator: (_, value) =>
      isNonNegativePtuRate(value)
        ? Promise.resolve()
        : Promise.reject(
            new Error(`Cost per PTU / Hour must be between 0 and ${MAX_COST_PER_PTU_PER_HOUR.toLocaleString()}`),
          ),
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
      isFilledPtuValue(value) === isFilledPtuValue(getFieldValue(siblingField))
        ? Promise.resolve()
        : Promise.reject(new Error("PTU Count and Cost per PTU / Hour must be set together")),
  });

/**
 * A PTU deployment is billed by the flat cost of its reserved capacity, so the backend refuses
 * a non-zero per-token price alongside PTU config and stores 0 when none is given. Pair this
 * with `dependencies` on the count so the error clears once the price or the PTU config goes.
 */
export const ptuNoUsageCostRule =
  (countField: string, thisField?: string) =>
  ({ getFieldValue, isFieldTouched }: FormInstance): ValidatorRule => ({
    validator: (_, value) => {
      // A cost the operator never typed was seeded from the rate /model/info resolved, which
      // for an unpriced deployment is the public cost map. Refusing it would block every
      // attempt to put an existing deployment on PTU, and the save omits it anyway.
      const echoed = thisField !== undefined && isFieldTouched !== undefined && !isFieldTouched(thisField);
      return echoed || !isFilledPtuValue(getFieldValue(countField)) || !isFilledPtuValue(value) || Number(value) === 0
        ? Promise.resolve()
        : Promise.reject(new Error("A PTU deployment bills by reserved capacity, so this cost must be 0 or blank"));
    },
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
      isFilledPtuValue(value) || !isFilledPtuValue(getFieldValue(countField))
        ? Promise.resolve()
        : Promise.reject(new Error("PTU Effective From is required when PTU Count is set")),
  });

/** Milliseconds for a picker value, which arrives as a Dayjs (or a Date/ISO string in tests). */
const toEpochMs = (value: unknown): number => {
  const raw = (value as { valueOf?: () => unknown } | null)?.valueOf?.();
  const asNumber = Number(raw);
  return Number.isFinite(asNumber) ? asNumber : new Date(String(value)).getTime();
};

/**
 * The backend rejects a window whose end is not strictly after its start, so an inverted or
 * zero-length window is caught in the form rather than answered with a 422 the operator
 * cannot anticipate. Pair this with `dependencies` on the sibling bound so the error clears
 * once the pair is ordered.
 */
export const ptuWindowIsOrdered = (start: unknown, end: unknown): boolean => {
  if (!isFilledPtuValue(start) || !isFilledPtuValue(end)) {
    return true;
  }
  const startMs = toEpochMs(start);
  const endMs = toEpochMs(end);
  return Number.isNaN(startMs) || Number.isNaN(endMs) || endMs > startMs;
};

export const ptuWindowOrderRule =
  (siblingField: string, thisBound: "start" | "end") =>
  ({ getFieldValue }: FormInstance): ValidatorRule => ({
    validator: (_, value) => {
      const sibling = getFieldValue(siblingField);
      const start = thisBound === "start" ? value : sibling;
      const end = thisBound === "start" ? sibling : value;
      return ptuWindowIsOrdered(start, end)
        ? Promise.resolve()
        : Promise.reject(new Error("PTU Effective To must be after PTU Effective From"));
    },
  });
