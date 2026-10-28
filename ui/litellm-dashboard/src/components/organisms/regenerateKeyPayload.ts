export const MAX_BUDGET_PRECISION = 2;

export interface RegenerateKeyFormValues {
  key_alias?: string | null;
  max_budget?: number | null;
  tpm_limit?: number | null;
  rpm_limit?: number | null;
  duration: string;
  grace_period: string;
}

export type RegenerateKeyPayload = Omit<RegenerateKeyFormValues, "duration"> & { duration?: string };

const shiftExponent = (value: number, places: number): number => {
  const [mantissa, exponent = "0"] = value.toExponential().split("e");
  return Number(`${mantissa}e${Number(exponent) + places}`);
};

export const roundToPrecision = (value: number, precision: number): number => {
  const scaled = shiftExponent(Math.abs(value), precision);
  if (!Number.isFinite(scaled)) return value;
  const rounded = shiftExponent(Math.round(scaled), -precision);
  return value < 0 ? -rounded : rounded;
};

export const buildRegenerateKeyPayload = (values: RegenerateKeyFormValues): RegenerateKeyPayload => {
  const { duration, ...rest } = values;

  return {
    ...rest,
    max_budget:
      typeof rest.max_budget === "number" ? roundToPrecision(rest.max_budget, MAX_BUDGET_PRECISION) : rest.max_budget,
    ...(duration.trim() !== "" ? { duration } : {}),
  };
};
