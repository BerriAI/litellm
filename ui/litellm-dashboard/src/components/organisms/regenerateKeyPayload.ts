export const MAX_BUDGET_PRECISION = 2;

export interface RegenerateKeyFormValues {
  key_alias?: string | null;
  max_budget?: number | null;
  tpm_limit?: number | null;
  rpm_limit?: number | null;
  duration: string;
  grace_period: string;
}

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

export const buildRegenerateKeyPayload = (values: RegenerateKeyFormValues): RegenerateKeyFormValues => ({
  ...values,
  max_budget:
    typeof values.max_budget === "number"
      ? roundToPrecision(values.max_budget, MAX_BUDGET_PRECISION)
      : values.max_budget,
});
