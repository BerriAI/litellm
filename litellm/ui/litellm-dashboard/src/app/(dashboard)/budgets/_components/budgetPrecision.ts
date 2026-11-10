const PRECISION_FIELDS: ReadonlySet<string> = new Set(["tpm_limit", "rpm_limit", "max_budget"]);

const roundToPrecision = (value: number): number => {
  const shifted = Number(`${Math.abs(value)}e2`);
  if (!Number.isFinite(shifted)) {
    return value;
  }
  const rounded = Number(`${Math.round(shifted)}e-2`);
  return value < 0 ? -rounded : rounded;
};

export const applyBudgetPrecision = <TValues extends Record<string, unknown>>(formValues: TValues): TValues =>
  Object.fromEntries(
    Object.entries(formValues).map(([key, value]) => [
      key,
      PRECISION_FIELDS.has(key) && typeof value === "number" ? roundToPrecision(value) : value,
    ]),
  ) as TValues;
