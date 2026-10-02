export const rateLimitUpdate = (
  input: string | number | null | undefined,
  stored: number | null | undefined,
): number | null | undefined => {
  const isNullishInput = input === null || input === undefined;
  const isBlankString = typeof input === "string" && input.trim() === "";
  const normalized = isNullishInput || isBlankString ? null : Number(input);
  return normalized === (stored ?? null) ? undefined : normalized;
};

export const isValidRateLimitInput = (value: string | number | null | undefined): boolean => {
  if (value === "" || value === null || value === undefined) {
    return true;
  }
  const number = Number(value);
  return Number.isSafeInteger(number) && number >= 0;
};
