type Metadata = Record<string, unknown> | null | undefined;

type FormValues = Record<string, unknown>;

const ESTIMATE_FIELD = "default_estimated_output_tokens";
const PER_MODEL_FIELD = "default_estimated_output_tokens_per_model";

const INVALID_PER_MODEL_MESSAGE = 'Enter a JSON object of positive integers, e.g. {"gpt-4": 4096}';

const perModelEstimateToText = (value: unknown): string =>
  value != null && typeof value === "object" ? JSON.stringify(value) : "";

const isPositiveInteger = (value: unknown): boolean =>
  typeof value === "number" && Number.isInteger(value) && value > 0;

const parsePerModelEstimates = (value: string): Record<string, number> | null => {
  let parsed: unknown;
  try {
    parsed = JSON.parse(value);
  } catch {
    return null;
  }
  if (parsed == null || typeof parsed !== "object" || Array.isArray(parsed)) return null;
  const entries = Object.entries(parsed as Record<string, unknown>);
  if (entries.length === 0 || !entries.every(([, v]) => isPositiveInteger(v))) return null;
  return Object.fromEntries(entries) as Record<string, number>;
};

export const estimateFields = (metadata: Metadata) => ({
  [ESTIMATE_FIELD]: metadata?.[ESTIMATE_FIELD],
  [PER_MODEL_FIELD]: perModelEstimateToText(metadata?.[PER_MODEL_FIELD]),
});

const ADMIN_ONLY_TOOLTIP =
  "Only a proxy admin can change this. It sets how many output tokens the rate limiter reserves for a request " +
  "that omits max_tokens, which is charged against the team and organization TPM windows.";

export const estimateTooltips = (canEdit: boolean, entity: "key" | "team" = "key") => ({
  estimate: canEdit
    ? `Expected output tokens reserved for TPM limiting when a request omits max_tokens. Overrides the built-in estimate for this ${entity}.`
    : ADMIN_ONLY_TOOLTIP,
  perModel: canEdit
    ? `Per-model expected output tokens reserved for TPM limiting when a request omits max_tokens. Takes precedence over the ${entity}-wide estimate.`
    : ADMIN_ONLY_TOOLTIP,
});

export const estimateChecks = {
  perModel: {
    isValid: (value: unknown): boolean =>
      typeof value !== "string" || value.trim() === "" ? true : parsePerModelEstimates(value) !== null,
    message: INVALID_PER_MODEL_MESSAGE,
  },
  positive: {
    isValid: (value: unknown): boolean =>
      value === "" || value === null || value === undefined ? true : isPositiveInteger(Number(value)),
    message: "Enter a positive integer",
  },
};

const asValidatorRule = ({ isValid, message }: { isValid: (value: unknown) => boolean; message: string }) => ({
  validator: (_: unknown, value: unknown) => (isValid(value) ? Promise.resolve() : Promise.reject(new Error(message))),
});

export const estimateRules = {
  perModel: asValidatorRule(estimateChecks.perModel),
  positive: asValidatorRule(estimateChecks.positive),
};

export const withNormalizedEstimates = <T extends FormValues>(values: T): FormValues => {
  const { [ESTIMATE_FIELD]: estimate, [PER_MODEL_FIELD]: perModel, ...rest } = values;

  const normalizedEstimate = estimate === "" || estimate === null || estimate === undefined ? null : Number(estimate);
  const normalizedPerModel = typeof perModel === "string" ? parsePerModelEstimates(perModel) : null;

  return {
    ...rest,
    ...(normalizedEstimate === null ? {} : { [ESTIMATE_FIELD]: normalizedEstimate }),
    ...(normalizedPerModel === null ? {} : { [PER_MODEL_FIELD]: normalizedPerModel }),
  };
};
