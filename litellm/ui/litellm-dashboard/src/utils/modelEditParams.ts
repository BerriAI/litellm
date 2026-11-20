import { isMaskedSecret } from "./maskedSecretUtils";

const FORM_PARAMS = new Set([
  "api_base",
  "custom_llm_provider",
  "organization",
  "tpm",
  "rpm",
  "max_retries",
  "timeout",
  "stream_timeout",
  "model",
  "input_cost_per_token",
  "output_cost_per_token",
  "cache_read_input_token_cost",
  "cache_creation_input_token_cost",
  "cache_creation_input_token_cost_above_1hr",
  "litellm_credential_name",
  "cache_control_injection_points",
  "vector_store_ids",
  "guardrails",
  "tags",
]);

export const editableModelExtraParams = (params: Record<string, unknown>): Record<string, unknown> =>
  Object.fromEntries(Object.entries(params).filter(([key, value]) => !FORM_PARAMS.has(key) && !isMaskedSecret(value)));

export const parseModelExtraParams = (value: string | undefined): Record<string, unknown> | null => {
  try {
    const parsed: unknown = JSON.parse(value?.trim() || "{}");
    return parsed !== null && typeof parsed === "object" && !Array.isArray(parsed)
      ? (parsed as Record<string, unknown>)
      : null;
  } catch {
    return null;
  }
};

const PRICING_KEY = /_cost|^cost_per_second$|_uplift_multiplier|_above_\d+k?_tokens(_batches)?$/;

/**
 * Rates the operator changed in the Model Info JSON, as the litellm_params overrides they stand for.
 * The proxy never stores pricing on model_info, since most of what the form shows there is the
 * cost map filled in for display, so only a rate that differs from what the form loaded is an edit.
 * A rate the same save changes in the LiteLLM Params box wins.
 */
const isRate = (key: string, value: unknown): value is number =>
  PRICING_KEY.test(key) && typeof value === "number" && Number.isFinite(value);

export const editedModelInfoPricing = (
  editedModelInfo: Record<string, unknown>,
  loadedModelInfo: Record<string, unknown>,
  editedParams: Record<string, unknown>,
  loadedParams: Record<string, unknown>,
): Record<string, number> =>
  Object.fromEntries(
    Object.entries(editedModelInfo).filter(
      (entry): entry is [string, number] =>
        isRate(entry[0], entry[1]) &&
        entry[1] !== loadedModelInfo[entry[0]] &&
        !(!FORM_PARAMS.has(entry[0]) && entry[0] in editedParams && editedParams[entry[0]] !== loadedParams[entry[0]]),
    ),
  );
