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
