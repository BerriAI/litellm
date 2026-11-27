import { editableModelExtraParams } from "@/utils/modelEditParams";
import { utcIsoToPickerValue } from "@/utils/ptuDatetime";
import type { ModelEditFormValues } from "./schema";

const perMillionTokens = (...rates: (number | null | undefined)[]): number | null => {
  const rate = rates.find((candidate) => candidate != null);
  return rate == null ? null : rate * 1_000_000;
};

export const toModelEditFormValues = (localModelData: any, isWildcardModel: boolean): ModelEditFormValues => ({
  model_name: localModelData.model_name,
  litellm_model_name: localModelData.litellm_model_name,
  api_base: localModelData.litellm_params.api_base,
  custom_llm_provider: localModelData.litellm_params.custom_llm_provider,
  organization: localModelData.litellm_params.organization,
  tpm: localModelData.litellm_params.tpm,
  rpm: localModelData.litellm_params.rpm,
  max_retries: localModelData.litellm_params.max_retries,
  timeout: localModelData.litellm_params.timeout,
  stream_timeout: localModelData.litellm_params.stream_timeout,
  input_cost: perMillionTokens(
    localModelData.litellm_params.input_cost_per_token,
    localModelData.model_info?.input_cost_per_token,
  ),
  output_cost: perMillionTokens(
    localModelData.litellm_params?.output_cost_per_token,
    localModelData.model_info?.output_cost_per_token,
  ),
  ptu_count: localModelData.model_info?.ptu_count ?? null,
  cost_per_ptu_per_hour: localModelData.model_info?.cost_per_ptu_per_hour ?? null,
  ptu_effective_from: utcIsoToPickerValue(localModelData.model_info?.ptu_effective_from),
  ptu_effective_to: utcIsoToPickerValue(localModelData.model_info?.ptu_effective_to),
  cache_read_cost: perMillionTokens(
    localModelData.litellm_params?.cache_read_input_token_cost,
    localModelData.model_info?.cache_read_input_token_cost,
  ),
  cache_write_cost: perMillionTokens(
    localModelData.litellm_params?.cache_creation_input_token_cost,
    localModelData.model_info?.cache_creation_input_token_cost,
  ),
  cache_write_cost_1h: perMillionTokens(
    localModelData.litellm_params?.cache_creation_input_token_cost_above_1hr,
    localModelData.model_info?.cache_creation_input_token_cost_above_1hr,
  ),
  cache_control: localModelData.litellm_params?.cache_control_injection_points ? true : false,
  cache_control_injection_points: localModelData.litellm_params?.cache_control_injection_points || [],
  model_access_group: Array.isArray(localModelData.model_info?.access_groups)
    ? localModelData.model_info.access_groups
    : [],
  guardrails: Array.isArray(localModelData.litellm_params?.guardrails) ? localModelData.litellm_params.guardrails : [],
  vector_store_ids:
    Array.isArray(localModelData.litellm_params?.vector_store_ids) &&
    localModelData.litellm_params.vector_store_ids.length > 0
      ? localModelData.litellm_params.vector_store_ids
      : undefined,
  tags: Array.isArray(localModelData.litellm_params?.tags) ? localModelData.litellm_params.tags : [],
  // antd never mounted this field for a non-wildcard model, so the key must be absent, not null.
  ...(isWildcardModel ? { health_check_model: localModelData.model_info?.health_check_model } : {}),
  litellm_credential_name: localModelData.litellm_params?.litellm_credential_name ?? null,
  litellm_extra_params: JSON.stringify(editableModelExtraParams(localModelData.litellm_params ?? {}), null, 2),
  team_id: localModelData.model_info?.team_id ?? undefined,
});
