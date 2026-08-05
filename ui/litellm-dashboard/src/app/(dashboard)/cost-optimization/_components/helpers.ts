import { GuardrailDefinitionLocation } from "@/components/guardrails/types";

export interface GuardrailLitellmParams {
  guardrail?: string | null;
  api_base?: string | null;
  default_on?: boolean | null;
}

export interface GuardrailListItem {
  guardrail_id: string;
  guardrail_name: string | null;
  litellm_params?: GuardrailLitellmParams | null;
  guardrail_definition_location?: GuardrailDefinitionLocation | null;
}

export interface GuardrailListResponse {
  guardrails?: GuardrailListItem[];
}

export const COMPRESSION_GUARDRAIL_PROVIDER = "headroom";

export const isCompressionGuardrail = (guardrail: GuardrailListItem): boolean =>
  (guardrail.litellm_params?.guardrail ?? "").toLowerCase() === COMPRESSION_GUARDRAIL_PROVIDER;

export const compressionGuardrailsOf = (response: GuardrailListResponse): GuardrailListItem[] =>
  (response.guardrails ?? []).filter(isCompressionGuardrail);

/** Guardrails declared in the proxy config file are owned by that file; the API cannot rewrite them. */
export const isConfigDefinedGuardrail = (guardrail: GuardrailListItem): boolean =>
  guardrail.guardrail_definition_location === GuardrailDefinitionLocation.CONFIG;

export interface CompressionGuardrailInput {
  name: string;
  apiBase: string;
  defaultOn: boolean;
}

export const buildCompressionGuardrailPayload = (input: CompressionGuardrailInput): Record<string, unknown> => ({
  guardrail_name: input.name.trim(),
  litellm_params: {
    guardrail: COMPRESSION_GUARDRAIL_PROVIDER,
    mode: "pre_call",
    api_base: input.apiBase.trim(),
    default_on: input.defaultOn,
  },
});
