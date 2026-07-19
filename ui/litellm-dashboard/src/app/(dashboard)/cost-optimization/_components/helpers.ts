import { Model } from "@/components/networking";

export interface GuardrailLitellmParams {
  guardrail?: string | null;
  api_base?: string | null;
  default_on?: boolean | null;
}

export interface GuardrailListItem {
  guardrail_id: string;
  guardrail_name: string | null;
  litellm_params?: GuardrailLitellmParams | null;
}

export interface GuardrailListResponse {
  guardrails?: GuardrailListItem[];
}

export const COMPRESSION_GUARDRAIL_PROVIDER = "headroom";

export const isCompressionGuardrail = (guardrail: GuardrailListItem): boolean =>
  (guardrail.litellm_params?.guardrail ?? "").toLowerCase() === COMPRESSION_GUARDRAIL_PROVIDER;

export const compressionGuardrailsOf = (response: GuardrailListResponse): GuardrailListItem[] =>
  (response.guardrails ?? []).filter(isCompressionGuardrail);

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

export interface CacheConnectionInput {
  host: string;
  port: string;
  password: string;
}

export interface CachePayload {
  type: "redis";
  host: string;
  port: number;
  password?: string;
}

export const buildCachePayload = (input: CacheConnectionInput): CachePayload => {
  const base: CachePayload = { type: "redis", host: input.host.trim(), port: Number(input.port) };
  const password = input.password.trim();
  return password === "" ? base : { ...base, password };
};

export const COMPLEXITY_TIERS = ["SIMPLE", "MEDIUM", "COMPLEX", "REASONING"] as const;
export type ComplexityTier = (typeof COMPLEXITY_TIERS)[number];
export type ComplexityTiers = Record<ComplexityTier, string[]>;

export const emptyComplexityTiers = (): ComplexityTiers => ({
  SIMPLE: [],
  MEDIUM: [],
  COMPLEX: [],
  REASONING: [],
});

export interface AutorouterInput {
  name: string;
  defaultModel: string;
  tiers: ComplexityTiers;
}

export const buildComplexityAutorouterPayload = (input: AutorouterInput): Model => ({
  model_name: input.name.trim(),
  litellm_params: {
    model: "auto_router/complexity_router",
    complexity_router_config: { tiers: input.tiers, classifier_type: "heuristic" },
    complexity_router_default_model: input.defaultModel,
  },
  model_info: {},
});

export interface DeploymentLitellmParams {
  model?: string | null;
  complexity_router_default_model?: string | null;
}

export interface DeploymentListItem {
  model_name: string;
  litellm_params?: DeploymentLitellmParams | null;
  model_info?: { id?: string | null } | null;
}

export interface DeploymentListResponse {
  data?: DeploymentListItem[];
}

export const AUTO_ROUTER_MODEL_PREFIX = "auto_router/";

export const isAutoRouterDeployment = (deployment: DeploymentListItem): boolean =>
  (deployment.litellm_params?.model ?? "").startsWith(AUTO_ROUTER_MODEL_PREFIX);

export const autoRoutersOf = (response: DeploymentListResponse): DeploymentListItem[] =>
  (response.data ?? []).filter(isAutoRouterDeployment);
