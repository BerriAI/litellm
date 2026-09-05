/**
 * Maps the auto router's compression form state to the two flat litellm_params keys
 * the backend reads (litellm.proxy.guardrails.auto_router_compression), and back.
 *
 * `routing` being undefined means the section was never touched: both keys are
 * omitted from the payload, and the request's own compression guardrails apply to
 * both hops unchanged. Once `routing` has a value (a guardrail name, or the "none"
 * sentinel for explicit no-compression), the auto router is authoritative and the
 * model side always gets a concrete value too, mirroring `routing` when same-as
 * is chosen and defaulting to "none" otherwise.
 */

export const NO_COMPRESSION = "none";

/** Guardrail providers that compress prompts, mirroring COMPRESSION_GUARDRAIL_PROVIDERS in
 * litellm/proxy/guardrails/auto_router_compression.py. Both are selectable per hop. */
export const COMPRESSION_GUARDRAIL_PROVIDERS: readonly string[] = ["headroom", "compresr"];

export const isCompressionGuardrailProvider = (provider: unknown): boolean =>
  typeof provider === "string" && COMPRESSION_GUARDRAIL_PROVIDERS.includes(provider.toLowerCase());

export interface AutoRouterCompressionState {
  routing: string | undefined;
  sameAsRouting: boolean;
  model: string | undefined;
}

export interface AutoRouterCompressionLitellmParams {
  auto_router_routing_compression?: string;
  auto_router_model_compression?: string;
}

export const DEFAULT_AUTO_ROUTER_COMPRESSION: AutoRouterCompressionState = {
  routing: undefined,
  sameAsRouting: true,
  model: undefined,
};

export const buildAutoRouterCompressionParams = (
  state: AutoRouterCompressionState,
): AutoRouterCompressionLitellmParams => {
  if (state.routing === undefined) return {};
  return {
    auto_router_routing_compression: state.routing,
    auto_router_model_compression: state.sameAsRouting ? state.routing : state.model ?? NO_COMPRESSION,
  };
};

export const hydrateAutoRouterCompression = (litellmParams: {
  auto_router_routing_compression?: string | null;
  auto_router_model_compression?: string | null;
}): AutoRouterCompressionState => {
  const routing = litellmParams.auto_router_routing_compression ?? undefined;
  if (routing === undefined) return DEFAULT_AUTO_ROUTER_COMPRESSION;

  // An absent model key is no model-hop compression, not same-as-routing: the backend
  // reads it as None (policy_from_litellm_params). Hydrating it as same-as-routing
  // would make re-saving an unrelated edit write the routing guardrail onto the model
  // hop and silently start compressing the model call.
  const model = litellmParams.auto_router_model_compression ?? NO_COMPRESSION;
  const sameAsRouting = model === routing;
  return { routing, sameAsRouting, model: sameAsRouting ? undefined : model };
};
