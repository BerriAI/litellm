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
  const storedRouting = litellmParams.auto_router_routing_compression ?? undefined;
  const storedModel = litellmParams.auto_router_model_compression ?? undefined;

  // Only neither key set means the section was never touched. The backend treats
  // either key on its own as an authoritative policy (policy_from_litellm_params), so
  // reading a model-only config as untouched would hide it from the form and let the
  // next save overwrite the stored model hop.
  if (storedRouting === undefined && storedModel === undefined) return DEFAULT_AUTO_ROUTER_COMPRESSION;

  // An absent key on either hop is no compression for that hop, not same-as-the-other:
  // the backend reads it as None. Hydrating it as same-as-routing would make re-saving
  // an unrelated edit write one hop's guardrail onto the other.
  const routing = storedRouting ?? NO_COMPRESSION;
  const model = storedModel ?? NO_COMPRESSION;
  const sameAsRouting = model === routing;
  return { routing, sameAsRouting, model: sameAsRouting ? undefined : model };
};
