/**
 * Maps the auto router's shunt form state to the three flat litellm_params keys the backend
 * reads (litellm.proxy.guardrails.auto_router_shunt), and back.
 *
 * `minLines` being undefined means the section was never touched: all three keys are omitted
 * from the payload, and shunt stays unarmed for this router. Once a caller sets a threshold,
 * shunt is armed; the two worker-model fields are optional even then, since the backend
 * defaults each to the router's own default model when unset (auto_router_default_model /
 * complexity_router_default_model).
 */

export interface AutoRouterShuntState {
  minLines: number | undefined;
  bulkReadModel: string | undefined;
  codeWriteModel: string | undefined;
}

export interface AutoRouterShuntLitellmParams {
  auto_router_shunt_min_lines?: number;
  auto_router_shunt_bulk_read_model?: string;
  auto_router_shunt_code_write_model?: string;
}

export const DEFAULT_AUTO_ROUTER_SHUNT_MIN_LINES = 350;

export const DEFAULT_AUTO_ROUTER_SHUNT: AutoRouterShuntState = {
  minLines: undefined,
  bulkReadModel: undefined,
  codeWriteModel: undefined,
};

export const buildAutoRouterShuntParams = (state: AutoRouterShuntState): AutoRouterShuntLitellmParams => {
  if (state.minLines === undefined) return {};
  return {
    auto_router_shunt_min_lines: state.minLines,
    ...(state.bulkReadModel && { auto_router_shunt_bulk_read_model: state.bulkReadModel }),
    ...(state.codeWriteModel && { auto_router_shunt_code_write_model: state.codeWriteModel }),
  };
};

export const hydrateAutoRouterShunt = (litellmParams: {
  auto_router_shunt_min_lines?: number | null;
  auto_router_shunt_bulk_read_model?: string | null;
  auto_router_shunt_code_write_model?: string | null;
}): AutoRouterShuntState => {
  const storedMinLines = litellmParams.auto_router_shunt_min_lines ?? undefined;
  if (storedMinLines === undefined) return DEFAULT_AUTO_ROUTER_SHUNT;
  return {
    minLines: storedMinLines,
    bulkReadModel: litellmParams.auto_router_shunt_bulk_read_model ?? undefined,
    codeWriteModel: litellmParams.auto_router_shunt_code_write_model ?? undefined,
  };
};

/** From a partial-fit preset's own top-level auto_router_shunt_min_lines (see
 * autorouter_presets.ts's AutoRouterPreset.auto_router_shunt_min_lines), or untouched. */
export const shuntStateFromPreset = (presetMinLines: number | undefined): AutoRouterShuntState =>
  presetMinLines === undefined
    ? DEFAULT_AUTO_ROUTER_SHUNT
    : { minLines: presetMinLines, bulkReadModel: undefined, codeWriteModel: undefined };
