export type FusionFailureMode = "fail" | "aggregator_only";
export type FusionPreset = "quality" | "resilient";

export interface FusionRouterConfigValue {
  panel_models: string[];
  aggregator_model: string;
  min_successful_panelists: number;
  panel_timeout_seconds: number;
  max_candidate_chars: number;
  on_quorum_failure: FusionFailureMode;
}

export interface FusionFormValue extends FusionRouterConfigValue {
  model_name: string;
  team_id: string;
}

export const DEFAULT_FUSION_CONFIG: FusionRouterConfigValue = {
  panel_models: [],
  aggregator_model: "",
  min_successful_panelists: 2,
  panel_timeout_seconds: 120,
  max_candidate_chars: 12000,
  on_quorum_failure: "fail",
};

export const presetFailureMode = (preset: FusionPreset): FusionFailureMode =>
  preset === "quality" ? "fail" : "aggregator_only";

const asRecord = (value: unknown): Record<string, unknown> =>
  typeof value === "object" && value !== null && !Array.isArray(value) ? (value as Record<string, unknown>) : {};

const numberOr = (value: unknown, fallback: number): number =>
  typeof value === "number" && Number.isFinite(value) ? value : fallback;

export const parseFusionConfig = (value: unknown): FusionRouterConfigValue => {
  const config = asRecord(value);
  const panelModels = Array.isArray(config.panel_models)
    ? config.panel_models.filter((model): model is string => typeof model === "string" && model.length > 0)
    : [];
  return {
    panel_models: Array.from(new Set(panelModels)),
    aggregator_model: typeof config.aggregator_model === "string" ? config.aggregator_model : "",
    min_successful_panelists: numberOr(config.min_successful_panelists, 2),
    panel_timeout_seconds: numberOr(config.panel_timeout_seconds, 120),
    max_candidate_chars: numberOr(config.max_candidate_chars, 12000),
    on_quorum_failure: config.on_quorum_failure === "aggregator_only" ? "aggregator_only" : "fail",
  };
};

export const fusionConfigError = (value: FusionFormValue, requiresTeamScope: boolean): string | null => {
  if (!value.model_name.trim()) return "Fusion model name is required.";
  if (requiresTeamScope && !value.team_id) return "Select a team to continue.";
  if (!value.aggregator_model) return "Select an aggregator model.";
  if (value.panel_models.length < 2) return "Select at least two panel models.";
  if (value.panel_models.length > 6) return "A Fusion panel can contain at most six models.";
  if (
    !Number.isInteger(value.min_successful_panelists) ||
    value.min_successful_panelists < 1 ||
    value.min_successful_panelists > value.panel_models.length
  ) {
    return "Successful panelists must be between 1 and the panel size.";
  }
  if (value.panel_timeout_seconds <= 0 || value.panel_timeout_seconds > 600) {
    return "Panel timeout must be between 1 and 600 seconds.";
  }
  if (value.max_candidate_chars < 1000 || value.max_candidate_chars > 50000) {
    return "Candidate limit must be between 1,000 and 50,000 characters.";
  }
  return null;
};

export const fusionModelPayload = (value: FusionFormValue, requiresTeamScope: boolean) => ({
  model_name: value.model_name.trim(),
  litellm_params: {
    model: "fusion_router",
    fusion_router_config: {
      panel_models: value.panel_models,
      aggregator_model: value.aggregator_model,
      min_successful_panelists: value.min_successful_panelists,
      panel_timeout_seconds: value.panel_timeout_seconds,
      max_candidate_chars: value.max_candidate_chars,
      on_quorum_failure: value.on_quorum_failure,
    },
  },
  model_info: requiresTeamScope ? { team_id: value.team_id } : {},
});
