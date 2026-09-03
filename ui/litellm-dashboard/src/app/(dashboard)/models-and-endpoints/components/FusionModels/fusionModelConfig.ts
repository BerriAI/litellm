export type FusionInvocation = "auto" | "required";
export type FusionPreset = "auto" | "always";
export type FusionReasoningEffort = "none" | "minimal" | "low" | "medium" | "high" | "xhigh";

export interface FusionRouterConfigValue {
  outer_model: string;
  panel_models: string[];
  analyst_model: string;
  invocation: FusionInvocation;
  panel_timeout_seconds: number;
  max_candidate_chars: number;
  max_completion_tokens: number;
  temperature: number;
  reasoning_effort: FusionReasoningEffort;
  search_tool_name: string;
  max_tool_calls: number;
}

export interface FusionFormValue extends FusionRouterConfigValue {
  model_name: string;
  team_id: string;
  web_access_enabled: boolean;
}

export const DEFAULT_FUSION_CONFIG: FusionRouterConfigValue = {
  outer_model: "",
  panel_models: [],
  analyst_model: "",
  invocation: "auto",
  panel_timeout_seconds: 120,
  max_candidate_chars: 12000,
  max_completion_tokens: 16000,
  temperature: 0,
  reasoning_effort: "none",
  search_tool_name: "",
  max_tool_calls: 4,
};

export const presetInvocation = (preset: FusionPreset): FusionInvocation => (preset === "always" ? "required" : "auto");

const asRecord = (value: unknown): Record<string, unknown> =>
  typeof value === "object" && value !== null && !Array.isArray(value) ? (value as Record<string, unknown>) : {};

const numberOr = (value: unknown, fallback: number): number =>
  typeof value === "number" && Number.isFinite(value) ? value : fallback;

const REASONING_EFFORTS = new Set<FusionReasoningEffort>(["none", "minimal", "low", "medium", "high", "xhigh"]);

const webAccessConfigError = (value: FusionFormValue): string | null =>
  value.web_access_enabled && !value.search_tool_name ? "Select a Search Tool or turn Web access off." : null;

export const parseFusionConfig = (value: unknown): FusionRouterConfigValue => {
  const config = asRecord(value);
  const panelModels = Array.isArray(config.panel_models)
    ? config.panel_models.filter((model): model is string => typeof model === "string" && model.length > 0)
    : [];
  const reasoningEffort =
    typeof config.reasoning_effort === "string" &&
    REASONING_EFFORTS.has(config.reasoning_effort as FusionReasoningEffort)
      ? (config.reasoning_effort as FusionReasoningEffort)
      : "none";
  return {
    outer_model: typeof config.outer_model === "string" ? config.outer_model : "",
    panel_models: panelModels,
    analyst_model: typeof config.analyst_model === "string" ? config.analyst_model : "",
    invocation: config.invocation === "required" ? "required" : "auto",
    panel_timeout_seconds: numberOr(config.panel_timeout_seconds, 120),
    max_candidate_chars: numberOr(config.max_candidate_chars, 12000),
    max_completion_tokens: numberOr(config.max_completion_tokens, 16000),
    temperature: numberOr(config.temperature, 0),
    reasoning_effort: reasoningEffort,
    search_tool_name: typeof config.search_tool_name === "string" ? config.search_tool_name : "",
    max_tool_calls: numberOr(config.max_tool_calls, 4),
  };
};

export const fusionConfigError = (value: FusionFormValue, requiresTeamScope: boolean): string | null => {
  if (!value.model_name.trim()) return "Fusion model name is required.";
  if (requiresTeamScope && !value.team_id) return "Select a team to continue.";
  if (!value.outer_model) return "Select the outer model.";
  if (value.panel_models.length < 1) return "Select at least one panel model.";
  if (value.panel_models.length > 8) return "A Fusion panel can contain at most eight models.";
  const webAccessError = webAccessConfigError(value);
  if (webAccessError) return webAccessError;
  if (value.panel_timeout_seconds <= 0 || value.panel_timeout_seconds > 600) {
    return "Panel and analyst timeout must be between 1 and 600 seconds.";
  }
  if (value.max_candidate_chars < 1000 || value.max_candidate_chars > 50000) {
    return "Candidate limit must be between 1,000 and 50,000 characters.";
  }
  if (
    !Number.isInteger(value.max_completion_tokens) ||
    value.max_completion_tokens < 1 ||
    value.max_completion_tokens > 128000
  ) {
    return "Internal output tokens must be between 1 and 128,000.";
  }
  if (value.temperature < 0 || value.temperature > 2) return "Panel temperature must be between 0 and 2.";
  if (!Number.isInteger(value.max_tool_calls) || value.max_tool_calls < 1 || value.max_tool_calls > 16) {
    return "Tool calls must be between 1 and 16.";
  }
  return null;
};

export const fusionModelPayload = (value: FusionFormValue, requiresTeamScope: boolean) => ({
  model_name: value.model_name.trim(),
  litellm_params: {
    model: "fusion_router",
    fusion_router_config: {
      outer_model: value.outer_model,
      panel_models: value.panel_models,
      ...(value.analyst_model ? { analyst_model: value.analyst_model } : {}),
      invocation: value.invocation,
      panel_timeout_seconds: value.panel_timeout_seconds,
      max_candidate_chars: value.max_candidate_chars,
      max_completion_tokens: value.max_completion_tokens,
      temperature: value.temperature,
      reasoning_effort: value.reasoning_effort,
      ...(value.web_access_enabled && value.search_tool_name ? { search_tool_name: value.search_tool_name } : {}),
      max_tool_calls: value.max_tool_calls,
    },
  },
  model_info: requiresTeamScope ? { team_id: value.team_id } : {},
});
