import { providerLogoMap, Providers, provider_map } from "../provider_info_helpers";

export interface ModelInfo {
  key: string;
  litellm_provider: string;
  mode: string;
  max_input_tokens?: number;
  max_output_tokens?: number;
  max_tokens?: number;
  input_cost_per_token?: number;
  output_cost_per_token?: number;
  supports_function_calling?: boolean;
  supports_vision?: boolean;
  supports_reasoning?: boolean;
  supports_web_search?: boolean;
  supports_response_schema?: boolean;
  supports_audio_input?: boolean;
  supports_audio_output?: boolean;
  supports_prompt_caching?: boolean;
  added_date?: string;
  [other: string]: any;
}

export interface ProviderGroup {
  provider: string;
  displayName: string;
  logo: string;
  models: ModelInfo[];
  modelCount: number;
  modes: string[];
  capabilities: string[];
}

const PROVIDER_DISPLAY_NAMES: Record<string, string> = {};

for (const [enumKey, displayName] of Object.entries(Providers)) {
  const providerValue = provider_map[enumKey];
  if (providerValue) {
    PROVIDER_DISPLAY_NAMES[providerValue] = displayName;
  }
}

const PROVIDER_LOGO_BY_VALUE: Record<string, string> = {};
for (const [enumKey, logo] of Object.entries(providerLogoMap)) {
  const providerValue = provider_map[
    Object.keys(Providers).find(
      (k) => (Providers as Record<string, string>)[k] === enumKey
    ) || ""
  ];
  if (providerValue) {
    PROVIDER_LOGO_BY_VALUE[providerValue] = logo;
  }
}

export function getProviderDisplayName(providerValue: string): string {
  return PROVIDER_DISPLAY_NAMES[providerValue] || formatProviderName(providerValue);
}

export function getProviderLogo(providerValue: string): string {
  return PROVIDER_LOGO_BY_VALUE[providerValue] || "";
}

function formatProviderName(provider: string): string {
  return provider
    .replace(/_/g, " ")
    .replace(/-/g, " ")
    .split(" ")
    .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
    .join(" ");
}

function getModelBaseName(modelKey: string): string {
  const parts = modelKey.split("/");
  return parts[parts.length - 1];
}

export function getCapabilities(model: ModelInfo): string[] {
  const caps: string[] = [];
  if (model.supports_function_calling) caps.push("Function Calling");
  if (model.supports_vision) caps.push("Vision");
  if (model.supports_reasoning) caps.push("Reasoning");
  if (model.supports_web_search) caps.push("Web Search");
  if (model.supports_response_schema) caps.push("Structured Output");
  if (model.supports_audio_input) caps.push("Audio Input");
  if (model.supports_audio_output) caps.push("Audio Output");
  if (model.supports_prompt_caching) caps.push("Prompt Caching");
  return caps;
}

function getModeLabel(mode: string): string {
  const modeMap: Record<string, string> = {
    chat: "Chat",
    embedding: "Embedding",
    completion: "Completion",
    image_generation: "Image Generation",
    audio_transcription: "Audio Transcription",
    audio_speech: "Text-to-Speech",
    moderation: "Moderation",
    rerank: "Rerank",
    search: "Search",
    responses: "Responses",
    video_generation: "Video Generation",
    image_edit: "Image Edit",
    ocr: "OCR",
  };
  return modeMap[mode] || mode;
}

const TOP_PROVIDERS = [
  "openai",
  "anthropic",
  "gemini",
  "azure",
  "bedrock",
  "vertex_ai",
  "mistral",
  "groq",
  "deepseek",
  "fireworks_ai",
  "openrouter",
  "perplexity",
  "cohere",
  "together_ai",
  "cerebras",
  "xai",
  "databricks",
  "sambanova",
  "ollama",
  "azure_ai",
  "minimax",
  "deepinfra",
];

export function parseModelCostMap(
  costMap: Record<string, any>
): ProviderGroup[] {
  const providerGroups: Record<string, ModelInfo[]> = {};

  for (const [key, value] of Object.entries(costMap)) {
    if (key === "sample_spec") continue;
    const provider = value.litellm_provider;
    if (!provider) continue;

    const modelInfo: ModelInfo = {
      key,
      litellm_provider: provider,
      mode: value.mode || "unknown",
      max_input_tokens: value.max_input_tokens,
      max_output_tokens: value.max_output_tokens,
      max_tokens: value.max_tokens,
      input_cost_per_token: value.input_cost_per_token,
      output_cost_per_token: value.output_cost_per_token,
      supports_function_calling: value.supports_function_calling,
      supports_vision: value.supports_vision,
      supports_reasoning: value.supports_reasoning,
      supports_web_search: value.supports_web_search,
      supports_response_schema: value.supports_response_schema,
      supports_audio_input: value.supports_audio_input,
      supports_audio_output: value.supports_audio_output,
      supports_prompt_caching: value.supports_prompt_caching,
      added_date: value.added_date,
    };

    if (!providerGroups[provider]) {
      providerGroups[provider] = [];
    }
    providerGroups[provider].push(modelInfo);
  }

  const groups: ProviderGroup[] = Object.entries(providerGroups).map(
    ([provider, models]) => {
      const modes = [...new Set(models.map((m) => m.mode))].filter(
        (m) => m !== "unknown"
      );
      const allCaps = new Set<string>();
      models.forEach((m) => getCapabilities(m).forEach((c) => allCaps.add(c)));

      return {
        provider,
        displayName: getProviderDisplayName(provider),
        logo: getProviderLogo(provider),
        models,
        modelCount: models.length,
        modes: modes.map(getModeLabel),
        capabilities: [...allCaps],
      };
    }
  );

  groups.sort((a, b) => {
    const aIdx = TOP_PROVIDERS.indexOf(a.provider);
    const bIdx = TOP_PROVIDERS.indexOf(b.provider);
    if (aIdx !== -1 && bIdx !== -1) return aIdx - bIdx;
    if (aIdx !== -1) return -1;
    if (bIdx !== -1) return 1;
    return b.modelCount - a.modelCount;
  });

  return groups;
}

export function detectNewModels(
  costMap: Record<string, any>,
  knownBaselineKeys?: Set<string>
): ModelInfo[] {
  const newModels: ModelInfo[] = [];

  const newPatterns = [
    /gpt-5/i,
    /gpt-4\.1/i,
    /claude-opus-4/i,
    /claude-sonnet-4/i,
    /claude-haiku-4/i,
    /gemini-3/i,
    /gemini-2\.5/i,
    /grok-4/i,
    /deepseek-r2/i,
    /minimax-m2/i,
    /kimi-k2/i,
    /devstral/i,
    /qwen3/i,
    /codex/i,
  ];

  const seen = new Set<string>();

  for (const [key, value] of Object.entries(costMap)) {
    if (key === "sample_spec") continue;

    const baseName = getModelBaseName(key);
    if (seen.has(baseName)) continue;

    const isNew = newPatterns.some((p) => p.test(key));
    if (!isNew) continue;

    seen.add(baseName);

    if (value.mode !== "chat" && value.mode !== "responses") continue;

    newModels.push({
      key,
      litellm_provider: value.litellm_provider || "unknown",
      mode: value.mode || "unknown",
      max_input_tokens: value.max_input_tokens,
      max_output_tokens: value.max_output_tokens,
      max_tokens: value.max_tokens,
      input_cost_per_token: value.input_cost_per_token,
      output_cost_per_token: value.output_cost_per_token,
      supports_function_calling: value.supports_function_calling,
      supports_vision: value.supports_vision,
      supports_reasoning: value.supports_reasoning,
      supports_web_search: value.supports_web_search,
      supports_response_schema: value.supports_response_schema,
      supports_audio_input: value.supports_audio_input,
      supports_audio_output: value.supports_audio_output,
      supports_prompt_caching: value.supports_prompt_caching,
    });
  }

  const providerPriority: Record<string, number> = {
    openai: 0,
    anthropic: 1,
    gemini: 2,
    vertex_ai: 3,
    mistral: 4,
    deepseek: 5,
    xai: 6,
    groq: 7,
  };

  newModels.sort((a, b) => {
    const aPri = providerPriority[a.litellm_provider] ?? 99;
    const bPri = providerPriority[b.litellm_provider] ?? 99;
    return aPri - bPri;
  });

  return newModels;
}

export function formatCost(costPerToken: number | undefined): string {
  if (costPerToken === undefined || costPerToken === null) return "-";
  const perMillion = costPerToken * 1_000_000;
  if (perMillion < 0.01) return "<$0.01/M";
  return `$${perMillion.toFixed(2)}/M`;
}

export function formatContextWindow(tokens: number | undefined): string {
  if (!tokens) return "-";
  if (tokens >= 1_000_000) return `${(tokens / 1_000_000).toFixed(1)}M`;
  if (tokens >= 1_000) return `${Math.round(tokens / 1_000)}K`;
  return String(tokens);
}
