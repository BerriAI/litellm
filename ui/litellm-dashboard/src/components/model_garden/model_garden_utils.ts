import { Providers, provider_map } from "../provider_info_helpers";

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

export interface ModelCategory {
  key: string;
  label: string;
  description: string;
  models: ModelInfo[];
}

export interface WhatsNewItem {
  title: string;
  description: string;
  models: string[];
}

const LOGO_PATH = "/assets/logos/";

const PROVIDER_LOGO_FILES: Record<string, string> = {
  openai: "openai_small.svg",
  anthropic: "anthropic.svg",
  gemini: "google.svg",
  vertex_ai: "google.svg",
  azure: "microsoft_azure.svg",
  azure_ai: "microsoft_azure.svg",
  bedrock: "bedrock.svg",
  bedrock_converse: "bedrock.svg",
  sagemaker: "bedrock.svg",
  groq: "groq.svg",
  mistral: "mistral.svg",
  codestral: "mistral.svg",
  deepseek: "deepseek.svg",
  fireworks_ai: "fireworks.svg",
  openrouter: "openrouter.svg",
  perplexity: "perplexity-ai.svg",
  cohere: "cohere.svg",
  cohere_chat: "cohere.svg",
  together_ai: "togetherai.svg",
  cerebras: "cerebras.svg",
  xai: "xai.svg",
  databricks: "databricks.svg",
  sambanova: "sambanova.svg",
  ollama: "ollama.svg",
  minimax: "minimax.svg",
  aws: "aws.svg",
  aiml: "aiml_api.svg",
  snowflake: "snowflake.svg",
  oracle: "oracle.svg",
  oci: "oracle.svg",
};

const PROVIDER_DISPLAY_NAMES: Record<string, string> = {};
for (const [enumKey, displayName] of Object.entries(Providers)) {
  const providerValue = provider_map[enumKey];
  if (providerValue) {
    PROVIDER_DISPLAY_NAMES[providerValue] = displayName;
  }
}

export function getProviderDisplayName(providerValue: string): string {
  return PROVIDER_DISPLAY_NAMES[providerValue] || formatProviderName(providerValue);
}

export function getProviderLogo(providerValue: string): string {
  const file = PROVIDER_LOGO_FILES[providerValue];
  return file ? `${LOGO_PATH}${file}` : "";
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

export function getModeLabel(mode: string): string {
  const modeMap: Record<string, string> = {
    chat: "Text generation",
    embedding: "Embedding",
    completion: "Text generation",
    image_generation: "Image generation",
    audio_transcription: "Speech to text",
    audio_speech: "Text to speech",
    moderation: "Moderation",
    rerank: "Reranking",
    search: "Search",
    responses: "Text generation",
    video_generation: "Video generation",
    image_edit: "Image editing",
    ocr: "OCR",
  };
  return modeMap[mode] || mode;
}

const TOP_PROVIDERS = [
  "openai", "anthropic", "gemini", "azure", "bedrock", "vertex_ai",
  "mistral", "groq", "deepseek", "fireworks_ai", "openrouter",
  "perplexity", "cohere", "together_ai", "cerebras", "xai",
  "databricks", "sambanova", "ollama", "azure_ai", "minimax", "deepinfra",
];

function buildModelInfo(key: string, value: Record<string, any>): ModelInfo {
  return {
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
  };
}

export function parseModelCostMap(costMap: Record<string, any>): ProviderGroup[] {
  const providerGroups: Record<string, ModelInfo[]> = {};

  for (const [key, value] of Object.entries(costMap)) {
    if (key === "sample_spec") continue;
    const provider = value.litellm_provider;
    if (!provider) continue;
    if (!providerGroups[provider]) providerGroups[provider] = [];
    providerGroups[provider].push(buildModelInfo(key, value));
  }

  const groups: ProviderGroup[] = Object.entries(providerGroups).map(
    ([provider, models]) => {
      const modes = [...new Set(models.map((m) => m.mode))].filter((m) => m !== "unknown");
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

export function buildModelCategories(costMap: Record<string, any>): ModelCategory[] {
  const catMap: Record<string, ModelInfo[]> = {};

  for (const [key, value] of Object.entries(costMap)) {
    if (key === "sample_spec") continue;
    if (!value.litellm_provider) continue;
    const mode = value.mode || "unknown";
    if (mode === "unknown") continue;
    if (!catMap[mode]) catMap[mode] = [];
    catMap[mode].push(buildModelInfo(key, value));
  }

  const categoryOrder: Record<string, { label: string; description: string }> = {
    chat: { label: "Foundation models", description: "Large language models for text generation, chat, and reasoning." },
    responses: { label: "Response models", description: "Models optimized for structured response generation." },
    embedding: { label: "Embedding models", description: "Models that convert text into vector representations." },
    image_generation: { label: "Image generation", description: "Models that generate images from text prompts." },
    audio_transcription: { label: "Speech to text", description: "Models that transcribe audio to text." },
    audio_speech: { label: "Text to speech", description: "Models that convert text to spoken audio." },
    rerank: { label: "Reranking models", description: "Models that re-rank search results for relevance." },
    video_generation: { label: "Video generation", description: "Models that generate videos from text or images." },
    search: { label: "Search models", description: "Models for web and knowledge search." },
    moderation: { label: "Moderation", description: "Models for content moderation and safety." },
    completion: { label: "Text completion", description: "Legacy text completion models." },
    image_edit: { label: "Image editing", description: "Models for editing and transforming images." },
    ocr: { label: "OCR models", description: "Models for optical character recognition." },
  };

  const categories: ModelCategory[] = [];
  for (const [mode, meta] of Object.entries(categoryOrder)) {
    const models = catMap[mode];
    if (models && models.length > 0) {
      categories.push({ key: mode, label: meta.label, description: meta.description, models });
    }
  }

  for (const [mode, models] of Object.entries(catMap)) {
    if (!categoryOrder[mode] && models.length > 0) {
      categories.push({
        key: mode,
        label: formatProviderName(mode),
        description: `${models.length} models available.`,
        models,
      });
    }
  }

  return categories;
}

export function detectWhatsNew(costMap: Record<string, any>): WhatsNewItem[] {
  const newPatterns: { pattern: RegExp; title: string; description: string }[] = [
    { pattern: /gpt-5/i, title: "GPT-5 & GPT-5 Pro", description: "OpenAI's latest GPT-5 family models are now available through LiteLLM." },
    { pattern: /codex/i, title: "Codex models", description: "OpenAI Codex models for code generation and understanding." },
    { pattern: /claude-sonnet-4-6|claude-opus-4-6/i, title: "Claude Sonnet 4.6 & Opus 4.6", description: "Anthropic's newest Claude 4.6 models with improved capabilities." },
    { pattern: /gemini-3\.1/i, title: "Gemini 3.1 Pro Preview", description: "Google's Gemini 3.1 Pro is now available on LiteLLM." },
    { pattern: /devstral/i, title: "Devstral models", description: "Mistral's Devstral models for development workflows." },
    { pattern: /minimax-m2/i, title: "MiniMax M2 models", description: "MiniMax M2.1 and M2.5 now available across multiple providers." },
    { pattern: /kimi-k2/i, title: "Kimi K2.5", description: "Moonshot's Kimi K2.5 model now available on Bedrock and Fireworks." },
    { pattern: /qwen3-coder/i, title: "Qwen3 Coder", description: "Alibaba's Qwen3 Coder model for code generation." },
  ];

  const items: WhatsNewItem[] = [];

  for (const { pattern, title, description } of newPatterns) {
    const matchingModels: string[] = [];
    for (const [key, value] of Object.entries(costMap)) {
      if (key === "sample_spec") continue;
      if (pattern.test(key)) matchingModels.push(key);
    }
    if (matchingModels.length > 0) {
      items.push({ title, description, models: matchingModels.slice(0, 5) });
    }
  }

  return items;
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
