import { getProxyBaseUrl } from "@/components/networking";
import type { AgentCard, ModelGroupInfo } from "@/components/PublicModelHubTableColumns";
import { getProviderLogoAndName } from "@/components/provider_info_helpers";

export const SECTION_TITLE = "text-[15px] leading-[22px] font-semibold";

const MODE_LABELS: Record<string, string> = {
  chat: "Chat",
  completion: "Completion",
  embedding: "Embedding",
  rerank: "Rerank",
  image_generation: "Image generation",
  image_edit: "Image edit",
  audio_speech: "Text to speech",
  audio_transcription: "Transcription",
  video_generation: "Video generation",
  realtime: "Realtime",
  responses: "Responses",
  batch: "Batch",
  ocr: "OCR",
  decisions: "Decisions",
};

const sentenceCase = (text: string): string => text.replaceAll("_", " ").replace(/^./, (c) => c.toUpperCase());

export const modeLabel = (mode: string | null | undefined): string =>
  mode ? MODE_LABELS[mode] ?? sentenceCase(mode) : "-";

export const formatTokens = (tokens: number | null | undefined): string => {
  if (!tokens) return "-";
  if (tokens >= 1_000_000) return `${+(tokens / 1_000_000).toFixed(1)}M`;
  if (tokens >= 1_000) return `${Math.round(tokens / 1_000)}K`;
  return String(tokens);
};

export const formatPerMillion = (cost: number | null | undefined): string =>
  cost ? `$${(cost * 1_000_000).toFixed(2)}` : "-";

export const providerName = (provider: string): string =>
  getProviderLogoAndName(provider).displayName.replace(/\s*\(.*\)$/, "");

export const providerLogo = (provider: string): string => getProviderLogoAndName(provider).logo;

const CAPABILITY_LABELS: Record<string, string> = {
  supports_vision: "Vision",
  supports_function_calling: "Function calling",
  supports_reasoning: "Reasoning",
  supports_web_search: "Web search",
  supports_parallel_function_calling: "Parallel function calling",
  supports_url_context: "URL context",
  supports_fast_mode: "Fast mode",
};

const CAPABILITY_ORDER = Object.keys(CAPABILITY_LABELS);

const capabilityRank = (key: string): number => {
  const index = CAPABILITY_ORDER.indexOf(key);
  return index === -1 ? CAPABILITY_ORDER.length : index;
};

export const capabilityLabel = (key: string): string =>
  CAPABILITY_LABELS[key] ?? sentenceCase(key.replace(/^supports_/, ""));

export const featureFacetLabel = (feature: string): string => capabilityLabel(`supports_${feature}`);

export const modelCapabilities = (model: ModelGroupInfo): string[] =>
  Object.entries(model)
    .filter(([key, value]) => key.startsWith("supports_") && value === true)
    .map(([key]) => key)
    .sort((a, b) => capabilityRank(a) - capabilityRank(b))
    .map(capabilityLabel);

export const agentCapabilities = (agent: AgentCard): string[] =>
  Object.entries(agent.capabilities ?? {})
    .filter(([, on]) => on === true)
    .map(([key]) => (key === "pushNotifications" ? "Push notifications" : sentenceCase(key)));

export type HealthTone = "success" | "error" | "neutral";

const HEALTH: Record<string, { tone: HealthTone; label: string }> = {
  healthy: { tone: "success", label: "Healthy" },
  unhealthy: { tone: "error", label: "Unhealthy" },
};

export const healthOf = (model: ModelGroupInfo): { tone: HealthTone; label: string } =>
  HEALTH[model.health_status ?? ""] ?? { tone: "neutral", label: "Not checked" };

export const relativeTime = (iso: string | null | undefined, now: number = Date.now()): string => {
  if (!iso) return "-";
  const minutes = Math.max(0, Math.round((now - new Date(iso).getTime()) / 60_000));
  if (minutes < 1) return "just now";
  if (minutes < 60) return `${minutes} min ago`;
  const hours = Math.round(minutes / 60);
  if (hours < 24) return `${hours} h ago`;
  return `${Math.round(hours / 24)} days ago`;
};

export const gatewayUrl = (): string => getProxyBaseUrl();

export const wildcardExample = (modelGroup: string): string => modelGroup.replaceAll("*", "<model-name>");

export const firstPlus = (items: string[]): string | undefined =>
  items.length > 1 ? `${items[0]} +${items.length - 1}` : items[0];

const MCP_AUTH: Record<string, string> = {
  none: "None",
  oauth2: "OAuth",
  bearer_token: "Bearer token",
  api_key: "API key",
};

export const mcpAuthLabel = (authType: string | null | undefined): string =>
  authType ? MCP_AUTH[authType] ?? authType : "-";

export const usefulLinkList = (
  links: Record<string, string | { url: string; index?: number }> | undefined,
): { title: string; url: string }[] =>
  Object.entries(links ?? {})
    .map(([title, value]) => ({
      title,
      url: typeof value === "string" ? value : value.url,
      index: typeof value === "string" ? 0 : value.index ?? 0,
    }))
    .sort((a, b) => a.index - b.index)
    .map(({ title, url }) => ({ title, url }));
