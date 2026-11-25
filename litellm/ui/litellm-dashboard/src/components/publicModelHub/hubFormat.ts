import type { ParseKeys } from "i18next";

import { getProxyBaseUrl } from "@/components/networking";
import type { AgentCard, ModelGroupInfo } from "@/components/PublicModelHubTableColumns";
import { getProviderLogoAndName } from "@/components/provider_info_helpers";
import { displayMode } from "@/lib/decisionModels";
import i18n from "@/i18n/i18n";

export const SECTION_TITLE = "text-[15px] leading-[22px] font-semibold";

const MODE_LABELS: Record<string, ParseKeys> = {
  chat: "publicModelHub.format.modes.chat",
  completion: "publicModelHub.format.modes.completion",
  embedding: "publicModelHub.format.modes.embedding",
  rerank: "publicModelHub.format.modes.rerank",
  image_generation: "publicModelHub.format.modes.imageGeneration",
  image_edit: "publicModelHub.format.modes.imageEdit",
  audio_speech: "publicModelHub.format.modes.audioSpeech",
  audio_transcription: "publicModelHub.format.modes.audioTranscription",
  video_generation: "publicModelHub.format.modes.videoGeneration",
  realtime: "publicModelHub.format.modes.realtime",
  responses: "publicModelHub.format.modes.responses",
  batch: "publicModelHub.format.modes.batch",
  ocr: "publicModelHub.format.modes.ocr",
  decisions: "publicModelHub.format.modes.decisions",
};

const sentenceCase = (text: string): string => text.replaceAll("_", " ").replace(/^./, (c) => c.toUpperCase());

export const modeLabel = (mode: string | null | undefined): string => {
  const shown = displayMode(mode);
  return shown ? (Object.hasOwn(MODE_LABELS, shown) && i18n.t(MODE_LABELS[shown])) || sentenceCase(shown) : "-";
};

export const formatTokens = (tokens: number | null | undefined): string => {
  if (!tokens) return "-";
  if (tokens >= 1_000_000) return `${+(tokens / 1_000_000).toFixed(1)}M`;
  if (tokens >= 1_000) return `${Math.round(tokens / 1_000)}K`;
  return String(tokens);
};

const PER_MILLION_FORMAT: Intl.NumberFormatOptions = {
  style: "currency",
  currency: "USD",
  minimumFractionDigits: 2,
  maximumFractionDigits: 4,
  useGrouping: false,
};

const PER_MILLION = new Intl.NumberFormat("en-US", PER_MILLION_FORMAT);

export const formatPerMillion = (cost: number | null | undefined): string => {
  if (cost === 0) return i18n.t("publicModelHub.format.free");
  return cost ? PER_MILLION.format(cost * 1_000_000) : "-";
};

export const providerName = (provider: string): string =>
  getProviderLogoAndName(provider).displayName.replace(/\s*\(.*\)$/, "");

export const providerLogo = (provider: string): string => getProviderLogoAndName(provider).logo;

const CAPABILITY_LABELS: Record<string, ParseKeys> = {
  supports_vision: "publicModelHub.format.capabilities.vision",
  supports_function_calling: "publicModelHub.format.capabilities.functionCalling",
  supports_reasoning: "publicModelHub.format.capabilities.reasoning",
  supports_web_search: "publicModelHub.format.capabilities.webSearch",
  supports_parallel_function_calling: "publicModelHub.format.capabilities.parallelFunctionCalling",
  supports_url_context: "publicModelHub.format.capabilities.urlContext",
  supports_fast_mode: "publicModelHub.format.capabilities.fastMode",
};

const CAPABILITY_ORDER = Object.keys(CAPABILITY_LABELS);

const capabilityRank = (key: string): number => {
  const index = CAPABILITY_ORDER.indexOf(key);
  return index === -1 ? CAPABILITY_ORDER.length : index;
};

export const capabilityLabel = (key: string): string =>
  Object.hasOwn(CAPABILITY_LABELS, key) ? i18n.t(CAPABILITY_LABELS[key]) : sentenceCase(key.replace(/^supports_/, ""));

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
    .map(([key]) => sentenceCase(key.replace(/([a-z])([A-Z])/g, "$1_$2").toLowerCase()));

export type HealthTone = "success" | "error" | "neutral";

const HEALTH: Record<string, { tone: HealthTone; label: ParseKeys }> = {
  healthy: { tone: "success", label: "publicModelHub.format.health.healthy" },
  unhealthy: { tone: "error", label: "publicModelHub.format.health.unhealthy" },
};

export const healthOf = (model: ModelGroupInfo): { tone: HealthTone; label: string } => {
  const status = model.health_status ?? "";
  return Object.hasOwn(HEALTH, status)
    ? { tone: HEALTH[status].tone, label: i18n.t(HEALTH[status].label) }
    : { tone: "neutral", label: i18n.t("publicModelHub.format.notChecked") };
};

export const relativeTime = (iso: string | null | undefined, now: number = Date.now()): string => {
  const then = iso ? new Date(iso).getTime() : NaN;
  if (Number.isNaN(then)) return "-";
  const minutes = Math.max(0, Math.round((now - then) / 60_000));
  if (minutes < 1) return i18n.t("publicModelHub.format.relativeTime.justNow");
  if (minutes < 60) return i18n.t("publicModelHub.format.relativeTime.minutes", { minutes });
  const hours = Math.round(minutes / 60);
  if (hours < 24) return i18n.t("publicModelHub.format.relativeTime.hours", { hours });
  const days = Math.round(hours / 24);
  return i18n.t("publicModelHub.format.relativeTime.days", { count: days });
};

export const gatewayUrl = (): string => getProxyBaseUrl();

export const wildcardExample = (modelGroup: string): string => modelGroup.replaceAll("*", "<model-name>");

export const firstPlus = (items: string[]): string | undefined =>
  items.length > 1 ? `${items[0]} +${items.length - 1}` : items[0];

const MCP_AUTH: Record<string, ParseKeys> = {
  none: "publicModelHub.format.mcpAuth.none",
  oauth2: "publicModelHub.format.mcpAuth.oauth2",
  bearer_token: "publicModelHub.format.mcpAuth.bearerToken",
  api_key: "publicModelHub.format.mcpAuth.apiKey",
};

export const mcpAuthLabel = (authType: string | null | undefined): string =>
  authType ? (Object.hasOwn(MCP_AUTH, authType) && i18n.t(MCP_AUTH[authType])) || authType : "-";

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
