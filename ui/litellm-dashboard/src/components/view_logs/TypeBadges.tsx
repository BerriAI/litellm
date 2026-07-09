/**
 * Compact type-indicator badges for LLM, Agent, MCP, and Relay log entries.
 * Used in the request logs table and session type column.
 */
import { Cable, Monitor } from "lucide-react";
import { resolveLogoSrc } from "@/lib/assetPaths";

export const SparkleIcon = ({ size = 12 }: { size?: number }) => (
  <svg
    width={size}
    height={size}
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    strokeLinecap="round"
    strokeLinejoin="round"
    className="shrink-0 text-gray-400"
  >
    <path d="M12 3l1.912 5.813a2 2 0 0 0 1.275 1.275L21 12l-5.813 1.912a2 2 0 0 0-1.275 1.275L12 21l-1.912-5.813a2 2 0 0 0-1.275-1.275L3 12l5.813-1.912a2 2 0 0 0 1.275-1.275L12 3z" />
  </svg>
);

export const WrenchIcon = ({ size = 10 }: { size?: number }) => (
  <svg
    width={size}
    height={size}
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    strokeLinecap="round"
    strokeLinejoin="round"
    className="shrink-0"
  >
    <path d="M14.7 6.3a1 1 0 0 0 0 1.4l1.6 1.6a1 1 0 0 0 1.4 0l3.77-3.77a6 6 0 0 1-7.94 7.94l-6.91 6.91a2.12 2.12 0 0 1-3-3l6.91-6.91a6 6 0 0 1 7.94-7.94l-3.76 3.76z" />
  </svg>
);

/** Agent/bot icon for A2A and agent call types (Lucide Bot-style). */
export const AgentIcon = ({ size = 12 }: { size?: number }) => (
  <svg
    width={size}
    height={size}
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    strokeLinecap="round"
    strokeLinejoin="round"
    className="shrink-0"
  >
    <path d="M12 8V4H8" />
    <rect width="16" height="12" x="4" y="8" rx="2" />
    <path d="M2 14h2" />
    <path d="M20 14h2" />
    <path d="M15 13v2" />
    <path d="M9 13v2" />
  </svg>
);

export const RelayIcon = ({ size = 12 }: { size?: number }) => <Cable size={size} className="flex-shrink-0" />;

export const LlmBadge = ({ count }: { count?: number }) => (
  <span className="inline-flex items-center gap-1 px-2 py-0.5 bg-blue-50 text-blue-700 border border-blue-200 rounded-full text-[11px] font-medium whitespace-nowrap">
    <SparkleIcon />
    {count != null ? count : "LLM"}
  </span>
);

export const McpBadge = ({ count }: { count?: number }) => (
  <span className="inline-flex items-center gap-1 px-2 py-0.5 bg-amber-50 text-amber-700 border border-amber-200 rounded-full text-[11px] font-medium whitespace-nowrap">
    <WrenchIcon />
    {count != null ? count : "MCP"}
  </span>
);

export const AgentBadge = ({ count }: { count?: number }) => (
  <span className="inline-flex items-center gap-1 px-2 py-0.5 bg-violet-50 text-violet-700 border border-violet-200 rounded-full text-[11px] font-medium whitespace-nowrap">
    <AgentIcon />
    {count != null ? count : "Agent"}
  </span>
);

export const RelayBadge = ({ count }: { count?: number }) => (
  <span className="inline-flex items-center gap-1 px-2 py-0.5 bg-emerald-50 text-emerald-700 border border-emerald-200 rounded-full text-[11px] font-medium whitespace-nowrap">
    <RelayIcon />
    {count != null ? count : "litellm-relay"}
  </span>
);

type RelaySourceLike = {
  metadata?: Record<string, any>;
  model?: string;
  request_tags?: Record<string, any> | string[] | string;
};

const RELAY_SOURCE_LABELS: Record<string, string> = {
  notion: "Notion",
  codex: "Codex",
};

const RELAY_SOURCE_LOGOS: Record<string, string> = {
  notion: "/ui/assets/logos/notion.svg",
  codex: "/ui/assets/logos/openai_small.svg",
};

const normalizeRelaySource = (value: unknown): string | undefined => {
  if (typeof value !== "string") return undefined;
  const normalized = value.trim().toLowerCase();
  if (!normalized) return undefined;
  if (normalized === "litellm-relay") return undefined;
  if (normalized.includes("notion")) return "notion";
  if (normalized.includes("codex")) return "codex";
  return normalized.replace(/-ai$/, "");
};

const getTagSource = (requestTags: RelaySourceLike["request_tags"]): string | undefined => {
  if (Array.isArray(requestTags)) {
    return requestTags.map(normalizeRelaySource).find(Boolean);
  }
  if (typeof requestTags === "string") {
    try {
      const parsed = JSON.parse(requestTags);
      return getTagSource(parsed);
    } catch {
      return normalizeRelaySource(requestTags);
    }
  }
  return undefined;
};

export const getRelaySource = (entry: RelaySourceLike): string => {
  return (
    normalizeRelaySource(entry.metadata?.app) ||
    normalizeRelaySource(entry.metadata?.relay_app) ||
    normalizeRelaySource(entry.metadata?.shadow_source) ||
    normalizeRelaySource(entry.metadata?.host) ||
    getTagSource(entry.request_tags) ||
    normalizeRelaySource(entry.model) ||
    "unknown"
  );
};

export const getRelaySourceLabel = (source: string) => {
  return RELAY_SOURCE_LABELS[source] || source.charAt(0).toUpperCase() + source.slice(1);
};

export const RelaySourceLogo = ({ source, size = 16 }: { source: string; size?: number }) => {
  const logo = RELAY_SOURCE_LOGOS[source];
  if (logo) {
    return (
      <span aria-label={`${getRelaySourceLabel(source)} logo`} role="img" className="inline-flex">
        <img src={resolveLogoSrc(logo)} alt="" className="flex-shrink-0" style={{ width: size, height: size }} />
      </span>
    );
  }
  return (
    <span aria-label={`${getRelaySourceLabel(source)} logo`} role="img" className="inline-flex">
      <Monitor size={size} className="flex-shrink-0 text-slate-500" aria-hidden="true" />
    </span>
  );
};

export const RelaySourceBadge = ({ source }: { source: string }) => (
  <span className="inline-flex items-center gap-1.5 px-2 py-0.5 bg-slate-50 text-slate-700 border border-slate-200 rounded-full text-[11px] font-medium whitespace-nowrap">
    <RelaySourceLogo source={source} />
    {getRelaySourceLabel(source)}
  </span>
);

export const RelayTypeBadge = ({ source }: { source: string }) => (
  <span className="inline-flex items-center gap-1.5 whitespace-nowrap">
    <RelayBadge />
    <RelaySourceBadge source={source} />
  </span>
);
