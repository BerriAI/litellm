import { EndpointType } from "@/components/chat_ui/mode_endpoint_mapping";
import type { KeyValuePair } from "@/components/key_value_input";
import { parseStoredHeaderPairs } from "@/components/llm_calls/request_headers";
import { OPEN_AI_VOICES, type OpenAIVoice } from "../chat_ui/chatConstants";

export type KeySource = "session" | "custom";

export type ToolLimits = Readonly<Record<string, readonly string[]>>;

export type PlaygroundSettings = Readonly<{
  keySource: KeySource;
  virtualKey: string;
  proxyBaseUrl: string;
  endpoint: EndpointType;
  model: string | null;
  agent: string | null;
  voice: OpenAIVoice;
  tags: readonly string[];
  headerPairs: readonly KeyValuePair[];
  vectorStores: readonly string[];
  guardrails: readonly string[];
  policies: readonly string[];
  mcpSelection: readonly string[];
  toolLimits: ToolLimits;
  directTool: string | null;
  stream: boolean;
  simulateFallback: boolean;
  temperature: number | null;
  maxTokens: number | null;
}>;

export type StoredValue = Readonly<{ key: string; value: string | null; secure: boolean }>;

export type SettingsStorage = Readonly<{
  read: (key: string) => string | null;
  readSecure: (key: string) => string | null;
}>;

export const ALL_MCP_SERVERS = "__all__";
export const TOOLSET_PREFIX = "toolset:";

const ENDPOINTS: readonly string[] = Object.values(EndpointType);
const VOICES: readonly string[] = Object.values(OPEN_AI_VOICES);

const parseJson = (raw: string | null): unknown => {
  if (raw === null) return undefined;
  try {
    return JSON.parse(raw);
  } catch {
    return undefined;
  }
};

const isStringList = (value: unknown): value is readonly string[] =>
  Array.isArray(value) && value.every((item) => typeof item === "string");

const stringList = (raw: string | null): readonly string[] => {
  const parsed = parseJson(raw);
  return isStringList(parsed) ? parsed : [];
};

const toolLimits = (raw: string | null): ToolLimits => {
  const parsed = parseJson(raw);
  if (typeof parsed !== "object" || parsed === null || Array.isArray(parsed)) return {};
  return Object.fromEntries(
    Object.entries(parsed).filter((entry): entry is [string, string[]] => isStringList(entry[1])),
  );
};

const endpoint = (raw: string | null): EndpointType =>
  ENDPOINTS.includes(raw ?? "") ? (raw as EndpointType) : EndpointType.CHAT;

const voice = (raw: string | null): OpenAIVoice => {
  const parsed = parseJson(raw);
  const candidate = typeof parsed === "string" ? parsed : raw;
  return VOICES.includes(candidate ?? "") ? (candidate as OpenAIVoice) : OPEN_AI_VOICES.ALLOY;
};

const keySource = (raw: string | null, keysDisabled: boolean): KeySource => {
  if (keysDisabled) return "custom";
  const parsed = parseJson(raw);
  return parsed === "custom" ? "custom" : "session";
};

export const defaultSettings = (keysDisabled: boolean): PlaygroundSettings => ({
  keySource: keysDisabled ? "custom" : "session",
  virtualKey: "",
  proxyBaseUrl: "",
  endpoint: EndpointType.CHAT,
  model: null,
  agent: null,
  voice: OPEN_AI_VOICES.ALLOY,
  tags: [],
  headerPairs: [],
  vectorStores: [],
  guardrails: [],
  policies: [],
  mcpSelection: [],
  toolLimits: {},
  directTool: null,
  stream: true,
  simulateFallback: false,
  temperature: null,
  maxTokens: null,
});

export const loadSettings = (storage: SettingsStorage, keysDisabled: boolean): PlaygroundSettings => ({
  ...defaultSettings(keysDisabled),
  keySource: keySource(storage.readSecure("apiKeySource"), keysDisabled),
  virtualKey: storage.readSecure("apiKey") ?? "",
  headerPairs: parseStoredHeaderPairs(storage.readSecure("customHeaders")),
  proxyBaseUrl: storage.read("customProxyBaseUrl") ?? "",
  endpoint: endpoint(storage.read("endpointType")),
  model: storage.read("selectedModel"),
  voice: voice(storage.read("selectedVoice")),
  tags: stringList(storage.read("selectedTags")),
  vectorStores: stringList(storage.read("selectedVectorStores")),
  guardrails: stringList(storage.read("selectedGuardrails")),
  policies: stringList(storage.read("selectedPolicies")),
  mcpSelection: stringList(storage.read("selectedMCPServers")),
  toolLimits: toolLimits(storage.read("mcpServerToolRestrictions")),
  stream: storage.read("streamingEnabled") !== "false",
});

export const storedValues = (settings: PlaygroundSettings): readonly StoredValue[] => [
  { key: "apiKeySource", value: JSON.stringify(settings.keySource), secure: true },
  { key: "apiKey", value: settings.virtualKey, secure: true },
  { key: "customHeaders", value: JSON.stringify(settings.headerPairs), secure: true },
  { key: "customProxyBaseUrl", value: settings.proxyBaseUrl || null, secure: false },
  { key: "endpointType", value: settings.endpoint, secure: false },
  { key: "selectedModel", value: settings.model, secure: false },
  { key: "selectedVoice", value: settings.voice, secure: false },
  { key: "selectedTags", value: JSON.stringify(settings.tags), secure: false },
  { key: "selectedVectorStores", value: JSON.stringify(settings.vectorStores), secure: false },
  { key: "selectedGuardrails", value: JSON.stringify(settings.guardrails), secure: false },
  { key: "selectedPolicies", value: JSON.stringify(settings.policies), secure: false },
  { key: "selectedMCPServers", value: JSON.stringify(settings.mcpSelection), secure: false },
  { key: "mcpServerToolRestrictions", value: JSON.stringify(settings.toolLimits), secure: false },
  { key: "streamingEnabled", value: String(settings.stream), secure: false },
];

const singleServer = (selection: readonly string[]): readonly string[] =>
  selection.length === 1 && selection[0] !== ALL_MCP_SERVERS ? selection : [];

export const withEndpoint = (settings: PlaygroundSettings, next: EndpointType): PlaygroundSettings => ({
  ...settings,
  endpoint: next,
  model: null,
  agent: null,
  directTool: null,
  mcpSelection: next === EndpointType.MCP ? singleServer(settings.mcpSelection) : settings.mcpSelection,
});

export const toggleMcpItem = (selection: readonly string[], item: string, on: boolean): readonly string[] => {
  if (!on) return selection.filter((value) => value !== item);
  if (item === ALL_MCP_SERVERS) return [ALL_MCP_SERVERS];
  return [...selection.filter((value) => value !== ALL_MCP_SERVERS && value !== item), item];
};

export const withoutStaleLimits = (limits: ToolLimits, selection: readonly string[]): ToolLimits =>
  Object.fromEntries(Object.entries(limits).filter(([server]) => selection.includes(server)));

export const withReachableMcp = (
  settings: PlaygroundSettings,
  reachable: ReadonlySet<string> | null,
): PlaygroundSettings => {
  if (!reachable) return settings;
  const mcpSelection = settings.mcpSelection.filter((item) => item === ALL_MCP_SERVERS || reachable.has(item));
  if (mcpSelection.length === settings.mcpSelection.length) return settings;
  return { ...settings, mcpSelection, toolLimits: withoutStaleLimits(settings.toolLimits, mcpSelection) };
};

export const allowedTools = (limits: ToolLimits, server: string, tools: readonly string[]): readonly string[] => {
  const limit = limits[server] ?? [];
  return limit.length === 0 ? tools : tools.filter((tool) => limit.includes(tool));
};

export type ToolScope = Readonly<{ server: string; tools: readonly string[] }>;

export const toggleToolLimit = (
  limits: ToolLimits,
  { server, tools }: ToolScope,
  tool: string,
  on: boolean,
): ToolLimits => {
  const current = allowedTools(limits, server, tools);
  const next = on
    ? tools.filter((name) => name === tool || current.includes(name))
    : current.filter((name) => name !== tool);
  if (next.length === 0) return limits;
  return { ...limits, [server]: next.length === tools.length ? [] : next };
};
