import { ALL_PROXY_MCP_SERVERS_SENTINEL, NO_MCP_SERVERS_SENTINEL } from "@/components/mcp_tools/constants";
import type { ObjectPermission } from "@/components/object_permission_types";

export interface ModelLimitRow {
  readonly model: string;
  readonly rpm: number | null;
  readonly tpm: number | null;
}

export interface TagLimitRow {
  readonly tag: string;
  readonly rpm: number;
}

export interface KeyRateLimits {
  readonly rpm: number | null;
  readonly tpm: number | null;
  readonly tpd: number | null;
  readonly parallel: number | null;
  readonly perModel: readonly ModelLimitRow[];
  readonly perTag: readonly TagLimitRow[];
}

export interface RateLimitSource {
  readonly rpm_limit?: number | null;
  readonly tpm_limit?: number | null;
  readonly tpd_limit?: number | null;
  readonly max_parallel_requests?: number | null;
  readonly metadata?: Readonly<Record<string, unknown>> | null;
}

export type McpGrant =
  | { readonly kind: "none" }
  | { readonly kind: "blocked" }
  | { readonly kind: "allProxy" }
  | {
      readonly kind: "list";
      readonly servers: readonly string[];
      readonly accessGroups: readonly string[];
      readonly toolsets: readonly string[];
    };

export interface KeyAccess {
  readonly mcp: McpGrant;
  readonly mcpToolPermissions: Readonly<Record<string, readonly string[]>>;
  readonly vectorStores: readonly string[];
  readonly agents: readonly string[];
  readonly agentAccessGroups: readonly string[];
  readonly searchTools: readonly string[];
  readonly skills: readonly string[];
}

export type UnsetSection =
  | "rateLimits"
  | "mcpServers"
  | "vectorStores"
  | "agents"
  | "skills"
  | "guardrails"
  | "policies"
  | "logging";

export interface UnsetSource {
  readonly rateLimits: KeyRateLimits;
  readonly access: KeyAccess;
  readonly guardrails: readonly string[];
  readonly policies: readonly string[];
  readonly loggingCount: number;
  readonly disabledCallbacks: readonly string[];
}

const isRecord = (value: unknown): value is Readonly<Record<string, unknown>> =>
  typeof value === "object" && value !== null && !Array.isArray(value);

const finiteNumber = (value: unknown): number | null =>
  typeof value === "number" && Number.isFinite(value) ? value : null;

const numberEntries = (value: unknown): readonly (readonly [string, number])[] =>
  isRecord(value)
    ? Object.entries(value).flatMap(([name, limit]) => {
        const parsed = finiteNumber(limit);
        return parsed === null ? [] : [[name, parsed] as const];
      })
    : [];

export const stringList = (value: unknown): readonly string[] =>
  Array.isArray(value) ? value.filter((item): item is string => typeof item === "string" && item.length > 0) : [];

const stringListMap = (value: unknown): Readonly<Record<string, readonly string[]>> =>
  isRecord(value) ? Object.fromEntries(Object.entries(value).map(([name, list]) => [name, stringList(list)])) : {};

export const keyRateLimits = (key: RateLimitSource): KeyRateLimits => {
  const rpmByModel = new Map(numberEntries(key.metadata?.model_rpm_limit));
  const tpmByModel = new Map(numberEntries(key.metadata?.model_tpm_limit));
  const models = [...new Set([...rpmByModel.keys(), ...tpmByModel.keys()])];
  return {
    rpm: finiteNumber(key.rpm_limit),
    tpm: finiteNumber(key.tpm_limit),
    tpd: finiteNumber(key.tpd_limit),
    parallel: finiteNumber(key.max_parallel_requests),
    perModel: models.map((model) => ({
      model,
      rpm: rpmByModel.get(model) ?? null,
      tpm: tpmByModel.get(model) ?? null,
    })),
    perTag: numberEntries(key.metadata?.tag_rpm_limit).map(([tag, rpm]) => ({ tag, rpm })),
  };
};

export const hasRateLimits = (limits: KeyRateLimits): boolean =>
  [limits.rpm, limits.tpm, limits.tpd, limits.parallel].some((limit) => limit !== null) ||
  limits.perModel.length > 0 ||
  limits.perTag.length > 0;

const mcpGrant = (permission: ObjectPermission | null | undefined): McpGrant => {
  const servers = stringList(permission?.mcp_servers);
  if (servers.includes(NO_MCP_SERVERS_SENTINEL)) return { kind: "blocked" };
  if (servers.includes(ALL_PROXY_MCP_SERVERS_SENTINEL)) return { kind: "allProxy" };
  const accessGroups = stringList(permission?.mcp_access_groups);
  const toolsets = stringList(permission?.mcp_toolsets);
  if (servers.length === 0 && accessGroups.length === 0 && toolsets.length === 0) return { kind: "none" };
  return { kind: "list", servers, accessGroups, toolsets };
};

export const keyAccess = (permission: ObjectPermission | null | undefined): KeyAccess => ({
  mcp: mcpGrant(permission),
  mcpToolPermissions: stringListMap(permission?.mcp_tool_permissions),
  vectorStores: stringList(permission?.vector_stores),
  agents: stringList(permission?.agents),
  agentAccessGroups: stringList(permission?.agent_access_groups),
  searchTools: stringList(permission?.search_tools),
  skills: stringList(permission?.skills),
});

export const unsetSections = (source: UnsetSource): readonly UnsetSection[] => {
  const { access } = source;
  const noGuardrails = source.guardrails.length === 0 && source.policies.length === 0;
  const checks: readonly (readonly [UnsetSection, boolean])[] = [
    ["rateLimits", !hasRateLimits(source.rateLimits)],
    ["mcpServers", access.mcp.kind === "none"],
    ["vectorStores", access.vectorStores.length === 0],
    ["agents", access.agents.length === 0 && access.agentAccessGroups.length === 0],
    ["skills", access.skills.length === 0],
    ["guardrails", noGuardrails],
    ["policies", noGuardrails],
    ["logging", source.loggingCount === 0 && source.disabledCallbacks.length === 0],
  ];
  return checks.filter(([, unset]) => unset).map(([section]) => section);
};
