import {
  ADMIN_CONFIG_CREDENTIAL_KEYS,
  AUTH_TYPE,
  MCPEnvVar,
  MCPInfo,
  MCPServer,
  MCPServerCostInfo,
  MCP_OAUTH2_FLOW_INTERACTIVE,
  MCP_OAUTH2_FLOW_M2M,
  OAUTH_FLOW,
  TRANSPORT,
  isClientForwardedTokenMode,
  preservedAdminCredentials,
} from "@/components/mcp_tools/types";
import { AUTH_TYPES_REQUIRING_CREDENTIALS, reduceStaticHeaders } from "./createServerPayload";
import { TOOL_DISPLAY_NAME_PATTERN, normalizeEnvVars } from "./utils";

export type MCPAccessGroupValue = string | { readonly name?: string };

export interface EditServerFormValues {
  readonly static_headers?: unknown;
  readonly env_vars?: unknown;
  readonly credentials?: unknown;
  readonly stdio_config?: string;
  readonly env_json?: string;
  readonly command?: string;
  readonly args?: readonly string[];
  readonly allow_all_keys?: boolean;
  readonly available_on_public_internet?: boolean;
  readonly delegate_auth_to_upstream?: boolean;
  readonly oauth_passthrough?: boolean;
  readonly dcr_bridge?: boolean;
  readonly token_validation_json?: string;
  readonly mcp_access_groups?: readonly MCPAccessGroupValue[];
  readonly transport?: string;
  readonly server_name?: string;
  readonly url?: string;
  readonly alias?: string;
  readonly description?: string;
  readonly auth_type?: string;
  readonly extra_headers?: readonly string[];
  readonly disallowed_tools?: readonly string[];
  readonly oauth_flow_type?: string;
  readonly [key: string]: unknown;
}

export interface EditServerPayload {
  readonly server_id: string;
  readonly mcp_info: MCPInfo;
  readonly mcp_access_groups: readonly string[];
  readonly alias: string | undefined;
  readonly extra_headers: readonly string[];
  readonly disallowed_tools: readonly string[];
  readonly static_headers: Readonly<Record<string, string>>;
  readonly env_vars: readonly MCPEnvVar[];
  readonly allow_all_keys: boolean;
  readonly available_on_public_internet: boolean;
  readonly delegate_auth_to_upstream: boolean;
  readonly oauth_passthrough: boolean;
  readonly dcr_bridge: boolean;
  readonly stdio_config: undefined;
  readonly env_json: undefined;
  readonly command?: string;
  readonly args?: readonly string[];
  readonly env?: Readonly<Record<string, string>>;
  readonly allowed_tools?: readonly string[];
  readonly tool_name_to_display_name: Readonly<Record<string, string>> | null;
  readonly tool_name_to_description: Readonly<Record<string, string>> | null;
  readonly oauth2_flow?: string;
  readonly token_validation?: unknown;
  readonly credentials?: Readonly<Record<string, unknown>>;
  readonly [key: string]: unknown;
}

export interface EditServerUiState {
  readonly mcpServer: MCPServer;
  readonly logoUrl: string | undefined;
  readonly costConfig: MCPServerCostInfo;
  readonly allowedTools: readonly string[];
  readonly hasExistingToolAllowlist: boolean;
  readonly hasToolAllowlistInteraction: boolean;
  readonly toolNameToDisplayName: Readonly<Record<string, string>>;
  readonly toolNameToDescription: Readonly<Record<string, string>>;
  readonly removeStoredApp: boolean;
}

export type BuildEditPayloadResult =
  | { readonly kind: "ok"; readonly payload: EditServerPayload }
  | { readonly kind: "invalid_tool_display_name"; readonly displayName: string }
  | { readonly kind: "stdio_config_missing_command" }
  | { readonly kind: "invalid_stdio_json" }
  | { readonly kind: "invalid_stdio_env_json" }
  | { readonly kind: "stdio_command_required" }
  | { readonly kind: "invalid_token_validation_json" };

interface StdioFields {
  readonly command: string;
  readonly args: readonly string[];
  readonly env: Readonly<Record<string, string>>;
}

type StdioFieldsResult =
  | { readonly kind: "ok"; readonly fields: StdioFields | Record<string, never> }
  | { readonly kind: "stdio_config_missing_command" }
  | { readonly kind: "invalid_stdio_json" }
  | { readonly kind: "invalid_stdio_env_json" }
  | { readonly kind: "stdio_command_required" };

type TokenValidationResult = { readonly kind: "ok"; readonly value: unknown } | { readonly kind: "invalid" };

const assertNever = (value: never): never => {
  throw new Error(`unhandled edit payload result: ${JSON.stringify(value)}`);
};

export const editPayloadErrorMessage = (result: Exclude<BuildEditPayloadResult, { kind: "ok" }>): string => {
  switch (result.kind) {
    case "invalid_tool_display_name":
      return `Tool display name "${result.displayName}" is invalid. Only letters, digits, underscores, and hyphens are allowed (no spaces).`;
    case "stdio_config_missing_command":
      return "Stdio configuration must include a command";
    case "invalid_stdio_json":
      return "Invalid JSON in stdio configuration";
    case "invalid_stdio_env_json":
      return "Invalid JSON in stdio env configuration";
    case "stdio_command_required":
      return "Stdio transport requires a command";
    case "invalid_token_validation_json":
      return "Invalid JSON in Token Validation Rules";
    default:
      return assertNever(result);
  }
};

const toStringArgs = (raw: unknown): readonly string[] =>
  Array.isArray(raw) ? raw.map((v: unknown) => String(v)).filter((v: string) => v.trim() !== "") : [];

const toEnvRecord = (raw: unknown): Readonly<Record<string, string>> =>
  raw && typeof raw === "object" && !Array.isArray(raw)
    ? Object.fromEntries(
        Object.entries(raw)
          .filter(([k]) => k != null && String(k).trim() !== "")
          .map(([k, v]) => [String(k), v == null ? "" : String(v)]),
      )
    : {};

const buildStdioFields = (
  rawStdioConfig: string | undefined,
  rawEnvJson: string | undefined,
  rawCommand: string | undefined,
  rawArgs: readonly string[] | undefined,
): StdioFieldsResult => {
  if (rawStdioConfig) {
    try {
      const stdioConfig: unknown = JSON.parse(rawStdioConfig);
      const configRecord = stdioConfig && typeof stdioConfig === "object" ? (stdioConfig as StdioConfigShape) : null;
      const namedServers =
        configRecord?.mcpServers && typeof configRecord.mcpServers === "object" ? configRecord.mcpServers : null;
      const named = namedServers ? Object.keys(namedServers) : [];
      const actualConfig = named.length > 0 && namedServers ? namedServers[named[0]] : configRecord;
      const command = actualConfig?.command ? String(actualConfig.command) : undefined;
      if (!command) {
        return { kind: "stdio_config_missing_command" };
      }
      return {
        kind: "ok",
        fields: { command, args: toStringArgs(actualConfig?.args), env: toEnvRecord(actualConfig?.env) },
      };
    } catch {
      return { kind: "invalid_stdio_json" };
    }
  }

  const envResult = ((): Readonly<Record<string, string>> | "invalid" => {
    if (!rawEnvJson) return {};
    try {
      return toEnvRecord(JSON.parse(rawEnvJson));
    } catch {
      return "invalid";
    }
  })();
  if (envResult === "invalid") {
    return { kind: "invalid_stdio_env_json" };
  }

  const parsedCommand = rawCommand ? String(rawCommand).trim() : "";
  if (!parsedCommand) {
    return { kind: "stdio_command_required" };
  }
  return { kind: "ok", fields: { command: parsedCommand, args: toStringArgs(rawArgs), env: envResult } };
};

interface StdioConfigShape {
  readonly mcpServers?: Readonly<Record<string, StdioConfigShape>>;
  readonly command?: unknown;
  readonly args?: unknown;
  readonly env?: unknown;
}

const buildCredentials = (credentialValues: unknown): Readonly<Record<string, unknown>> | undefined => {
  if (!credentialValues || typeof credentialValues !== "object") return undefined;
  const kept = Object.entries(credentialValues).flatMap(([key, value]): readonly (readonly [string, unknown])[] => {
    if (value === undefined || value === null || value === "") {
      return value === "" && (ADMIN_CONFIG_CREDENTIAL_KEYS as readonly string[]).includes(key)
        ? [[key, null] as const]
        : [];
    }
    if (key !== "scopes") return [[key, value] as const];
    if (!Array.isArray(value)) return [];
    const filteredScopes = value.filter((scope: unknown) => scope != null && scope !== "");
    return filteredScopes.length > 0 ? [[key, filteredScopes] as const] : [];
  });
  return Object.fromEntries(kept);
};

interface CredentialsEntryInput {
  readonly authType: string | undefined;
  readonly credentials: Readonly<Record<string, unknown>> | undefined;
  readonly includeCredentials: boolean;
  readonly removeStoredApp: boolean;
}

// Explicit removal of a saved app for the client-forwarded modes, applied AFTER the filter so it
// always wins. Blank fields are the keep-existing convention (the backend merges partial credential
// updates), so removal must be an explicit-null write: encrypt skips nulls and the merge overrides
// the stored keys, returning the server to dynamic client registration.
const resolveCredentialsEntry = ({
  authType,
  credentials,
  includeCredentials,
  removeStoredApp,
}: CredentialsEntryInput): { readonly credentials?: Readonly<Record<string, unknown>> } => {
  if (removeStoredApp && isClientForwardedTokenMode(authType)) {
    return { credentials: { client_id: null, client_secret: null } };
  }
  if (includeCredentials && credentials && Object.keys(credentials).length > 0) {
    return { credentials };
  }
  return {};
};

export const buildEditServerPayload = (values: EditServerFormValues, ui: EditServerUiState): BuildEditPayloadResult => {
  const {
    mcpServer,
    logoUrl,
    costConfig,
    allowedTools,
    hasExistingToolAllowlist,
    hasToolAllowlistInteraction,
    toolNameToDisplayName,
    toolNameToDescription,
    removeStoredApp,
  } = ui;

  const invalidDisplayName = Object.entries(toolNameToDisplayName).find(
    ([, displayName]) => displayName && !TOOL_DISPLAY_NAME_PATTERN.test(displayName),
  );
  if (invalidDisplayName) {
    return { kind: "invalid_tool_display_name", displayName: String(invalidDisplayName[1]) };
  }

  const {
    static_headers: staticHeadersList,
    env_vars: envVarsList,
    credentials: credentialValues,
    stdio_config: rawStdioConfig,
    env_json: rawEnvJson,
    command: rawCommand,
    args: rawArgs,
    allow_all_keys: allowAllKeysRaw,
    available_on_public_internet: availableOnPublicInternetRaw,
    delegate_auth_to_upstream: delegateAuthToUpstreamRaw,
    oauth_passthrough: oauthPassthroughRaw,
    dcr_bridge: dcrBridgeRaw,
    token_validation_json: rawTokenValidationJson,
    ...rawRestValues
  } = values;

  const accessGroups = (rawRestValues.mcp_access_groups || []).map((g) =>
    typeof g === "string" ? g : g.name || String(g),
  );
  const staticHeaders = reduceStaticHeaders(staticHeadersList);
  const envVars = normalizeEnvVars(envVarsList);
  const credentialsPayload = buildCredentials(credentialValues);

  const stdio =
    rawRestValues.transport === "stdio"
      ? buildStdioFields(rawStdioConfig, rawEnvJson, rawCommand, rawArgs)
      : ({ kind: "ok", fields: {} } as StdioFieldsResult);
  if (stdio.kind !== "ok") {
    return stdio;
  }

  // Map "openapi" transport to "http" for the backend. Rewriting the existing key keeps its
  // position, which the payload's serialised key order depends on.
  const restValues =
    rawRestValues.transport === TRANSPORT.OPENAPI ? { ...rawRestValues, transport: "http" } : rawRestValues;

  const tokenValidation = ((): TokenValidationResult => {
    if (!rawTokenValidationJson || rawTokenValidationJson.trim() === "") return { kind: "ok", value: null };
    try {
      return { kind: "ok", value: JSON.parse(rawTokenValidationJson) };
    } catch {
      return { kind: "invalid" };
    }
  })();
  if (tokenValidation.kind === "invalid") {
    return { kind: "invalid_token_validation_json" };
  }

  const mcpInfoServerName =
    restValues.server_name ||
    restValues.url ||
    mcpServer.server_name ||
    mcpServer.url ||
    restValues.alias ||
    mcpServer.alias ||
    "unknown";

  const toolAllowlistEnforced = hasExistingToolAllowlist || hasToolAllowlistInteraction || allowedTools.length > 0;

  const extraHeaders = restValues.extra_headers || [];
  const hasAuthorizationHeader = extraHeaders.some((h) => typeof h === "string" && h.toLowerCase() === "authorization");
  const isNoneAuth = restValues.auth_type === AUTH_TYPE.NONE || restValues.auth_type == null;

  // Client-forwarded rows persist ONLY the declared app; strip any token material lingering in the
  // form (e.g. from a prior oauth2 authorize this session) so it can never reach the row.
  const submitCredentials = isClientForwardedTokenMode(restValues.auth_type)
    ? preservedAdminCredentials(credentialsPayload)
    : credentialsPayload;
  const includeCredentials = restValues.auth_type && AUTH_TYPES_REQUIRING_CREDENTIALS.includes(restValues.auth_type);

  const credentialsEntryInput: CredentialsEntryInput = {
    authType: restValues.auth_type,
    credentials: submitCredentials,
    includeCredentials: Boolean(includeCredentials),
    removeStoredApp,
  };
  const credentialsEntry = resolveCredentialsEntry(credentialsEntryInput);

  const payload: EditServerPayload = {
    ...restValues,
    ...stdio.fields,
    // Remove UI-only fields
    stdio_config: undefined,
    env_json: undefined,
    ...(mcpServer.auth_type === AUTH_TYPE.OAUTH2 && restValues.auth_type !== AUTH_TYPE.OAUTH2
      ? { issuer: null, authorization_url: null, token_url: null, registration_url: null }
      : {}),
    ...(mcpServer.auth_type === AUTH_TYPE.OAUTH2_TOKEN_EXCHANGE &&
    restValues.auth_type !== AUTH_TYPE.OAUTH2_TOKEN_EXCHANGE
      ? { token_exchange_endpoint: null, audience: null, subject_token_type: null, token_exchange_profile: null }
      : {}),
    server_id: mcpServer.server_id,
    mcp_info: {
      ...(mcpServer.mcp_info ?? {}),
      server_name: mcpInfoServerName,
      description: restValues.description,
      logo_url: logoUrl || undefined,
      mcp_server_cost_info: Object.keys(costConfig).length > 0 ? costConfig : null,
      tool_allowlist_enforced: toolAllowlistEnforced,
    },
    mcp_access_groups: accessGroups,
    alias: restValues.alias,
    // Include permission management fields
    extra_headers: extraHeaders,
    ...(toolAllowlistEnforced ? { allowed_tools: allowedTools } : {}),
    tool_name_to_display_name: Object.keys(toolNameToDisplayName).length > 0 ? toolNameToDisplayName : null,
    tool_name_to_description: Object.keys(toolNameToDescription).length > 0 ? toolNameToDescription : null,
    disallowed_tools: restValues.disallowed_tools || [],
    static_headers: staticHeaders,
    env_vars: envVars,
    allow_all_keys: Boolean(allowAllKeysRaw ?? mcpServer.allow_all_keys),
    available_on_public_internet: Boolean(availableOnPublicInternetRaw ?? mcpServer.available_on_public_internet),
    // ``delegate_auth_to_upstream`` is only honored server-side for ``auth_type=oauth2`` (PKCE
    // passthrough). The field unmounts on auth_type change, so force false for any other
    // configuration to avoid persisting a stale ``true`` that would silently re-activate.
    delegate_auth_to_upstream:
      restValues.auth_type === AUTH_TYPE.OAUTH2
        ? Boolean(delegateAuthToUpstreamRaw ?? mcpServer.delegate_auth_to_upstream)
        : false,
    // ``oauth_passthrough`` is the dedicated, non-oauth2 opt-in, honored only for ``auth_type=none``
    // servers that forward ``Authorization`` upstream. Kept separate from
    // ``delegate_auth_to_upstream`` so enabling pass-through never regresses oauth2 servers.
    oauth_passthrough:
      isNoneAuth && hasAuthorizationHeader ? Boolean(oauthPassthroughRaw ?? mcpServer.oauth_passthrough) : false,
    // ``dcr_bridge`` is only meaningful for the client-forwarded token modes. The field unmounts on
    // auth_type change, so force false otherwise rather than persisting a stale ``true``.
    dcr_bridge: isClientForwardedTokenMode(restValues.auth_type)
      ? Boolean(dcrBridgeRaw ?? mcpServer.dcr_bridge)
      : false,
    ...(restValues.auth_type === AUTH_TYPE.OAUTH2 && restValues.oauth_flow_type
      ? {
          oauth2_flow:
            restValues.oauth_flow_type === OAUTH_FLOW.M2M ? MCP_OAUTH2_FLOW_M2M : MCP_OAUTH2_FLOW_INTERACTIVE,
        }
      : {}),
    // Include token_validation when it is set (non-null) or when clearing an existing value
    ...(tokenValidation.value !== null || mcpServer.token_validation
      ? { token_validation: tokenValidation.value }
      : {}),
    ...credentialsEntry,
  };

  return { kind: "ok", payload };
};
