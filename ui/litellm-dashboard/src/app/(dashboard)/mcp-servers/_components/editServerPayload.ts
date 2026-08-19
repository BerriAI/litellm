import {
  ADMIN_CONFIG_CREDENTIAL_KEYS,
  AUTH_TYPE,
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
  | { readonly kind: "ok"; readonly payload: Record<string, any> }
  | { readonly kind: "invalid_tool_display_name"; readonly displayName: string }
  | { readonly kind: "stdio_config_missing_command" }
  | { readonly kind: "invalid_stdio_json" }
  | { readonly kind: "invalid_stdio_env_json" }
  | { readonly kind: "stdio_command_required" }
  | { readonly kind: "invalid_token_validation_json" };

type StdioFieldsResult =
  | { readonly kind: "ok"; readonly fields: Record<string, any> }
  | { readonly kind: "stdio_config_missing_command" }
  | { readonly kind: "invalid_stdio_json" }
  | { readonly kind: "invalid_stdio_env_json" }
  | { readonly kind: "stdio_command_required" };

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

const toStringArgs = (raw: unknown): string[] =>
  Array.isArray(raw) ? raw.map((v: any) => String(v)).filter((v: string) => v.trim() !== "") : [];

const toEnvRecord = (raw: unknown): Record<string, string> =>
  raw && typeof raw === "object" && !Array.isArray(raw)
    ? Object.entries(raw).reduce((acc: Record<string, string>, [k, v]) => {
        if (k == null || String(k).trim() === "") return acc;
        acc[String(k)] = v == null ? "" : String(v);
        return acc;
      }, {})
    : {};

const buildStdioFields = (
  rawStdioConfig: unknown,
  rawEnvJson: unknown,
  rawCommand: unknown,
  rawArgs: unknown,
): StdioFieldsResult => {
  if (rawStdioConfig) {
    try {
      const stdioConfig = JSON.parse(rawStdioConfig as string);
      const named =
        stdioConfig?.mcpServers && typeof stdioConfig.mcpServers === "object"
          ? Object.keys(stdioConfig.mcpServers)
          : [];
      const actualConfig = named.length > 0 ? stdioConfig.mcpServers[named[0]] : stdioConfig;
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

  const envResult = ((): Record<string, string> | "invalid" => {
    if (!rawEnvJson) return {};
    try {
      return toEnvRecord(JSON.parse(rawEnvJson as string));
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

const buildCredentials = (credentialValues: unknown): Record<string, any> | undefined =>
  credentialValues && typeof credentialValues === "object"
    ? Object.entries(credentialValues).reduce((acc: Record<string, any>, [key, value]) => {
        if (value === undefined || value === null || value === "") {
          if (value === "" && (ADMIN_CONFIG_CREDENTIAL_KEYS as readonly string[]).includes(key)) {
            acc[key] = null;
          }
          return acc;
        }
        if (key === "scopes") {
          if (Array.isArray(value)) {
            const filteredScopes = value.filter((scope) => scope != null && scope !== "");
            if (filteredScopes.length > 0) {
              acc[key] = filteredScopes;
            }
          }
        } else {
          acc[key] = value;
        }
        return acc;
      }, {})
    : undefined;

export const buildEditServerPayload = (values: Record<string, any>, ui: EditServerUiState): BuildEditPayloadResult => {
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

  const accessGroups = (rawRestValues.mcp_access_groups || []).map((g: any) =>
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

  const tokenValidation = ((): Record<string, any> | null | "invalid" => {
    if (!rawTokenValidationJson || rawTokenValidationJson.trim() === "") return null;
    try {
      return JSON.parse(rawTokenValidationJson);
    } catch {
      return "invalid";
    }
  })();
  if (tokenValidation === "invalid") {
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

  const payload: Record<string, any> = {
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
    extra_headers: restValues.extra_headers || [],
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
    oauth_passthrough: (() => {
      const isNoneAuth = restValues.auth_type === AUTH_TYPE.NONE || restValues.auth_type == null;
      const extraHeaders = Array.isArray(restValues.extra_headers) ? restValues.extra_headers : [];
      const hasAuthorizationHeader = extraHeaders.some(
        (h: unknown) => typeof h === "string" && h.toLowerCase() === "authorization",
      );
      return isNoneAuth && hasAuthorizationHeader ? Boolean(oauthPassthroughRaw ?? mcpServer.oauth_passthrough) : false;
    })(),
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
    ...(tokenValidation !== null || mcpServer.token_validation ? { token_validation: tokenValidation } : {}),
  };

  const includeCredentials = restValues.auth_type && AUTH_TYPES_REQUIRING_CREDENTIALS.includes(restValues.auth_type);

  // Client-forwarded rows persist ONLY the declared app; strip any token material lingering in the
  // form (e.g. from a prior oauth2 authorize this session) so it can never reach the row.
  const submitCredentials = isClientForwardedTokenMode(restValues.auth_type)
    ? preservedAdminCredentials(credentialsPayload)
    : credentialsPayload;

  if (includeCredentials && submitCredentials && Object.keys(submitCredentials).length > 0) {
    payload.credentials = submitCredentials;
  }

  // Explicit removal of a saved app for the client-forwarded modes, applied AFTER the filter so it
  // always wins. Blank fields are the keep-existing convention (the backend merges partial
  // credential updates), so removal must be an explicit-null write: encrypt skips nulls and the
  // merge overrides the stored keys, returning the server to dynamic client registration.
  if (removeStoredApp && isClientForwardedTokenMode(restValues.auth_type)) {
    payload.credentials = { client_id: null, client_secret: null };
  }

  return { kind: "ok", payload };
};
