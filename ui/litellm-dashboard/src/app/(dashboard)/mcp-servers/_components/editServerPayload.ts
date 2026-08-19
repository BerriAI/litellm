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
import { TOOL_DISPLAY_NAME_PATTERN, normalizeEnvVars } from "./utils";
import { AUTH_TYPES_REQUIRING_CREDENTIALS, reduceStaticHeaders } from "./createServerPayload";

export interface EditServerUiState {
  readonly costConfig: MCPServerCostInfo;
  readonly allowedTools: readonly string[];
  readonly hasExistingToolAllowlist: boolean;
  readonly hasToolAllowlistInteraction: boolean;
  readonly toolNameToDisplayName: Readonly<Record<string, string>>;
  readonly toolNameToDescription: Readonly<Record<string, string>>;
  readonly logoUrl: string | undefined;
  readonly removeStoredApp: boolean;
}

export type BuildEditPayloadResult =
  | { readonly kind: "ok"; readonly payload: Record<string, unknown> }
  | { readonly kind: "invalid_tool_display_name"; readonly displayName: string }
  | { readonly kind: "invalid_stdio_json" }
  | { readonly kind: "stdio_config_missing_command" }
  | { readonly kind: "invalid_stdio_env_json" }
  | { readonly kind: "stdio_missing_command" }
  | { readonly kind: "invalid_token_validation_json" };

type StdioFieldsResult =
  | { readonly kind: "ok"; readonly fields: Record<string, unknown> }
  | { readonly kind: "invalid_stdio_json" }
  | { readonly kind: "stdio_config_missing_command" }
  | { readonly kind: "invalid_stdio_env_json" }
  | { readonly kind: "stdio_missing_command" };

type JsonParseResult = { readonly kind: "ok"; readonly value: unknown } | { readonly kind: "invalid" };

const tryParseJson = (raw: string): JsonParseResult => {
  try {
    return { kind: "ok", value: JSON.parse(raw) };
  } catch {
    return { kind: "invalid" };
  }
};

const normalizeStdioArgs = (args: unknown): string[] =>
  Array.isArray(args) ? args.map((v: unknown) => String(v)).filter((v: string) => v.trim() !== "") : [];

const normalizeStdioEnv = (env: unknown): Record<string, string> =>
  env && typeof env === "object" && !Array.isArray(env)
    ? Object.entries(env as Record<string, unknown>).reduce((acc: Record<string, string>, [k, v]) => {
        if (k == null || String(k).trim() === "") return acc;
        acc[String(k)] = v == null ? "" : String(v);
        return acc;
      }, {})
    : {};

const unwrapStdioConfig = (parsed: unknown): Record<string, unknown> => {
  const config = parsed as Record<string, unknown> | null;
  const nested = config?.mcpServers;
  if (!nested || typeof nested !== "object") return (config ?? {}) as Record<string, unknown>;
  const names = Object.keys(nested as Record<string, unknown>);
  if (names.length === 0) return config as Record<string, unknown>;
  return (nested as Record<string, unknown>)[names[0]] as Record<string, unknown>;
};

const buildStdioFields = (
  rawStdioConfig: unknown,
  rawEnvJson: unknown,
  rawCommand: unknown,
  rawArgs: unknown,
): StdioFieldsResult => {
  if (rawStdioConfig) {
    const parsed = tryParseJson(rawStdioConfig as string);
    if (parsed.kind === "invalid") return { kind: "invalid_stdio_json" };

    const config = unwrapStdioConfig(parsed.value);
    const command = config?.command ? String(config.command) : undefined;
    if (!command) return { kind: "stdio_config_missing_command" };

    return {
      kind: "ok",
      fields: { command, args: normalizeStdioArgs(config?.args), env: normalizeStdioEnv(config?.env) },
    };
  }

  const envResult: JsonParseResult = rawEnvJson ? tryParseJson(rawEnvJson as string) : { kind: "ok", value: null };
  if (envResult.kind === "invalid") return { kind: "invalid_stdio_env_json" };

  const command = rawCommand ? String(rawCommand).trim() : "";
  if (!command) return { kind: "stdio_missing_command" };

  return {
    kind: "ok",
    fields: { command, args: normalizeStdioArgs(rawArgs), env: normalizeStdioEnv(envResult.value) },
  };
};

const filterCredentials = (credentialValues: unknown): Record<string, unknown> | undefined => {
  if (!credentialValues || typeof credentialValues !== "object") return undefined;
  return Object.entries(credentialValues as Record<string, unknown>).reduce(
    (acc: Record<string, unknown>, [key, value]) => {
      if (value === undefined || value === null || value === "") {
        // Blank is the keep-existing convention, except for the admin-config keys, where an
        // explicit null is how the backend is told to clear a previously stored value.
        if (value === "" && (ADMIN_CONFIG_CREDENTIAL_KEYS as readonly string[]).includes(key)) {
          acc[key] = null;
        }
        return acc;
      }
      if (key === "scopes") {
        if (Array.isArray(value)) {
          const filteredScopes = value.filter((scope) => scope != null && scope !== "");
          if (filteredScopes.length > 0) acc[key] = filteredScopes;
        }
      } else {
        acc[key] = value;
      }
      return acc;
    },
    {},
  );
};

const firstInvalidToolDisplayName = (toolNameToDisplayName: Readonly<Record<string, string>>): string | undefined =>
  Object.entries(toolNameToDisplayName).find(
    ([, displayName]) => displayName && !TOOL_DISPLAY_NAME_PATTERN.test(displayName),
  )?.[1];

export const buildEditServerPayload = (
  values: Record<string, unknown>,
  ui: EditServerUiState,
  mcpServer: MCPServer,
): BuildEditPayloadResult => {
  const badDisplayName = firstInvalidToolDisplayName(ui.toolNameToDisplayName);
  if (badDisplayName !== undefined) {
    return { kind: "invalid_tool_display_name", displayName: badDisplayName };
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
    ...restValues
  } = values;

  const stdio: StdioFieldsResult =
    restValues.transport === "stdio"
      ? buildStdioFields(rawStdioConfig, rawEnvJson, rawCommand, rawArgs)
      : { kind: "ok", fields: {} };
  if (stdio.kind !== "ok") return stdio;

  const rawTokenValidation = rawTokenValidationJson as string | undefined;
  const tokenValidationResult: JsonParseResult =
    rawTokenValidation && rawTokenValidation.trim() !== ""
      ? tryParseJson(rawTokenValidation)
      : { kind: "ok", value: null };
  if (tokenValidationResult.kind === "invalid") return { kind: "invalid_token_validation_json" };
  const tokenValidation = tokenValidationResult.value as Record<string, unknown> | null;

  // "openapi" is a UI-only transport; the backend stores those servers as plain http.
  const transport = restValues.transport === TRANSPORT.OPENAPI ? "http" : restValues.transport;
  const authType = restValues.auth_type as string | undefined;

  const accessGroups = ((restValues.mcp_access_groups as unknown[] | undefined) || []).map((g: unknown) =>
    typeof g === "string" ? g : (g as { name?: string })?.name || String(g),
  );

  const mcpInfoServerName =
    [
      restValues.server_name as string | undefined,
      restValues.url as string | undefined,
      mcpServer.server_name,
      mcpServer.url,
      restValues.alias as string | undefined,
      mcpServer.alias,
    ].find(Boolean) ?? "unknown";

  const toolAllowlistEnforced =
    ui.hasExistingToolAllowlist || ui.hasToolAllowlistInteraction || ui.allowedTools.length > 0;

  const extraHeaders = Array.isArray(restValues.extra_headers) ? restValues.extra_headers : [];
  const hasAuthorizationHeader = extraHeaders.some(
    (h: unknown) => typeof h === "string" && h.toLowerCase() === "authorization",
  );
  const isNoneAuth = authType === AUTH_TYPE.NONE || authType == null;

  const credentialsPayload = filterCredentials(credentialValues);
  const includeCredentials = authType !== undefined && AUTH_TYPES_REQUIRING_CREDENTIALS.includes(authType);
  // Client-forwarded rows persist ONLY the declared app; strip any token material lingering in the
  // form (e.g. from a prior oauth2 authorize this session) so it can never reach the row.
  const submitCredentials = isClientForwardedTokenMode(authType)
    ? preservedAdminCredentials(credentialsPayload)
    : credentialsPayload;
  const persistedCredentials =
    includeCredentials && submitCredentials && Object.keys(submitCredentials).length > 0
      ? submitCredentials
      : undefined;
  // Explicit removal of a saved app wins over the filter above. Blank fields are the keep-existing
  // convention (the backend merges partial credential updates), so removal must be an explicit-null
  // write: encrypt skips nulls and the merge overrides the stored keys, returning the server to
  // dynamic client registration.
  const credentials =
    ui.removeStoredApp && isClientForwardedTokenMode(authType)
      ? { client_id: null, client_secret: null }
      : persistedCredentials;

  return {
    kind: "ok",
    payload: {
      ...restValues,
      ...(transport === restValues.transport ? {} : { transport }),
      ...stdio.fields,
      stdio_config: undefined,
      env_json: undefined,
      ...(mcpServer.auth_type === AUTH_TYPE.OAUTH2 && authType !== AUTH_TYPE.OAUTH2
        ? { issuer: null, authorization_url: null, token_url: null, registration_url: null }
        : {}),
      ...(mcpServer.auth_type === AUTH_TYPE.OAUTH2_TOKEN_EXCHANGE && authType !== AUTH_TYPE.OAUTH2_TOKEN_EXCHANGE
        ? { token_exchange_endpoint: null, audience: null, subject_token_type: null, token_exchange_profile: null }
        : {}),
      server_id: mcpServer.server_id,
      mcp_info: {
        ...(mcpServer.mcp_info ?? {}),
        server_name: mcpInfoServerName,
        description: restValues.description,
        logo_url: ui.logoUrl || undefined,
        mcp_server_cost_info: Object.keys(ui.costConfig).length > 0 ? ui.costConfig : null,
        tool_allowlist_enforced: toolAllowlistEnforced,
      },
      mcp_access_groups: accessGroups,
      alias: restValues.alias,
      extra_headers: restValues.extra_headers || [],
      ...(toolAllowlistEnforced ? { allowed_tools: [...ui.allowedTools] } : {}),
      tool_name_to_display_name: Object.keys(ui.toolNameToDisplayName).length > 0 ? ui.toolNameToDisplayName : null,
      tool_name_to_description: Object.keys(ui.toolNameToDescription).length > 0 ? ui.toolNameToDescription : null,
      disallowed_tools: restValues.disallowed_tools || [],
      static_headers: reduceStaticHeaders(staticHeadersList),
      env_vars: normalizeEnvVars(envVarsList),
      allow_all_keys: Boolean(allowAllKeysRaw ?? mcpServer.allow_all_keys),
      available_on_public_internet: Boolean(availableOnPublicInternetRaw ?? mcpServer.available_on_public_internet),
      // ``delegate_auth_to_upstream`` is only honored server-side for ``auth_type=oauth2`` (PKCE
      // passthrough). The field is conditionally rendered so the value drops out of the form on
      // auth_type change; force false for any other configuration to avoid persisting a stale
      // ``true`` that would silently re-activate if the configuration is later switched back.
      delegate_auth_to_upstream:
        authType === AUTH_TYPE.OAUTH2
          ? Boolean(delegateAuthToUpstreamRaw ?? mcpServer.delegate_auth_to_upstream)
          : false,
      // ``oauth_passthrough`` is the dedicated, non-oauth2 opt-in, only honored for
      // ``auth_type=none`` servers that forward ``Authorization`` upstream. Kept separate from
      // ``delegate_auth_to_upstream`` so enabling pass-through never regresses oauth2 servers.
      oauth_passthrough:
        isNoneAuth && hasAuthorizationHeader ? Boolean(oauthPassthroughRaw ?? mcpServer.oauth_passthrough) : false,
      // ``dcr_bridge`` is only meaningful for the client-forwarded token modes. Same stale-value
      // reasoning as the two flags above.
      dcr_bridge: isClientForwardedTokenMode(authType) ? Boolean(dcrBridgeRaw ?? mcpServer.dcr_bridge) : false,
      ...(authType === AUTH_TYPE.OAUTH2 && restValues.oauth_flow_type
        ? {
            oauth2_flow:
              restValues.oauth_flow_type === OAUTH_FLOW.M2M ? MCP_OAUTH2_FLOW_M2M : MCP_OAUTH2_FLOW_INTERACTIVE,
          }
        : {}),
      ...(tokenValidation !== null || mcpServer.token_validation ? { token_validation: tokenValidation } : {}),
      ...(credentials === undefined ? {} : { credentials }),
    },
  };
};
