import {
  AUTH_TYPE,
  MCPServerCostInfo,
  MCP_OAUTH2_FLOW_INTERACTIVE,
  MCP_OAUTH2_FLOW_M2M,
  OAUTH_FLOW,
  TRANSPORT,
  isClientForwardedTokenMode,
  preservedAdminCredentials,
} from "@/components/mcp_tools/types";
import { TOOL_DISPLAY_NAME_PATTERN, normalizeEnvVars } from "./utils";

export const AUTH_TYPES_REQUIRING_AUTH_VALUE = [
  AUTH_TYPE.API_KEY,
  AUTH_TYPE.BEARER_TOKEN,
  AUTH_TYPE.TOKEN,
  AUTH_TYPE.BASIC,
];

export const AUTH_TYPES_REQUIRING_CREDENTIALS = [
  ...AUTH_TYPES_REQUIRING_AUTH_VALUE,
  AUTH_TYPE.OAUTH2,
  AUTH_TYPE.OAUTH2_TOKEN_EXCHANGE,
  AUTH_TYPE.OAUTH2_ID_JAG,
  AUTH_TYPE.AWS_SIGV4,
  AUTH_TYPE.TRUE_PASSTHROUGH,
  AUTH_TYPE.OAUTH_DELEGATE,
];

export interface DcrClient {
  readonly client_id: string;
  readonly client_secret?: string;
}

export interface CreateServerUiState {
  readonly transportType: string;
  readonly costConfig: MCPServerCostInfo;
  readonly allowedTools: readonly string[];
  readonly hasToolAllowlistInteraction: boolean;
  readonly toolNameToDisplayName: Readonly<Record<string, string>>;
  readonly toolNameToDescription: Readonly<Record<string, string>>;
  readonly logoUrl: string | undefined;
  readonly dcrClient: DcrClient | null;
}

export type BuildCreatePayloadResult =
  | { readonly kind: "ok"; readonly payload: Record<string, unknown> }
  | { readonly kind: "invalid_tool_display_name"; readonly displayName: string }
  | { readonly kind: "invalid_stdio_json" }
  | { readonly kind: "invalid_token_validation_json" };

export type StdioParseResult =
  | { readonly kind: "ok"; readonly fields: Record<string, unknown>; readonly derivedServerName?: string }
  | { readonly kind: "invalid" };

type JsonParseResult =
  | { readonly kind: "ok"; readonly value: Record<string, unknown> | null }
  | { readonly kind: "invalid" };

const tryParseJson = (raw: string): JsonParseResult => {
  try {
    return { kind: "ok", value: JSON.parse(raw) };
  } catch {
    return { kind: "invalid" };
  }
};

export const reduceStaticHeaders = (list: unknown): Record<string, string> => {
  if (!Array.isArray(list)) return {};
  return list.reduce((acc: Record<string, string>, entry: Record<string, string>) => {
    const header = entry?.header?.trim();
    if (header) acc[header] = (entry?.value ?? "").trim();
    return acc;
  }, {});
};

// Accepts both the full `{"mcpServers": {"name": {...}}}` shape a user copies out of a client config
// and a bare `{"command": ..., "args": ..., "env": ...}`. A non-object JSON body (null, a number)
// falls through to the invalid branch, which is what the caller surfaces to the admin.
export const parseStdioConfig = (raw: string): StdioParseResult => {
  try {
    const stdioConfig = JSON.parse(raw);
    const nestedName =
      stdioConfig.mcpServers && typeof stdioConfig.mcpServers === "object"
        ? Object.keys(stdioConfig.mcpServers)[0]
        : undefined;
    const actualConfig = nestedName === undefined ? stdioConfig : stdioConfig.mcpServers[nestedName];

    return {
      kind: "ok",
      fields: { command: actualConfig.command, args: actualConfig.args, env: actualConfig.env },
      // The JSON's own server key is the fallback name when the admin left the field blank.
      ...(nestedName === undefined ? {} : { derivedServerName: nestedName.replace(/-/g, "_") }),
    };
  } catch {
    return { kind: "invalid" };
  }
};

const filterCredentials = (credentialValues: unknown): Record<string, unknown> | undefined => {
  if (!credentialValues || typeof credentialValues !== "object") return undefined;
  return Object.entries(credentialValues as Record<string, unknown>).reduce(
    (acc: Record<string, unknown>, [key, value]) => {
      if (value === undefined || value === null || value === "") {
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
    },
    {},
  );
};

const firstInvalidToolDisplayName = (toolNameToDisplayName: Readonly<Record<string, string>>): string | undefined =>
  Object.entries(toolNameToDisplayName).find(
    ([, displayName]) => displayName && !TOOL_DISPLAY_NAME_PATTERN.test(displayName),
  )?.[1];

export const buildCreateServerPayload = (
  values: Record<string, unknown>,
  ui: CreateServerUiState,
): BuildCreatePayloadResult => {
  const badDisplayName = firstInvalidToolDisplayName(ui.toolNameToDisplayName);
  if (badDisplayName !== undefined) {
    return { kind: "invalid_tool_display_name", displayName: badDisplayName };
  }

  const {
    static_headers: staticHeadersList,
    env_vars: envVarsList,
    stdio_config: rawStdioConfig,
    credentials: credentialValues,
    allow_all_keys: allowAllKeysRaw,
    available_on_public_internet: availableOnPublicInternetRaw,
    delegate_auth_to_upstream: delegateAuthToUpstreamRaw,
    oauth_passthrough: oauthPassthroughRaw,
    dcr_bridge: dcrBridgeRaw,
    token_validation_json: rawTokenValidationJson,
    ...restValues
  } = values;

  const stdio: StdioParseResult =
    rawStdioConfig && ui.transportType === "stdio"
      ? parseStdioConfig(rawStdioConfig as string)
      : { kind: "ok", fields: {} };
  if (stdio.kind === "invalid") {
    return { kind: "invalid_stdio_json" };
  }

  const rawTokenValidation = rawTokenValidationJson as string | undefined;
  const tokenValidationResult: JsonParseResult =
    rawTokenValidation && rawTokenValidation.trim() !== ""
      ? tryParseJson(rawTokenValidation)
      : { kind: "ok", value: null };
  if (tokenValidationResult.kind === "invalid") {
    return { kind: "invalid_token_validation_json" };
  }
  const tokenValidation = tokenValidationResult.value;

  const serverName = (restValues.server_name as string | undefined) || stdio.derivedServerName;
  // "openapi" is a UI-only transport; the backend stores those servers as plain http.
  const transport = restValues.transport === TRANSPORT.OPENAPI ? "http" : restValues.transport;
  const authType = restValues.auth_type as string | undefined;

  const credentialsPayload = filterCredentials(credentialValues);
  const includeCredentials = authType !== undefined && AUTH_TYPES_REQUIRING_CREDENTIALS.includes(authType);
  // Client-forwarded rows persist ONLY the declared app; strip any token material that lingered in
  // the form (e.g. from a prior oauth2 authorize on the same session) so it can never reach the row.
  const submitCredentials = isClientForwardedTokenMode(authType)
    ? preservedAdminCredentials(credentialsPayload)
    : credentialsPayload;
  const persistedCredentials =
    includeCredentials && submitCredentials && Object.keys(submitCredentials).length > 0
      ? submitCredentials
      : undefined;
  // An interactive (oauth2) create persists its DCR-minted client from the ref (kept out of the
  // form store); reuse a re-authorize's registered client instead of re-registering.
  const credentials =
    authType === AUTH_TYPE.OAUTH2 && ui.dcrClient
      ? { ...(persistedCredentials ?? {}), ...ui.dcrClient }
      : persistedCredentials;

  return {
    kind: "ok",
    payload: {
      ...restValues,
      ...stdio.fields,
      ...(serverName === restValues.server_name ? {} : { server_name: serverName }),
      ...(transport === restValues.transport ? {} : { transport }),
      // Remove the raw stdio_config field as we've extracted its components
      stdio_config: undefined,
      mcp_info: {
        server_name: serverName || restValues.url,
        description: restValues.description,
        logo_url: ui.logoUrl || undefined,
        mcp_server_cost_info: Object.keys(ui.costConfig).length > 0 ? ui.costConfig : null,
        tool_allowlist_enforced: ui.hasToolAllowlistInteraction || ui.allowedTools.length > 0,
      },
      mcp_access_groups: restValues.mcp_access_groups,
      alias: restValues.alias,
      allowed_tools: [...ui.allowedTools],
      tool_name_to_display_name: ui.toolNameToDisplayName,
      tool_name_to_description: ui.toolNameToDescription,
      allow_all_keys: Boolean(allowAllKeysRaw),
      available_on_public_internet: Boolean(availableOnPublicInternetRaw),
      delegate_auth_to_upstream: Boolean(delegateAuthToUpstreamRaw),
      oauth_passthrough: Boolean(oauthPassthroughRaw),
      // ``dcr_bridge`` is only meaningful for the client-forwarded token
      // modes (true_passthrough / oauth_delegate) and defaults on when the
      // toggle is shown; force false for any other auth type so a stale
      // ``true`` is never persisted. Mirrors the sibling flags above.
      dcr_bridge: isClientForwardedTokenMode(authType) ? Boolean(dcrBridgeRaw ?? true) : false,
      ...(authType === AUTH_TYPE.OAUTH2
        ? {
            oauth2_flow: values.oauth_flow_type === OAUTH_FLOW.M2M ? MCP_OAUTH2_FLOW_M2M : MCP_OAUTH2_FLOW_INTERACTIVE,
          }
        : {}),
      static_headers: reduceStaticHeaders(staticHeadersList),
      env_vars: normalizeEnvVars(envVarsList),
      ...(tokenValidation !== null && { token_validation: tokenValidation }),
      ...(credentials === undefined ? {} : { credentials }),
    },
  };
};
