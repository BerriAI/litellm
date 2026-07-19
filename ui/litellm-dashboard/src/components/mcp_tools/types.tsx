/** A single MCP tool event emitted by the LiteLLM proxy during a Responses API turn. */
export interface MCPEvent {
  type: string;
  sequence_number?: number;
  output_index?: number;
  item_id?: string;
  item?: {
    id?: string;
    type?: string;
    server_label?: string;
    tools?: Array<{
      name: string;
      description: string;
      annotations?: {
        read_only?: boolean;
      };
      input_schema?: unknown;
    }>;
    name?: string;
    arguments?: string;
    output?: string;
  };
  delta?: string;
  arguments?: string;
  timestamp?: number;
}

export interface Team {
  team_id: string;
  team_alias?: string;
  organization_id?: string | null;
}

// Default no auth value
export const AUTH_TYPE = {
  NONE: "none",
  API_KEY: "api_key",
  BEARER_TOKEN: "bearer_token",
  TOKEN: "token",
  BASIC: "basic",
  OAUTH2: "oauth2",
  OAUTH2_TOKEN_EXCHANGE: "oauth2_token_exchange",
  AWS_SIGV4: "aws_sigv4",
  TRUE_PASSTHROUGH: "true_passthrough",
  OAUTH_DELEGATE: "oauth_delegate",
};

// The two client-forwarded token modes: the caller supplies the upstream Authorization (forwarded
// verbatim for true_passthrough, alongside LiteLLM admission for oauth_delegate). The dashboard holds
// their token in sessionStorage instead of persisting it, and the browser-authorize temp payload keeps
// their real auth_type so the backend does not treat them as needing a stored per-user token.
export const isClientForwardedTokenMode = (authType?: string | null): boolean =>
  authType === AUTH_TYPE.TRUE_PASSTHROUGH || authType === AUTH_TYPE.OAUTH_DELEGATE;

export const OAUTH_FLOW = {
  INTERACTIVE: "interactive",
  M2M: "m2m",
};

// The fields that determine which upstream OAuth token "Authorize & Fetch" mints: the resource/audience
// (url, or spec_path for OpenAPI servers), the OAuth mode/grant (auth_type, oauth_flow_type), the OAuth
// client and requested scope (credentials.client_id / client_secret / scopes), and the authorization-server
// endpoints (authorization_url / token_url / registration_url). Grounded in RFC 8707 / RFC 8693 and the MCP
// auth spec: an access token is bound to exactly this tuple (resource/audience + scope + client + issuer), so
// a previously authorized token is stale if and only if this identity changes and must be re-minted.
// url and spec_path are compared independently rather than selected by transport: the create form keeps
// transport in component state, not in form values, so a transport-conditional target would silently pin the
// audience to a missing url and never fire for spec_path edits on OpenAPI servers. Mirrors the backend's
// mcp_oauth_token_identity. Deliberately EXCLUDES: transport itself (http<->sse on the same url is the same
// audience; a switch to/from OpenAPI shows up as url/spec_path changes because each form clears the field the
// new transport does not use), delegate_auth_to_upstream (a downstream-usage toggle that is never sent to the
// authorize request), and all metadata/RBAC/routing fields. Shared by the create and edit forms so their
// invalidation logic cannot drift.
export const getOAuthAuthorizationIdentity = (values: Record<string, unknown>): string => {
  const credentials = (values.credentials ?? {}) as Record<string, unknown>;
  const identity = {
    url: typeof values.url === "string" ? values.url : null,
    spec_path: typeof values.spec_path === "string" ? values.spec_path : null,
    auth_type: values.auth_type ?? null,
    oauth_flow_type: values.oauth_flow_type ?? null,
    client_id: credentials.client_id ?? null,
    client_secret: credentials.client_secret ?? null,
    scopes: credentials.scopes ?? null,
    issuer: values.issuer ?? null,
    authorization_url: values.authorization_url ?? null,
    token_url: values.token_url ?? null,
    registration_url: values.registration_url ?? null,
  };
  return JSON.stringify(identity);
};

// The form fields wiped when a held OAuth token is invalidated: only `credentials`, which holds the
// minted material (the fetched token + DCR client). The authorization/token/registration endpoint
// fields are deliberately NOT wiped: nothing programmatic ever writes them (upstream discovery happens
// backend-side), so they only ever hold admin input, and resetting them would wipe it (create) or
// silently revert it to the saved record (edit, whose Form has initialValues). Shared by the create and
// edit forms so what gets wiped cannot drift.
export const CLEARED_ON_INVALIDATION = ["credentials"] as const;

// The declared-app filter over form.credentials. It is a pure key filter with no mode/transition
// guard because the surrounding code establishes that a client_id/client_secret in form.credentials
// is ALWAYS admin-typed in every reachable state: the create form holds the DCR-minted client in a
// ref and never writes it into the form store, the edit form's onTokenReceived never writes client
// keys, and the invalidation reset clears the whole object atomically. So preserving the string
// client keys across any invalidation (URL/endpoint edit, true_passthrough<->oauth_delegate switch,
// or a round trip through another mode) is always legitimate, while the output key filter excludes
// token-shaped keys so a preserve can never carry minted material through. Shared by both forms.
const DECLARED_APP_CREDENTIAL_KEYS = ["client_id", "client_secret"] as const;

// Minted token material the oauth2 authorize path writes beside the app keys; stripped from restored
// snapshots and from any credentials that transit to the temp-session preview so a stale token never
// reaches the backend or a client-forwarded server row.
export const MINTED_TOKEN_CREDENTIAL_KEYS = ["access_token", "refresh_token", "expires_in", "scope"] as const;

export const preservedDeclaredAppCredentials = (
  credentials: Record<string, unknown> | null | undefined,
): Record<string, string> | undefined => {
  if (!credentials) return undefined;
  const kept = Object.fromEntries(
    DECLARED_APP_CREDENTIAL_KEYS.filter((key) => typeof credentials[key] === "string" && credentials[key] !== "").map(
      (key) => [key, credentials[key] as string],
    ),
  );
  return Object.keys(kept).length > 0 ? kept : undefined;
};

// Drop minted token keys, keeping everything else (the declared app plus any non-token config).
export const withoutMintedTokenCredentials = (
  credentials: Record<string, unknown> | null | undefined,
): Record<string, unknown> | undefined => {
  if (!credentials) return undefined;
  const kept = Object.fromEntries(
    Object.entries(credentials).filter(([key]) => !(MINTED_TOKEN_CREDENTIAL_KEYS as readonly string[]).includes(key)),
  );
  // Return undefined (not {}) when only minted keys were present, so a restore spreads `credentials:
  // undefined` (the fields keep their placeholder / keep-existing state) rather than blanking them.
  return Object.keys(kept).length > 0 ? kept : undefined;
};

// The client-forwarded modes share one credential class (same declared app, same authorize relay), so
// a switch between them must NOT be treated as an app change. Mirrors the backend _credential_auth_class
// in db.py; kept in sync so the UI's keep-existing copy and the backend's merge cannot disagree.
export const credentialAuthClass = (authType: string | null | undefined): string | null => {
  if (authType === AUTH_TYPE.TRUE_PASSTHROUGH || authType === AUTH_TYPE.OAUTH_DELEGATE) return "client_forwarded";
  return authType ?? null;
};

// True when a token was authorized in this session (authorizedIdentity recorded at mint time) and the
// form's current identity no longer matches it. Every invalidation decision in both forms goes through
// this single check: onValuesChange for user edits, and an explicit recheck after any programmatic
// form.setFieldsValue (antd does not fire onValuesChange for those), so a missed event path cannot let a
// stale token survive.
export const isHeldOAuthTokenStale = (
  values: Record<string, unknown>,
  authorizedIdentity: string | undefined,
): boolean => authorizedIdentity !== undefined && getOAuthAuthorizationIdentity(values) !== authorizedIdentity;

// Backend value of `oauth2_flow` that marks a machine-to-machine server. Distinct
// from the UI-local OAUTH_FLOW.M2M ("m2m"); this is what the API actually returns.
export const MCP_OAUTH2_FLOW_M2M = "client_credentials";

export const MCP_OAUTH2_FLOW_INTERACTIVE = "authorization_code";

export type McpOAuthMode = "m2m" | "passthrough" | "authorization_code" | "token_exchange";

// Classify an OAuth MCP server into the mode that decides how the tool list is
// authenticated. token_exchange (RFC 8693 / OBO) is its own auth_type
// (`oauth2_token_exchange`), so it is keyed off auth_type directly; the other
// three all share auth_type `oauth2` and are told apart by secondary fields:
// M2M (backend service token via the client_credentials grant), PKCE passthrough
// (browser-held session token), or authorization_code (per-user token obtained
// via the interactive authorization_code/PKCE grant and stored by the backend).
// `token_url` is intentionally not consulted for the oauth2 modes: every OAuth2
// grant that exchanges for a token carries one (interactive PKCE and
// client_credentials alike), so it cannot distinguish the modes; `oauth2_flow`
// is the authoritative M2M signal.
export function getMcpOAuthMode(s: {
  auth_type?: string | null;
  oauth2_flow?: string | null;
  delegate_auth_to_upstream?: boolean | null;
}): McpOAuthMode | null {
  if (s.auth_type === AUTH_TYPE.OAUTH2_TOKEN_EXCHANGE) return "token_exchange";
  if (s.auth_type !== AUTH_TYPE.OAUTH2) return null;
  if (s.oauth2_flow === MCP_OAUTH2_FLOW_M2M) return "m2m";
  return s.delegate_auth_to_upstream ? "passthrough" : "authorization_code";
}

// Map a server's stored `oauth2_flow` (the API value: client_credentials /
// authorization_code / null) to the edit form's OAuth Flow Type select value.
// A null/unset flow returns undefined so the select shows its placeholder rather
// than a guessed default — an unstamped legacy row must be assigned explicitly.
export function oauth2FlowToFormValue(oauth2Flow?: string | null): string | undefined {
  if (oauth2Flow === MCP_OAUTH2_FLOW_M2M) return OAUTH_FLOW.M2M;
  if (oauth2Flow) return OAUTH_FLOW.INTERACTIVE;
  return undefined;
}

export const TRANSPORT = {
  SSE: "sse",
  HTTP: "http",
  STDIO: "stdio",
  OPENAPI: "openapi",
};

export const handleTransport = (transport?: string | null, specPath?: string | null): string => {
  if (transport === null || transport === undefined) {
    return TRANSPORT.SSE;
  }

  // If server has spec_path, display as "openapi" instead of the raw transport type
  if (specPath && transport !== TRANSPORT.STDIO) {
    return TRANSPORT.OPENAPI;
  }

  return transport;
};

export const handleAuth = (authType?: string | null): string => {
  if (authType === null || authType === undefined) {
    return AUTH_TYPE.NONE;
  }

  return authType;
};

// Define the structure for tool input schema properties
export interface InputSchemaProperty {
  type: string;
  description?: string;
  properties?: Record<string, InputSchemaProperty>; // For nested object properties
  required?: string[]; // For required fields in nested objects
  enum?: string[]; // For enum values
  default?: any; // For default values
  items?: InputSchemaProperty | InputSchemaProperty[]; // For array item schemas
}

// Define the structure for the input schema of a tool
export interface InputSchema {
  type: "object";
  properties: Record<string, InputSchemaProperty>;
  required?: string[];
}

// Define MCPServerCostInfo for cost tracking
export interface MCPServerCostInfo {
  default_cost_per_query?: number | null;
  tool_name_to_cost_per_query?: Record<string, number | null>;
}

// Define MCP provider info
export interface MCPInfo {
  server_name: string;
  description?: string;
  logo_url?: string;
  mcp_server_cost_info?: MCPServerCostInfo | null;
  tool_allowlist_enforced?: boolean;
}

// Define the structure for a single MCP tool
export interface MCPTool {
  name: string;
  description?: string;
  inputSchema: InputSchema | string; // API returns string "tool_input_schema" or the actual schema
  mcp_info: MCPInfo;
  // Function to select a tool (added in the component)
  onToolSelect?: (tool: MCPTool) => void;
}

// Define the response structure for the listMCPTools endpoint - now a flat array
export type ListMCPToolsResponse = MCPTool[];

// Define the argument structure for calling an MCP tool
export interface CallMCPToolArgs {
  name: string;
  arguments: Record<string, any> | null;
  server_name?: string; // Now using server_name from mcp_info
}

// Define the possible content types in the response
export interface MCPTextContent {
  type: "text";
  text: string;
  annotations?: any;
}

export interface MCPImageContent {
  type: "image";
  url?: string;
  data?: string;
}

export interface MCPEmbeddedResource {
  type: "embedded_resource";
  resource_type?: string;
  url?: string;
  data?: any;
}

// Define the union type for the content array in the response
export type MCPContent = MCPTextContent | MCPImageContent | MCPEmbeddedResource;

// Define the response structure for the callMCPTool endpoint
export type CallMCPToolResponse = {
  content: MCPContent[];
  _meta: any;
  isError: boolean;
  structuredContent: any;
};

// Props for the main component
export interface MCPToolsViewerProps {
  serverId: string;
  accessToken: string | null;
  auth_type?: string | null;
  /** Backend OAuth2 grant; `client_credentials` marks an M2M server. */
  oauth2_flow?: string | null;
  /** When true (interactive OAuth2), the server uses PKCE passthrough. */
  delegate_auth_to_upstream?: boolean | null;
  /**
   * Connection field present on every OAuth2 flow (interactive and M2M alike),
   * so it does not indicate the mode. Retained for callers/other uses; not read
   * for mode detection — see getMcpOAuthMode.
   */
  tokenUrl?: string | null;
  userRole: string | null;
  userID: string | null;
  serverAlias?: string | null;
  extraHeaders?: string[] | null;
}

export interface MCPServer {
  server_id: string;
  server_name?: string | null;
  alias?: string | null;
  description?: string | null;
  /**
   * Only required for HTTP/SSE transports.
   * For `stdio`, the backend can return null/undefined.
   */
  url?: string | null;
  spec_path?: string | null;
  transport?: string | null;
  auth_type?: string | null;
  oauth2_flow?: string | null;
  issuer?: string | null;
  authorization_url?: string | null;
  token_url?: string | null;
  registration_url?: string | null;
  token_exchange_endpoint?: string | null;
  audience?: string | null;
  subject_token_type?: string | null;
  token_exchange_profile?: string | null;
  mcp_info?: MCPInfo | null;
  created_at: string;
  created_by: string;
  updated_at: string;
  updated_by: string;
  extra_headers?: string[] | null;
  static_headers?: Record<string, string> | null;
  status?: "healthy" | "unhealthy" | "unknown";
  last_health_check?: string | null;
  health_check_error?: string | null;
  teams?: Team[];
  mcp_access_groups?: string[];
  allowed_tools?: string[];
  tool_name_to_display_name?: Record<string, string>;
  tool_name_to_description?: Record<string, string>;
  allow_all_keys?: boolean;
  available_on_public_internet?: boolean;
  allowed_cidrs?: string[] | null;
  delegate_auth_to_upstream?: boolean;
  oauth_passthrough?: boolean;
  dcr_bridge?: boolean | null;
  max_concurrent_requests?: number | null;
  /** Redacted to null in server responses; present when constructing a server locally. */
  credentials?: Record<string, unknown> | null;

  /** Stdio-only fields (present when transport === 'stdio') */
  command?: string | null;
  args?: string[] | null;
  env?: Record<string, string> | null;

  /** BYOK (Bring Your Own Key) fields */
  is_byok?: boolean | null;
  byok_description?: string[] | null;
  byok_api_key_help_url?: string | null;
  has_user_credential?: boolean | null;

  /** GitHub / source repository URL */
  source_url?: string | null;

  /** BYOM (Bring Your Own MCP) submission fields */
  approval_status?: "active" | "pending_review" | "rejected" | null;
  submitted_by?: string | null;
  submitted_at?: string | null;
  reviewed_at?: string | null;
  review_notes?: string | null;

  /** Per-user OAuth token storage settings (interactive OAuth only) */
  token_validation?: Record<string, any> | null;
  token_storage_ttl_seconds?: number | null;

  /**
   * Admin-configured env vars interpolated into static_headers via ${NAME}.
   * Stored as a list so the UI can preserve admin-entered ordering.
   */
  env_vars?: MCPEnvVar[] | null;
}

/** One environment variable entry on an MCP server. */
export type MCPEnvVarScope = "global" | "user";

export interface MCPEnvVar {
  name: string;
  /** For scope="global": the value used in interpolation.
   *  For scope="user": optional placeholder/description shown to users. */
  value: string;
  scope: MCPEnvVarScope;
  description?: string | null;
}

/** One required per-user env var slot returned by the user-env-vars endpoint. */
export interface MCPUserEnvVarSpec {
  name: string;
  description?: string | null;
  is_set: boolean;
}

/** Per-server per-user env var status returned by the API. */
export interface MCPUserEnvVarsStatus {
  server_id: string;
  server_name?: string | null;
  alias?: string | null;
  required: MCPUserEnvVarSpec[];
  missing_count: number;
  setup_url?: string | null;
}

export interface MCPServerProps {
  accessToken: string | null;
  userRole: string | null;
  userID: string | null;
}

export interface MCPToolsetTool {
  server_id: string;
  tool_name: string;
}

export interface MCPToolset {
  toolset_id: string;
  toolset_name: string;
  description?: string;
  tools: MCPToolsetTool[];
  created_at?: string;
  created_by?: string;
}

// Discoverable MCP server from the curated registry
export interface DiscoverableMCPServer {
  name: string;
  title: string;
  description: string;
  icon_url?: string | null;
  category: string;
  registry_url?: string | null;
  transport: string;
  url?: string | null;
  command?: string | null;
  args?: string[] | null;
  env_vars?: Array<{ name: string; description?: string; secret?: boolean }> | null;
}

export interface DiscoverMCPServersResponse {
  servers: DiscoverableMCPServer[];
  categories: string[];
}

export interface MCPSubmissionsSummary {
  total: number;
  pending_review: number;
  active: number;
  rejected: number;
  items: MCPServer[];
}
