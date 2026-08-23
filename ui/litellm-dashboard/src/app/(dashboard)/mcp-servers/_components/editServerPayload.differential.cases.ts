import { MCPServer } from "@/components/mcp_tools/types";
import type { EditServerUiState } from "./editServerPayload";

const SERVER: MCPServer = {
  server_id: "srv_1",
  server_name: "srv",
  alias: "srv_alias",
  description: "a server",
  transport: "http",
  url: "https://example.com/mcp",
  auth_type: "none",
  created_at: "2024-01-01T00:00:00Z",
  created_by: "user-1",
  updated_at: "2024-01-01T00:00:00Z",
  updated_by: "user-1",
  mcp_access_groups: [],
};

export const baseUi: EditServerUiState = {
  mcpServer: SERVER,
  logoUrl: undefined,
  costConfig: {},
  allowedTools: [],
  hasExistingToolAllowlist: false,
  hasToolAllowlistInteraction: false,
  toolNameToDisplayName: {},
  toolNameToDescription: {},
  removeStoredApp: false,
};

export interface DifferentialCase {
  readonly label: string;
  readonly values: Record<string, any>;
  readonly ui: Partial<EditServerUiState>;
}

// Always-mounted root fields. A "collapsed" section is modelled as its keys being
// ABSENT, which is what antd's mounted-only onFinish produces, and an expanded but
// untouched one as the key present holding `undefined`.
const ROOT = {
  server_name: "srv",
  alias: "srv_alias",
  description: "a server",
  transport: "http",
  url: "https://example.com/mcp",
  auth_type: "none",
  max_concurrent_requests: undefined,
  mcp_access_groups: [],
  extra_headers: [],
  static_headers: [],
  env_vars: [],
  allow_all_keys: false,
  available_on_public_internet: true,
};

const CREDS = {
  auth_value: "secret-value",
  client_id: "cid",
  client_secret: "csec",
  scopes: ["read", "write"],
  aws_region_name: "us-east-1",
  aws_service_name: "bedrock",
  aws_access_key_id: "AKIA",
  aws_secret_access_key: "shh",
  aws_session_token: "tok",
  aws_role_name: "role",
  aws_session_name: "sess",
  id_jag_resource: "api://res",
  id_jag_resource_token_endpoint: "https://idp/jag",
  client_private_key: "-----KEY-----",
  client_private_key_id: "kid",
  client_assertion_signing_alg: "RS256",
  token_endpoint_auth_method: "client_secret_basic",
  upstream_resource: "api://up",
};

const AUTH_TYPES = [
  "none",
  "api_key",
  "bearer_token",
  "token",
  "basic",
  "oauth2",
  "oauth2_token_exchange",
  "oauth2_id_jag",
  "aws_sigv4",
  "true_passthrough",
  "oauth_delegate",
] as const;

export const CASES: readonly DifferentialCase[] = [
  // --- mount-state axis: everything collapsed / expanded-undefined / expanded-valued ---
  { label: "everything collapsed, only the always-mounted root", values: { ...ROOT }, ui: {} },
  {
    label: "everything expanded, all optional keys present as undefined",
    values: {
      ...ROOT,
      credentials: {},
      stdio_config: undefined,
      env_json: undefined,
      command: undefined,
      args: undefined,
      delegate_auth_to_upstream: undefined,
      oauth_passthrough: undefined,
      dcr_bridge: undefined,
      token_validation_json: undefined,
      spec_path: undefined,
      oauth_flow_type: undefined,
      issuer: undefined,
      authorization_url: undefined,
      token_url: undefined,
      registration_url: undefined,
      token_storage_ttl_seconds: undefined,
      token_exchange_endpoint: undefined,
      token_exchange_profile: undefined,
      audience: undefined,
      subject_token_type: undefined,
      disallowed_tools: undefined,
    },
    ui: {},
  },
  {
    label: "expand then collapse: the key is absent again even though a value was typed",
    values: { ...ROOT },
    ui: {},
  },

  // --- transport axis ---
  {
    label: "stdio via the JSON config path",
    values: {
      ...ROOT,
      transport: "stdio",
      url: undefined,
      stdio_config: JSON.stringify({ command: "npx", args: ["-y", "pkg"], env: { A: "1" } }),
    },
    ui: {},
  },
  {
    label: "stdio via a wrapped mcpServers config",
    values: {
      ...ROOT,
      transport: "stdio",
      url: undefined,
      stdio_config: JSON.stringify({ mcpServers: { first: { command: "uvx", args: [1, "  ", "b"] } } }),
    },
    ui: {},
  },
  {
    label: "stdio via the dedicated command/args/env fields",
    values: {
      ...ROOT,
      transport: "stdio",
      url: undefined,
      command: "  npx  ",
      args: ["-y", "  ", "pkg"],
      env_json: JSON.stringify({ A: 1, "": "skipped", B: null }),
    },
    ui: {},
  },
  {
    label: "openapi swaps url for spec_path",
    values: { ...ROOT, transport: "openapi", url: undefined, spec_path: "https://x/openapi.json" },
    ui: {},
  },

  // --- auth-type axis, one case per type, credentials mounted ---
  ...AUTH_TYPES.map((auth) => ({
    label: `auth_type ${auth} with the full credential bag mounted`,
    values: { ...ROOT, auth_type: auth, credentials: { ...CREDS } },
    ui: {},
  })),

  // --- oauth2 flow sub-branches ---
  {
    label: "oauth2 m2m flow",
    values: {
      ...ROOT,
      auth_type: "oauth2",
      oauth_flow_type: "m2m",
      token_url: "https://idp/token",
      credentials: { ...CREDS },
    },
    ui: {},
  },
  {
    label: "oauth2 interactive flow",
    values: {
      ...ROOT,
      auth_type: "oauth2",
      oauth_flow_type: "interactive",
      issuer: "https://idp",
      credentials: { ...CREDS },
    },
    ui: {},
  },
  {
    label: "oauth2 with delegate_auth_to_upstream on",
    values: { ...ROOT, auth_type: "oauth2", delegate_auth_to_upstream: true, credentials: { ...CREDS } },
    ui: {},
  },

  // --- the five coalesced keys: unbound raw against each server-side fallback ---
  {
    label: "coalesced booleans unbound, server has them all true",
    values: { ...ROOT, auth_type: "oauth2", allow_all_keys: undefined, available_on_public_internet: undefined },
    ui: {
      mcpServer: {
        ...SERVER,
        auth_type: "oauth2",
        allow_all_keys: true,
        available_on_public_internet: true,
        delegate_auth_to_upstream: true,
        oauth_passthrough: true,
        dcr_bridge: true,
      },
    },
  },
  {
    label: "coalesced booleans unbound, server has them all false",
    values: { ...ROOT, allow_all_keys: undefined, available_on_public_internet: undefined },
    ui: { mcpServer: { ...SERVER, allow_all_keys: false, available_on_public_internet: false } },
  },
  {
    label: "dcr_bridge bound true on a client-forwarded mode",
    values: { ...ROOT, auth_type: "true_passthrough", dcr_bridge: true },
    ui: { mcpServer: { ...SERVER, auth_type: "true_passthrough" } },
  },
  {
    label: "dcr_bridge bound true but auth switched away, forced false",
    values: { ...ROOT, auth_type: "api_key", dcr_bridge: true },
    ui: {},
  },
  {
    label: "oauth_passthrough with an Authorization extra header",
    values: { ...ROOT, auth_type: "none", extra_headers: ["Authorization"], oauth_passthrough: true },
    ui: {},
  },
  {
    label: "oauth_passthrough without the Authorization header, forced false",
    values: { ...ROOT, auth_type: "none", extra_headers: ["X-Other"], oauth_passthrough: true },
    ui: {},
  },

  // --- auth-type transitions that null out the previous subtree ---
  {
    label: "was oauth2, now api_key: nulls the four oauth endpoints",
    values: { ...ROOT, auth_type: "api_key", credentials: { auth_value: "v" } },
    ui: { mcpServer: { ...SERVER, auth_type: "oauth2" } },
  },
  {
    label: "was token_exchange, now none: nulls the four exchange fields",
    values: { ...ROOT, auth_type: "none" },
    ui: { mcpServer: { ...SERVER, auth_type: "oauth2_token_exchange" } },
  },

  // --- credentials filtering ---
  // ADMIN_CONFIG_CREDENTIAL_KEYS is exactly ["upstream_resource"], so only that key
  // takes the blank-to-explicit-null branch. A blank client_id is dropped instead.
  {
    label: "blank upstream_resource becomes an explicit null",
    values: { ...ROOT, auth_type: "oauth2", credentials: { upstream_resource: "", client_secret: "keep" } },
    ui: {},
  },
  {
    label: "blank non-admin credential is dropped, not nulled",
    values: { ...ROOT, auth_type: "oauth2", credentials: { client_id: "", client_secret: "keep", scopes: [] } },
    ui: {},
  },
  {
    label: "scopes array filters empties and drops when nothing survives",
    values: { ...ROOT, auth_type: "oauth2", credentials: { scopes: ["", null, "read"] } },
    ui: {},
  },
  {
    label: "scopes entirely empty drops the key",
    values: { ...ROOT, auth_type: "oauth2", credentials: { scopes: ["", null] } },
    ui: {},
  },
  { label: "credentials absent entirely", values: { ...ROOT, auth_type: "oauth2" }, ui: {} },
  {
    label: "removeStoredApp forces an explicit-null app write",
    values: { ...ROOT, auth_type: "true_passthrough", credentials: { ...CREDS } },
    ui: { removeStoredApp: true, mcpServer: { ...SERVER, auth_type: "true_passthrough" } },
  },
  {
    label: "removeStoredApp ignored outside a client-forwarded mode",
    values: { ...ROOT, auth_type: "api_key", credentials: { ...CREDS } },
    ui: { removeStoredApp: true },
  },

  // --- headers, env vars, access groups ---
  {
    label: "static headers trim and drop blank names",
    values: {
      ...ROOT,
      static_headers: [
        { header: "  X-A  ", value: "  v  " },
        { header: "   ", value: "x" },
      ],
    },
    ui: {},
  },
  {
    label: "env vars normalize",
    values: { ...ROOT, env_vars: [{ name: "A", value: "1", scope: "server", description: "d" }] },
    ui: {},
  },
  { label: "access groups given as objects", values: { ...ROOT, mcp_access_groups: [{ name: "g1" }, "g2"] }, ui: {} },
  { label: "extra_headers absent falls back to an empty array", values: { ...ROOT, extra_headers: undefined }, ui: {} },

  // --- tool allowlist and overrides ---
  {
    label: "existing allowlist enforces and emits allowed_tools",
    values: { ...ROOT },
    ui: { hasExistingToolAllowlist: true, allowedTools: ["alpha"] },
  },
  {
    label: "allowlist interaction with an empty list still enforces",
    values: { ...ROOT },
    ui: { hasToolAllowlistInteraction: true },
  },
  {
    label: "tool display name and description maps present",
    values: { ...ROOT },
    ui: { toolNameToDisplayName: { a: "Alpha" }, toolNameToDescription: { a: "desc" } },
  },

  // --- cost config, logo, token validation ---
  // mcp_info.server_name walks a six-step fallback chain. Every step needs a case whose
  // earlier terms are falsy, or the chain is unreachable and a mutation deleting it survives.
  { label: "mcp_info server name falls back to the form url", values: { ...ROOT, server_name: "" }, ui: {} },
  {
    label: "mcp_info server name falls back to the stored server_name",
    values: { ...ROOT, server_name: "", url: "" },
    ui: {},
  },
  {
    label: "mcp_info server name falls back to the stored url",
    values: { ...ROOT, server_name: "", url: "" },
    ui: { mcpServer: { ...SERVER, server_name: "" } },
  },
  {
    label: "mcp_info server name falls back to the form alias",
    values: { ...ROOT, server_name: "", url: "" },
    ui: { mcpServer: { ...SERVER, server_name: "", url: "" } },
  },
  {
    label: "mcp_info server name falls back to unknown",
    values: { ...ROOT, server_name: "", url: "", alias: "" },
    ui: { mcpServer: { ...SERVER, server_name: "", url: "", alias: "" } },
  },

  { label: "cost config present", values: { ...ROOT }, ui: { costConfig: { default_cost_per_query: 0.01 } as never } },
  { label: "logo url present", values: { ...ROOT }, ui: { logoUrl: "https://cdn/logo.png" } },
  { label: "token validation JSON parses", values: { ...ROOT, token_validation_json: '{"aud":"x"}' }, ui: {} },
  {
    label: "token validation blank clears an existing value",
    values: { ...ROOT, token_validation_json: "   " },
    ui: { mcpServer: { ...SERVER, token_validation: { aud: "old" } } },
  },
  {
    label: "token validation blank with no existing value omits the key",
    values: { ...ROOT, token_validation_json: "" },
    ui: {},
  },

  // --- the six failure branches ---
  { label: "ERR invalid tool display name", values: { ...ROOT }, ui: { toolNameToDisplayName: { a: "has spaces" } } },
  {
    label: "ERR stdio config missing a command",
    values: { ...ROOT, transport: "stdio", stdio_config: JSON.stringify({ args: [] }) },
    ui: {},
  },
  {
    label: "ERR stdio config invalid JSON",
    values: { ...ROOT, transport: "stdio", stdio_config: "{not json" },
    ui: {},
  },
  {
    label: "ERR stdio env invalid JSON",
    values: { ...ROOT, transport: "stdio", command: "npx", env_json: "{not json" },
    ui: {},
  },
  {
    label: "ERR stdio dedicated path with a blank command",
    values: { ...ROOT, transport: "stdio", command: "   " },
    ui: {},
  },
  { label: "ERR token validation invalid JSON", values: { ...ROOT, token_validation_json: "{not json" }, ui: {} },
];
