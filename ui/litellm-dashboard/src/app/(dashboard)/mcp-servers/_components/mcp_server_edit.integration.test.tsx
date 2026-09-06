import React from "react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor, act } from "@testing-library/react";

import MCPServerEdit from "./mcp_server_edit";
import * as networking from "@/components/networking";
import { MCPServer } from "@/components/mcp_tools/types";

vi.mock("@/components/networking", () => ({
  updateMCPServer: vi.fn(),
  listMCPTools: vi.fn().mockResolvedValue({ tools: [], error: null }),
  storeMCPOAuthUserCredential: vi.fn().mockResolvedValue({}),
  testMCPToolsListRequest: vi.fn().mockResolvedValue({ tools: [], error: null }),
}));

vi.mock("@/hooks/useMcpOAuthFlow", () => ({
  useMcpOAuthFlow: () => ({
    startOAuthFlow: vi.fn(),
    status: "idle",
    error: null,
    tokenResponse: null,
    reset: vi.fn(),
  }),
}));

vi.mock("./mcp_server_cost_config", () => ({
  default: () => <div data-testid="mcp-cost-config" />,
}));

vi.mock("./mcp_tool_configuration", () => ({
  default: () => <div data-testid="mcp-tool-config" />,
}));

const BASE: MCPServer = {
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

const CREDS = {
  auth_value: "secret-value",
  client_id: "cid",
  client_secret: "csec",
  scopes: "read write",
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

const EXPECTED_BASE: Readonly<Record<string, unknown>> = {
  alias: "srv_alias",
  allow_all_keys: false,
  auth_type: "none",
  available_on_public_internet: true,
  dcr_bridge: false,
  delegate_auth_to_upstream: false,
  description: "a server",
  disallowed_tools: [],
  env_json: undefined,
  env_vars: [],
  extra_headers: [],
  max_concurrent_requests: undefined,
  mcp_access_groups: [],
  mcp_info: {
    description: "a server",
    logo_url: undefined,
    mcp_server_cost_info: null,
    server_name: "srv",
    tool_allowlist_enforced: false,
  },
  oauth_passthrough: false,
  server_id: "srv_1",
  server_name: "srv",
  static_headers: {},
  stdio_config: undefined,
  tool_name_to_description: null,
  tool_name_to_display_name: null,
  transport: "http",
  url: "https://example.com/mcp",
};

const without = (keys: readonly string[]): Record<string, unknown> =>
  Object.fromEntries(Object.entries(EXPECTED_BASE).filter(([k]) => !keys.includes(k)));

interface Case {
  readonly label: string;
  readonly server: MCPServer;
  readonly expected: Record<string, unknown>;
}

const CASES: readonly Case[] = [
  { label: "http + none", server: BASE, expected: { ...EXPECTED_BASE } },
  {
    label: "stdio drops auth_type and url and adds the stdio trio",
    server: { ...BASE, transport: "stdio", url: null, command: "npx", args: ["-y", "pkg"], env: { A: "1" } },
    expected: { ...without(["auth_type", "url"]), args: ["-y", "pkg"], command: "npx", env: {}, transport: "stdio" },
  },
  {
    label: "openapi swaps url for spec_path",
    server: { ...BASE, transport: "openapi", url: null, spec_path: "https://example.com/openapi.json" },
    expected: { ...without(["url"]), spec_path: "https://example.com/openapi.json" },
  },
  {
    label: "api_key sends only auth_value under credentials",
    server: { ...BASE, auth_type: "api_key", credentials: CREDS },
    expected: { ...EXPECTED_BASE, auth_type: "api_key", credentials: { auth_value: "secret-value" } },
  },
  {
    label: "bearer_token sends only auth_value under credentials",
    server: { ...BASE, auth_type: "bearer_token", credentials: CREDS },
    expected: { ...EXPECTED_BASE, auth_type: "bearer_token", credentials: { auth_value: "secret-value" } },
  },
  {
    label: "basic sends only auth_value under credentials",
    server: { ...BASE, auth_type: "basic", credentials: CREDS },
    expected: { ...EXPECTED_BASE, auth_type: "basic", credentials: { auth_value: "secret-value" } },
  },
  {
    label: "oauth2 m2m mounts token_url and omits the interactive endpoints entirely",
    server: {
      ...BASE,
      auth_type: "oauth2",
      oauth2_flow: "client_credentials",
      token_url: "https://idp/token",
      credentials: CREDS,
    },
    expected: {
      ...EXPECTED_BASE,
      auth_type: "oauth2",
      credentials: {
        client_id: "cid",
        client_secret: "csec",
        token_endpoint_auth_method: "client_secret_basic",
        upstream_resource: "api://up",
      },
      oauth2_flow: "client_credentials",
      oauth_flow_type: "m2m",
      token_url: "https://idp/token",
    },
  },
  {
    label: "oauth2 interactive mounts the full endpoint set, empty ones present as undefined",
    server: {
      ...BASE,
      auth_type: "oauth2",
      oauth2_flow: "authorization_code",
      issuer: "https://idp",
      credentials: CREDS,
    },
    expected: {
      ...EXPECTED_BASE,
      auth_type: "oauth2",
      authorization_url: undefined,
      credentials: {
        client_id: "cid",
        client_secret: "csec",
        token_endpoint_auth_method: "client_secret_basic",
        upstream_resource: "api://up",
      },
      issuer: "https://idp",
      oauth2_flow: "authorization_code",
      oauth_flow_type: "interactive",
      registration_url: undefined,
      token_storage_ttl_seconds: undefined,
      token_url: undefined,
    },
  },
  {
    label: "token_exchange mounts audience and subject_token_type as undefined",
    server: {
      ...BASE,
      auth_type: "oauth2_token_exchange",
      token_exchange_endpoint: "https://idp/x",
      credentials: CREDS,
    },
    expected: {
      ...EXPECTED_BASE,
      audience: undefined,
      auth_type: "oauth2_token_exchange",
      credentials: { client_id: "cid", client_secret: "csec" },
      subject_token_type: undefined,
      token_exchange_endpoint: "https://idp/x",
      token_exchange_profile: undefined,
    },
  },
  {
    label: "token_exchange entra_obo omits audience and subject_token_type as keys",
    server: {
      ...BASE,
      auth_type: "oauth2_token_exchange",
      token_exchange_profile: "entra_obo",
      token_exchange_endpoint: "https://idp/x",
      credentials: CREDS,
    },
    expected: {
      ...EXPECTED_BASE,
      auth_type: "oauth2_token_exchange",
      credentials: { client_id: "cid", client_secret: "csec" },
      token_exchange_endpoint: "https://idp/x",
      token_exchange_profile: "entra_obo",
    },
  },
  {
    label: "id_jag sends its seven credential fields",
    server: { ...BASE, auth_type: "oauth2_id_jag", credentials: CREDS },
    expected: {
      ...EXPECTED_BASE,
      audience: undefined,
      auth_type: "oauth2_id_jag",
      credentials: {
        client_assertion_signing_alg: "RS256",
        client_id: "cid",
        client_private_key: "-----KEY-----",
        client_private_key_id: "kid",
        client_secret: "csec",
        id_jag_resource: "api://res",
        id_jag_resource_token_endpoint: "https://idp/jag",
      },
      subject_token_type: undefined,
      token_exchange_endpoint: undefined,
    },
  },
  {
    label: "aws_sigv4 sends its seven credential fields",
    server: { ...BASE, auth_type: "aws_sigv4", credentials: CREDS },
    expected: {
      ...EXPECTED_BASE,
      auth_type: "aws_sigv4",
      credentials: {
        aws_access_key_id: "AKIA",
        aws_region_name: "us-east-1",
        aws_role_name: "role",
        aws_secret_access_key: "shh",
        aws_service_name: "bedrock",
        aws_session_name: "sess",
        aws_session_token: "tok",
      },
    },
  },
  {
    label: "true_passthrough sends only the passthrough app credentials",
    server: { ...BASE, auth_type: "true_passthrough", credentials: CREDS },
    expected: {
      ...EXPECTED_BASE,
      auth_type: "true_passthrough",
      credentials: { client_id: "cid", client_secret: "csec" },
    },
  },
  {
    label: "oauth_delegate sends only the passthrough app credentials",
    server: { ...BASE, auth_type: "oauth_delegate", credentials: CREDS },
    expected: {
      ...EXPECTED_BASE,
      auth_type: "oauth_delegate",
      credentials: { client_id: "cid", client_secret: "csec" },
    },
  },
  {
    label: "an Authorization entry survives in extra_headers without enabling passthrough",
    server: { ...BASE, auth_type: "none", extra_headers: ["Authorization"] },
    expected: { ...EXPECTED_BASE, extra_headers: ["Authorization"] },
  },
  {
    label: "dcr_bridge stays true for a client-forwarded mode",
    server: { ...BASE, auth_type: "true_passthrough", dcr_bridge: true },
    expected: { ...EXPECTED_BASE, auth_type: "true_passthrough", dcr_bridge: true },
  },
  {
    label: "an existing allowed_tools list enforces the allowlist and round-trips",
    server: { ...BASE, allowed_tools: ["alpha"] },
    expected: {
      ...EXPECTED_BASE,
      allowed_tools: ["alpha"],
      mcp_info: { ...(EXPECTED_BASE.mcp_info as object), tool_allowlist_enforced: true },
    },
  },
  {
    label: "no allowed_tools key when the allowlist is not enforced",
    server: { ...BASE, allowed_tools: [] },
    expected: { ...EXPECTED_BASE },
  },
];

const saveAndCapture = async (server: MCPServer): Promise<Record<string, unknown>> => {
  vi.mocked(networking.updateMCPServer).mockResolvedValue(server as never);
  render(
    <MCPServerEdit
      mcpServer={server}
      accessToken="access-token"
      userID="user-1"
      onCancel={vi.fn()}
      onSuccess={vi.fn()}
      availableAccessGroups={[]}
    />,
  );
  await act(async () => {
    screen.getAllByRole("button", { name: "Save Changes" })[0].click();
  });
  await waitFor(() => {
    expect(networking.updateMCPServer).toHaveBeenCalled();
  });
  const [, payload] = vi.mocked(networking.updateMCPServer).mock.calls[0];
  return payload as Record<string, unknown>;
};

describe("mcp_server_edit save payload contract", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it.each(CASES.map((c) => [c.label, c] as const))("%s", async (_label, testCase) => {
    const payload = await saveAndCapture(testCase.server);
    expect(payload).toStrictEqual(testCase.expected);
  });

  it("never sends a server-owned field the form does not bind", async () => {
    const payload = await saveAndCapture({
      ...BASE,
      status: "healthy",
      last_health_check: "2024-01-02T00:00:00Z",
      health_check_error: null,
      teams: [{ team_id: "t1" }],
      allowed_tools: ["a"],
      has_user_credential: true,
      approval_status: "approved",
      submitted_by: "someone",
      submitted_at: "2024-01-01T00:00:00Z",
      reviewed_at: "2024-01-02T00:00:00Z",
      review_notes: "looks fine",
    } as MCPServer);

    for (const leaked of [
      "created_at",
      "created_by",
      "updated_at",
      "updated_by",
      "status",
      "last_health_check",
      "health_check_error",
      "teams",
      "has_user_credential",
      "approval_status",
      "submitted_by",
      "submitted_at",
      "reviewed_at",
      "review_notes",
    ]) {
      expect(payload).not.toHaveProperty(leaked);
    }
  });
});
