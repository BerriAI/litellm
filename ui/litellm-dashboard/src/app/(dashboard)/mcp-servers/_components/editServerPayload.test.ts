import { describe, it, expect } from "vitest";

import { buildEditServerPayload, EditServerUiState } from "./editServerPayload";
import { MCPServer } from "@/components/mcp_tools/types";

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

const UI: EditServerUiState = {
  costConfig: {},
  allowedTools: [],
  hasExistingToolAllowlist: false,
  hasToolAllowlistInteraction: false,
  toolNameToDisplayName: {},
  toolNameToDescription: {},
  logoUrl: undefined,
  removeStoredApp: false,
};

const VALUES = { transport: "http", auth_type: "none", url: "https://example.com/mcp" };

const build = (values: Record<string, unknown> = {}, ui: Partial<EditServerUiState> = {}, server: MCPServer = SERVER) =>
  buildEditServerPayload({ ...VALUES, ...values }, { ...UI, ...ui }, server);

const ok = (result: ReturnType<typeof build>): Record<string, unknown> => {
  expect(result.kind).toBe("ok");
  if (result.kind !== "ok") throw new Error("unreachable");
  return result.payload;
};

describe("buildEditServerPayload rejections", () => {
  it("rejects a tool display name with a space before anything else is parsed", () => {
    const result = build({ stdio_config: "{" }, { toolNameToDisplayName: { a: "bad name" } });
    expect(result).toStrictEqual({ kind: "invalid_tool_display_name", displayName: "bad name" });
  });

  it("accepts a display name made of letters, digits, underscores and hyphens", () => {
    expect(build({}, { toolNameToDisplayName: { a: "Good_name-1" } }).kind).toBe("ok");
  });

  it("rejects unparseable stdio config json", () => {
    expect(build({ transport: "stdio", stdio_config: "{not json" }).kind).toBe("invalid_stdio_json");
  });

  it("rejects stdio config json that parses but carries no command", () => {
    const result = build({ transport: "stdio", stdio_config: JSON.stringify({ args: ["x"] }) });
    expect(result.kind).toBe("stdio_config_missing_command");
  });

  it("rejects unparseable stdio env json on the dedicated-fields path", () => {
    const result = build({ transport: "stdio", command: "npx", env_json: "{not json" });
    expect(result.kind).toBe("invalid_stdio_env_json");
  });

  it("rejects a blank command on the dedicated-fields path", () => {
    expect(build({ transport: "stdio", command: "   " }).kind).toBe("stdio_missing_command");
  });

  it("rejects unparseable token validation json", () => {
    expect(build({ token_validation_json: "{not json" }).kind).toBe("invalid_token_validation_json");
  });

  it("ignores a whitespace-only token validation body rather than rejecting it", () => {
    expect(build({ token_validation_json: "   " }).kind).toBe("ok");
  });

  it("never leaves the stdio branch reachable for a non-stdio transport", () => {
    expect(build({ transport: "http", stdio_config: "{not json" }).kind).toBe("ok");
  });
});

describe("buildEditServerPayload stdio parsing", () => {
  it("unwraps a pasted client config keyed under mcpServers", () => {
    const config = JSON.stringify({ mcpServers: { fs: { command: "npx", args: ["-y", "pkg"], env: { A: "1" } } } });
    const payload = ok(build({ transport: "stdio", stdio_config: config }));
    expect(payload.command).toBe("npx");
    expect(payload.args).toStrictEqual(["-y", "pkg"]);
    expect(payload.env).toStrictEqual({ A: "1" });
  });

  it("accepts a bare command object with no mcpServers wrapper", () => {
    const config = JSON.stringify({ command: "uvx", args: [], env: {} });
    expect(ok(build({ transport: "stdio", stdio_config: config })).command).toBe("uvx");
  });

  it("stringifies non-string args and drops blank ones", () => {
    const config = JSON.stringify({ command: "npx", args: [1, "  ", "keep", true] });
    expect(ok(build({ transport: "stdio", stdio_config: config })).args).toStrictEqual(["1", "keep", "true"]);
  });

  it("stringifies env values and drops blank keys", () => {
    const config = JSON.stringify({ command: "npx", env: { A: 1, "": "x", B: null } });
    expect(ok(build({ transport: "stdio", stdio_config: config })).env).toStrictEqual({ A: "1", B: "" });
  });

  it("trims the command on the dedicated-fields path", () => {
    expect(ok(build({ transport: "stdio", command: "  npx  " })).command).toBe("npx");
  });

  it("defaults env to an empty object when no env json is supplied", () => {
    expect(ok(build({ transport: "stdio", command: "npx" })).env).toStrictEqual({});
  });

  it("prefers the pasted json over the dedicated fields when both are present", () => {
    const config = JSON.stringify({ command: "from-json" });
    expect(ok(build({ transport: "stdio", stdio_config: config, command: "from-field" })).command).toBe("from-json");
  });
});

describe("buildEditServerPayload credentials", () => {
  it("drops blank credential values so the backend keeps what it stored", () => {
    const payload = ok(build({ auth_type: "api_key", credentials: { auth_value: "", client_id: "cid" } }));
    expect(payload.credentials).toStrictEqual({ client_id: "cid" });
  });

  it("sends an explicit null for a blanked admin-config key so the stored value is cleared", () => {
    const payload = ok(build({ auth_type: "oauth2", credentials: { upstream_resource: "", client_id: "cid" } }));
    expect(payload.credentials).toStrictEqual({ upstream_resource: null, client_id: "cid" });
  });

  it("drops a scopes array once every entry is blank", () => {
    const payload = ok(build({ auth_type: "oauth2", credentials: { client_id: "cid", scopes: ["", null] } }));
    expect(payload.credentials).toStrictEqual({ client_id: "cid" });
  });

  it("keeps the surviving scopes when only some are blank", () => {
    const payload = ok(build({ auth_type: "oauth2", credentials: { client_id: "cid", scopes: ["read", ""] } }));
    expect(payload.credentials).toStrictEqual({ client_id: "cid", scopes: ["read"] });
  });

  it("omits credentials entirely for an auth type that takes none", () => {
    const payload = ok(build({ auth_type: "none", credentials: { auth_value: "x" } }));
    expect(payload).not.toHaveProperty("credentials");
  });

  it("writes explicit nulls when a stored app is being removed", () => {
    const payload = ok(
      build({ auth_type: "true_passthrough", credentials: { client_id: "cid" } }, { removeStoredApp: true }),
    );
    expect(payload.credentials).toStrictEqual({ client_id: null, client_secret: null });
  });

  it("ignores removeStoredApp for a mode that does not forward client tokens", () => {
    const payload = ok(build({ auth_type: "api_key", credentials: { auth_value: "v" } }, { removeStoredApp: true }));
    expect(payload.credentials).toStrictEqual({ auth_value: "v" });
  });
});

describe("buildEditServerPayload flags and clearing", () => {
  it("maps the openapi transport to http for the backend", () => {
    expect(ok(build({ transport: "openapi" })).transport).toBe("http");
  });

  it("leaves a transport the backend already understands alone", () => {
    expect(ok(build({ transport: "http" })).transport).toBe("http");
  });

  it("nulls the oauth2 endpoints when moving off oauth2", () => {
    const payload = ok(build({ auth_type: "api_key" }, {}, { ...SERVER, auth_type: "oauth2" }));
    expect(payload.issuer).toBeNull();
    expect(payload.authorization_url).toBeNull();
    expect(payload.token_url).toBeNull();
    expect(payload.registration_url).toBeNull();
  });

  it("leaves the oauth2 endpoints untouched when staying on oauth2", () => {
    const payload = ok(build({ auth_type: "oauth2" }, {}, { ...SERVER, auth_type: "oauth2" }));
    expect(payload).not.toHaveProperty("issuer");
  });

  it("nulls the token exchange fields when moving off token exchange", () => {
    const payload = ok(build({ auth_type: "none" }, {}, { ...SERVER, auth_type: "oauth2_token_exchange" }));
    expect(payload.token_exchange_endpoint).toBeNull();
    expect(payload.audience).toBeNull();
    expect(payload.subject_token_type).toBeNull();
    expect(payload.token_exchange_profile).toBeNull();
  });

  it("forces delegate_auth_to_upstream false for any auth type other than oauth2", () => {
    const payload = ok(build({ auth_type: "api_key", delegate_auth_to_upstream: true }));
    expect(payload.delegate_auth_to_upstream).toBe(false);
  });

  it("honours delegate_auth_to_upstream for oauth2", () => {
    const payload = ok(build({ auth_type: "oauth2", delegate_auth_to_upstream: true }));
    expect(payload.delegate_auth_to_upstream).toBe(true);
  });

  it("forces oauth_passthrough false unless an Authorization header is forwarded", () => {
    const payload = ok(build({ auth_type: "none", oauth_passthrough: true, extra_headers: [] }));
    expect(payload.oauth_passthrough).toBe(false);
  });

  it("honours oauth_passthrough for a none-auth server forwarding Authorization", () => {
    const payload = ok(build({ auth_type: "none", oauth_passthrough: true, extra_headers: ["authorization"] }));
    expect(payload.oauth_passthrough).toBe(true);
  });

  it("forces dcr_bridge false outside the client-forwarded modes", () => {
    expect(ok(build({ auth_type: "oauth2", dcr_bridge: true })).dcr_bridge).toBe(false);
  });

  it("honours dcr_bridge for a client-forwarded mode", () => {
    expect(ok(build({ auth_type: "true_passthrough", dcr_bridge: true })).dcr_bridge).toBe(true);
  });

  it("sends token_validation as null to clear a value the server already had", () => {
    const payload = ok(build({ token_validation_json: "" }, {}, { ...SERVER, token_validation: { a: 1 } }));
    expect(payload.token_validation).toBeNull();
  });

  it("omits token_validation entirely when neither side has one", () => {
    expect(ok(build({ token_validation_json: "" }))).not.toHaveProperty("token_validation");
  });

  it("falls back through the name chain to the server url", () => {
    const server = { ...SERVER, server_name: "", url: "https://fallback" };
    const payload = ok(build({ server_name: "", url: "" }, {}, server));
    expect((payload.mcp_info as Record<string, unknown>).server_name).toBe("https://fallback");
  });

  it("falls back to the literal unknown when every name candidate is blank", () => {
    const server = { ...SERVER, server_name: "", url: "", alias: "" };
    const payload = ok(build({ server_name: "", url: "", alias: "" }, {}, server));
    expect((payload.mcp_info as Record<string, unknown>).server_name).toBe("unknown");
  });

  it("emits allowed_tools only once the allowlist is enforced", () => {
    expect(ok(build())).not.toHaveProperty("allowed_tools");
    expect(ok(build({}, { allowedTools: ["a"] })).allowed_tools).toStrictEqual(["a"]);
    expect(ok(build({}, { hasExistingToolAllowlist: true })).allowed_tools).toStrictEqual([]);
  });

  it("normalises object-shaped access groups to their names", () => {
    const payload = ok(build({ mcp_access_groups: [{ name: "eng" }, "ops"] }));
    expect(payload.mcp_access_groups).toStrictEqual(["eng", "ops"]);
  });
});
