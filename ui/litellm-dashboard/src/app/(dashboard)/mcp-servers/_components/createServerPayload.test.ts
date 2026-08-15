import { describe, expect, it } from "vitest";
import {
  BuildCreatePayloadResult,
  CreateServerUiState,
  buildCreateServerPayload,
  parseStdioConfig,
  reduceStaticHeaders,
} from "./createServerPayload";

const baseUi: CreateServerUiState = {
  transportType: "http",
  costConfig: {},
  allowedTools: [],
  hasToolAllowlistInteraction: false,
  toolNameToDisplayName: {},
  toolNameToDescription: {},
  logoUrl: undefined,
  dcrClient: null,
};

/** Narrow to the success branch so a regression surfaces as a failed assertion, not a type error. */
const payloadOf = (result: BuildCreatePayloadResult): Record<string, unknown> => {
  expect(result.kind).toBe("ok");
  if (result.kind !== "ok") throw new Error("unreachable");
  return result.payload;
};

const build = (values: Record<string, unknown>, ui: Partial<CreateServerUiState> = {}) =>
  buildCreateServerPayload(values, { ...baseUi, ...ui });

describe("reduceStaticHeaders", () => {
  it("returns an empty map for a non-array", () => {
    expect(reduceStaticHeaders(undefined)).toEqual({});
    expect(reduceStaticHeaders("X-Api-Key: v")).toEqual({});
  });

  it("trims header and value and drops rows with a blank header", () => {
    expect(
      reduceStaticHeaders([
        { header: "  X-Api-Key  ", value: "  secret  " },
        { header: "   ", value: "orphaned" },
        { header: "X-Empty" },
      ]),
    ).toEqual({ "X-Api-Key": "secret", "X-Empty": "" });
  });

  it("keeps the last value when a header repeats", () => {
    expect(
      reduceStaticHeaders([
        { header: "X-Dup", value: "first" },
        { header: "X-Dup", value: "second" },
      ]),
    ).toEqual({ "X-Dup": "second" });
  });
});

describe("parseStdioConfig", () => {
  it("reads a direct command/args/env config", () => {
    const result = parseStdioConfig('{"command":"npx","args":["-y","srv"],"env":{"TOKEN":"t"}}');
    expect(result).toEqual({
      kind: "ok",
      fields: { command: "npx", args: ["-y", "srv"], env: { TOKEN: "t" } },
    });
  });

  it("unwraps the mcpServers form and derives the server name with underscores", () => {
    const result = parseStdioConfig('{"mcpServers":{"my-github-server":{"command":"npx","args":["-y"]}}}');
    expect(result).toEqual({
      kind: "ok",
      fields: { command: "npx", args: ["-y"], env: undefined },
      derivedServerName: "my_github_server",
    });
  });

  it("takes the first server when mcpServers holds several", () => {
    const result = parseStdioConfig('{"mcpServers":{"first":{"command":"a"},"second":{"command":"b"}}}');
    expect(result).toMatchObject({ kind: "ok", fields: { command: "a" }, derivedServerName: "first" });
  });

  it("treats an empty mcpServers object as a direct config rather than deriving a name", () => {
    const result = parseStdioConfig('{"mcpServers":{},"command":"direct"}');
    expect(result).toEqual({ kind: "ok", fields: { command: "direct", args: undefined, env: undefined } });
  });

  it.each([["not json{"], ["null"]])("reports %s as invalid", (raw) => {
    expect(parseStdioConfig(raw)).toEqual({ kind: "invalid" });
  });
});

describe("buildCreateServerPayload validation", () => {
  it("rejects a tool display name containing a space and names the offender", () => {
    const result = build({ auth_type: "none" }, { toolNameToDisplayName: { search: "My Tool" } });
    expect(result).toEqual({ kind: "invalid_tool_display_name", displayName: "My Tool" });
  });

  it("accepts letters, digits, underscores and hyphens in a display name", () => {
    const result = build({ auth_type: "none" }, { toolNameToDisplayName: { search: "my-tool_2" } });
    expect(result.kind).toBe("ok");
  });

  it("rejects unparseable stdio JSON only when the stdio transport is selected", () => {
    expect(build({ stdio_config: "{oops" }, { transportType: "stdio" })).toEqual({ kind: "invalid_stdio_json" });
    // The same bad string on an http server is an inert leftover field, not a submit blocker.
    expect(build({ stdio_config: "{oops" }, { transportType: "http" }).kind).toBe("ok");
  });

  it("rejects unparseable token validation JSON", () => {
    expect(build({ token_validation_json: "not-valid-json{" })).toEqual({ kind: "invalid_token_validation_json" });
  });

  it("ignores a whitespace-only token validation body", () => {
    const payload = payloadOf(build({ token_validation_json: "   " }));
    expect(payload).not.toHaveProperty("token_validation");
  });

  it("includes parsed token validation rules when the JSON is valid", () => {
    const payload = payloadOf(build({ token_validation_json: '{"organization":"my-org","team.id":"42"}' }));
    expect(payload.token_validation).toEqual({ organization: "my-org", "team.id": "42" });
  });
});

describe("buildCreateServerPayload transport and naming", () => {
  it("maps the UI-only openapi transport to http for the backend", () => {
    const payload = payloadOf(build({ transport: "openapi", spec_path: "https://api.example.com/openapi.json" }));
    expect(payload.transport).toBe("http");
  });

  it("leaves http and sse transports untouched", () => {
    expect(payloadOf(build({ transport: "sse" })).transport).toBe("sse");
  });

  it("falls back to the stdio JSON's server key when the name field is blank", () => {
    const payload = payloadOf(
      build(
        { transport: "stdio", stdio_config: '{"mcpServers":{"my-server":{"command":"npx"}}}' },
        { transportType: "stdio" },
      ),
    );
    expect(payload.server_name).toBe("my_server");
    expect(payload.command).toBe("npx");
  });

  it("keeps an explicit server name over the stdio JSON's key", () => {
    const payload = payloadOf(
      build(
        { server_name: "Chosen", transport: "stdio", stdio_config: '{"mcpServers":{"my-server":{"command":"npx"}}}' },
        { transportType: "stdio" },
      ),
    );
    expect(payload.server_name).toBe("Chosen");
  });

  it("falls back to the url for mcp_info.server_name when no name is given", () => {
    const payload = payloadOf(build({ url: "https://example.com/mcp" }));
    expect((payload.mcp_info as Record<string, unknown>).server_name).toBe("https://example.com/mcp");
  });
});

describe("buildCreateServerPayload credentials", () => {
  it("drops empty, null and undefined credential entries", () => {
    const payload = payloadOf(
      build({ auth_type: "api_key", credentials: { auth_value: "secret", client_id: "", client_secret: null } }),
    );
    expect(payload.credentials).toEqual({ auth_value: "secret" });
  });

  it("filters blank scopes and omits the key when none survive", () => {
    expect(
      payloadOf(build({ auth_type: "oauth2", credentials: { client_id: "c", scopes: ["read", "", null] } }))
        .credentials,
    ).toEqual({ client_id: "c", scopes: ["read"] });
    expect(payloadOf(build({ auth_type: "oauth2", credentials: { client_id: "c", scopes: [] } })).credentials).toEqual({
      client_id: "c",
    });
  });

  it("omits credentials entirely for an auth type that needs none", () => {
    const payload = payloadOf(build({ auth_type: "none", credentials: { auth_value: "stale" } }));
    expect(payload).not.toHaveProperty("credentials");
  });

  it.each([["true_passthrough"], ["oauth_delegate"]])(
    "persists only the declared app for %s, never minted token material",
    (authType) => {
      const payload = payloadOf(
        build({
          auth_type: authType,
          credentials: {
            client_id: "org-app",
            client_secret: "org-secret",
            access_token: "upstream-tok",
            refresh_token: "refresh-tok",
            expires_in: 3600,
            scope: "read",
          },
        }),
      );
      expect(payload.credentials).toEqual({ client_id: "org-app", client_secret: "org-secret" });
      expect(JSON.stringify(payload)).not.toContain("upstream-tok");
      expect(JSON.stringify(payload)).not.toContain("refresh-tok");
    },
  );

  it("merges the DCR-minted client into an oauth2 payload", () => {
    const payload = payloadOf(
      build(
        { auth_type: "oauth2", credentials: { access_token: "tok" } },
        { dcrClient: { client_id: "dcr-id", client_secret: "dcr-secret" } },
      ),
    );
    expect(payload.credentials).toMatchObject({
      client_id: "dcr-id",
      client_secret: "dcr-secret",
      access_token: "tok",
    });
  });

  it("never leaks the DCR-minted client onto a non-oauth2 server", () => {
    const payload = payloadOf(
      build({ auth_type: "true_passthrough" }, { dcrClient: { client_id: "dcr-id", client_secret: "dcr-secret" } }),
    );
    expect(JSON.stringify(payload)).not.toContain("dcr-id");
  });
});

describe("buildCreateServerPayload flags", () => {
  it.each([["true_passthrough"], ["oauth_delegate"]])("defaults dcr_bridge on for %s", (authType) => {
    expect(payloadOf(build({ auth_type: authType })).dcr_bridge).toBe(true);
  });

  it.each([["true_passthrough"], ["oauth_delegate"]])("honours an explicit dcr_bridge false for %s", (authType) => {
    expect(payloadOf(build({ auth_type: authType, dcr_bridge: false })).dcr_bridge).toBe(false);
  });

  it.each([["none"], ["api_key"], ["oauth2"]])(
    "forces dcr_bridge off for %s even when the form still holds true",
    (authType) => {
      expect(payloadOf(build({ auth_type: authType, dcr_bridge: true })).dcr_bridge).toBe(false);
    },
  );

  it("stamps the interactive oauth2 flow by default", () => {
    expect(payloadOf(build({ auth_type: "oauth2" })).oauth2_flow).toBe("authorization_code");
  });

  it("stamps client_credentials for an M2M oauth2 server", () => {
    expect(payloadOf(build({ auth_type: "oauth2", oauth_flow_type: "m2m" })).oauth2_flow).toBe("client_credentials");
  });

  it("sends no oauth2_flow for a non-oauth2 server", () => {
    expect(payloadOf(build({ auth_type: "api_key", oauth_flow_type: "m2m" }))).not.toHaveProperty("oauth2_flow");
  });

  it.each([["allow_all_keys"], ["available_on_public_internet"], ["delegate_auth_to_upstream"], ["oauth_passthrough"]])(
    "coerces %s to a boolean",
    (key) => {
      expect(payloadOf(build({ auth_type: "none" }))[key]).toBe(false);
      expect(payloadOf(build({ auth_type: "none", [key]: true }))[key]).toBe(true);
    },
  );
});

describe("buildCreateServerPayload tool allowlist", () => {
  it("marks the allowlist enforced once the admin has touched it, even with nothing selected", () => {
    const payload = payloadOf(build({ auth_type: "none" }, { hasToolAllowlistInteraction: true }));
    expect((payload.mcp_info as Record<string, unknown>).tool_allowlist_enforced).toBe(true);
    expect(payload.allowed_tools).toEqual([]);
  });

  it("marks the allowlist enforced when tools are selected without an explicit interaction", () => {
    const payload = payloadOf(build({ auth_type: "none" }, { allowedTools: ["search"] }));
    expect((payload.mcp_info as Record<string, unknown>).tool_allowlist_enforced).toBe(true);
    expect(payload.allowed_tools).toEqual(["search"]);
  });

  it("leaves the allowlist unenforced when untouched and empty", () => {
    const payload = payloadOf(build({ auth_type: "none" }));
    expect((payload.mcp_info as Record<string, unknown>).tool_allowlist_enforced).toBe(false);
  });
});

describe("buildCreateServerPayload mcp_info", () => {
  it("sends a null cost map when nothing is configured and the map when it is", () => {
    expect(
      (payloadOf(build({ auth_type: "none" })).mcp_info as Record<string, unknown>).mcp_server_cost_info,
    ).toBeNull();
    const priced = payloadOf(build({ auth_type: "none" }, { costConfig: { default_cost_per_query: 0.01 } }));
    expect((priced.mcp_info as Record<string, unknown>).mcp_server_cost_info).toEqual({ default_cost_per_query: 0.01 });
  });

  it("carries the selected logo and drops the raw stdio_config field", () => {
    const payload = payloadOf(build({ auth_type: "none", stdio_config: "{}" }, { logoUrl: "https://cdn/logo.png" }));
    expect((payload.mcp_info as Record<string, unknown>).logo_url).toBe("https://cdn/logo.png");
    expect(payload.stdio_config).toBeUndefined();
  });
});
