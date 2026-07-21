import { describe, it, expect } from "vitest";
import {
  AUTH_TYPE,
  OAUTH_FLOW,
  MCP_OAUTH2_FLOW_M2M,
  TRANSPORT,
  handleTransport,
  handleAuth,
  getMcpOAuthMode,
  getOAuthAuthorizationIdentity,
  isHeldOAuthTokenStale,
  oauth2FlowToFormValue,
  preservedDeclaredAppCredentials,
  withoutMintedTokenCredentials,
  credentialAuthClass,
} from "./types";

describe("getOAuthAuthorizationIdentity", () => {
  // Regression: the identity used to pick the audience from spec_path only when values.transport was
  // OPENAPI, but the create form keeps transport in component state, so values.transport was absent and
  // spec_path edits on OpenAPI servers never invalidated a held token.
  it("changes when spec_path changes even when transport is absent from form values", () => {
    const authorized = { auth_type: AUTH_TYPE.OAUTH2, spec_path: "https://a.example.com/openapi.json" };
    const edited = { auth_type: AUTH_TYPE.OAUTH2, spec_path: "https://b.example.com/openapi.json" };
    expect(getOAuthAuthorizationIdentity(edited)).not.toBe(getOAuthAuthorizationIdentity(authorized));
    expect(isHeldOAuthTokenStale(edited, getOAuthAuthorizationIdentity(authorized))).toBe(true);
  });

  it("changes when url changes", () => {
    const authorized = { auth_type: AUTH_TYPE.OAUTH2, url: "https://a.example.com/mcp" };
    const edited = { auth_type: AUTH_TYPE.OAUTH2, url: "https://b.example.com/mcp" };
    expect(getOAuthAuthorizationIdentity(edited)).not.toBe(getOAuthAuthorizationIdentity(authorized));
  });

  it("is stable across non-mint fields", () => {
    const authorized = { auth_type: AUTH_TYPE.OAUTH2, url: "https://a.example.com/mcp", server_name: "one" };
    const renamed = { auth_type: AUTH_TYPE.OAUTH2, url: "https://a.example.com/mcp", server_name: "two" };
    expect(getOAuthAuthorizationIdentity(renamed)).toBe(getOAuthAuthorizationIdentity(authorized));
    expect(isHeldOAuthTokenStale(renamed, getOAuthAuthorizationIdentity(authorized))).toBe(false);
  });
});

describe("handleTransport", () => {
  it("should default to SSE when transport is null", () => {
    expect(handleTransport(null)).toBe(TRANSPORT.SSE);
  });

  it("should default to SSE when transport is undefined", () => {
    expect(handleTransport(undefined)).toBe(TRANSPORT.SSE);
  });

  it("should return openapi when specPath is present and transport is not stdio", () => {
    expect(handleTransport("http", "/spec.yaml")).toBe(TRANSPORT.OPENAPI);
  });

  it("should keep stdio even when specPath is present", () => {
    expect(handleTransport(TRANSPORT.STDIO, "/spec.yaml")).toBe(TRANSPORT.STDIO);
  });

  it("should return the transport as-is when no specPath", () => {
    expect(handleTransport("http")).toBe("http");
  });
});

describe("handleAuth", () => {
  it("should default to NONE when authType is null", () => {
    expect(handleAuth(null)).toBe(AUTH_TYPE.NONE);
  });

  it("should default to NONE when authType is undefined", () => {
    expect(handleAuth(undefined)).toBe(AUTH_TYPE.NONE);
  });

  it("should return the provided auth type", () => {
    expect(handleAuth(AUTH_TYPE.OAUTH2)).toBe("oauth2");
  });
});

describe("constants", () => {
  it("should define all expected auth types", () => {
    expect(AUTH_TYPE.NONE).toBe("none");
    expect(AUTH_TYPE.API_KEY).toBe("api_key");
    expect(AUTH_TYPE.BEARER_TOKEN).toBe("bearer_token");
    expect(AUTH_TYPE.OAUTH2).toBe("oauth2");
  });

  it("should define all expected transport types", () => {
    expect(TRANSPORT.SSE).toBe("sse");
    expect(TRANSPORT.HTTP).toBe("http");
    expect(TRANSPORT.STDIO).toBe("stdio");
    expect(TRANSPORT.OPENAPI).toBe("openapi");
  });

  it("should define OAuth flow types", () => {
    expect(OAUTH_FLOW.INTERACTIVE).toBe("interactive");
    expect(OAUTH_FLOW.M2M).toBe("m2m");
  });

  it("should define the backend M2M flow value", () => {
    expect(MCP_OAUTH2_FLOW_M2M).toBe("client_credentials");
  });
});

describe("getMcpOAuthMode", () => {
  it("returns null for non-OAuth2 servers", () => {
    expect(getMcpOAuthMode({ auth_type: AUTH_TYPE.API_KEY })).toBeNull();
    expect(getMcpOAuthMode({ auth_type: AUTH_TYPE.NONE })).toBeNull();
    expect(getMcpOAuthMode({})).toBeNull();
  });

  it("classifies client_credentials as m2m", () => {
    expect(getMcpOAuthMode({ auth_type: AUTH_TYPE.OAUTH2, oauth2_flow: MCP_OAUTH2_FLOW_M2M })).toBe("m2m");
  });

  it("classifies oauth2_token_exchange as token_exchange regardless of the oauth2 secondary fields", () => {
    expect(getMcpOAuthMode({ auth_type: AUTH_TYPE.OAUTH2_TOKEN_EXCHANGE })).toBe("token_exchange");
    expect(
      getMcpOAuthMode({
        auth_type: AUTH_TYPE.OAUTH2_TOKEN_EXCHANGE,
        oauth2_flow: MCP_OAUTH2_FLOW_M2M,
        delegate_auth_to_upstream: true,
      }),
    ).toBe("token_exchange");
  });

  it("treats m2m as m2m even when delegate_auth_to_upstream is true", () => {
    expect(
      getMcpOAuthMode({
        auth_type: AUTH_TYPE.OAUTH2,
        oauth2_flow: MCP_OAUTH2_FLOW_M2M,
        delegate_auth_to_upstream: true,
      }),
    ).toBe("m2m");
  });

  it("classifies an interactive server with delegate_auth_to_upstream as passthrough", () => {
    expect(getMcpOAuthMode({ auth_type: AUTH_TYPE.OAUTH2, oauth2_flow: null, delegate_auth_to_upstream: true })).toBe(
      "passthrough",
    );
  });

  it("classifies an interactive server without delegation as authorization_code", () => {
    expect(getMcpOAuthMode({ auth_type: AUTH_TYPE.OAUTH2, oauth2_flow: null, delegate_auth_to_upstream: false })).toBe(
      "authorization_code",
    );
  });

  it("defaults to authorization_code when delegate_auth_to_upstream is undefined", () => {
    expect(getMcpOAuthMode({ auth_type: AUTH_TYPE.OAUTH2 })).toBe("authorization_code");
  });

  it("treats explicit authorization_code as interactive, not m2m", () => {
    expect(
      getMcpOAuthMode({
        auth_type: AUTH_TYPE.OAUTH2,
        oauth2_flow: "authorization_code",
        delegate_auth_to_upstream: false,
      }),
    ).toBe("authorization_code");
  });

  // Regression: the old heuristic labeled any OAuth2 server with a token endpoint
  // as M2M. getMcpOAuthMode ignores token_url, so an interactive server that
  // legitimately carries one is classified by oauth2_flow + delegate, never M2M.
  it("does not treat an interactive server with a token endpoint as m2m", () => {
    expect(getMcpOAuthMode({ auth_type: AUTH_TYPE.OAUTH2, oauth2_flow: null, delegate_auth_to_upstream: false })).toBe(
      "authorization_code",
    );
  });
});

describe("oauth2FlowToFormValue", () => {
  it("maps client_credentials to the M2M select value", () => {
    expect(oauth2FlowToFormValue(MCP_OAUTH2_FLOW_M2M)).toBe(OAUTH_FLOW.M2M);
  });

  it("maps authorization_code to the Interactive select value", () => {
    expect(oauth2FlowToFormValue("authorization_code")).toBe(OAUTH_FLOW.INTERACTIVE);
  });

  it("returns undefined for a null/unset flow so the select shows its placeholder", () => {
    expect(oauth2FlowToFormValue(null)).toBeUndefined();
    expect(oauth2FlowToFormValue(undefined)).toBeUndefined();
  });
});

describe("preservedDeclaredAppCredentials", () => {
  it("keeps only non-empty string declared-app keys and never token-shaped keys", () => {
    expect(preservedDeclaredAppCredentials(undefined)).toBeUndefined();
    expect(preservedDeclaredAppCredentials({})).toBeUndefined();
    expect(preservedDeclaredAppCredentials({ client_id: 123 })).toBeUndefined();
    expect(preservedDeclaredAppCredentials({ client_id: "" })).toBeUndefined();
    expect(preservedDeclaredAppCredentials({ client_id: "a", access_token: "t", scopes: ["s"] })).toEqual({
      client_id: "a",
    });
    expect(preservedDeclaredAppCredentials({ client_secret: "s" })).toEqual({ client_secret: "s" });
    expect(preservedDeclaredAppCredentials({ client_id: "a", client_secret: "b", refresh_token: "r" })).toEqual({
      client_id: "a",
      client_secret: "b",
    });
  });
});

describe("withoutMintedTokenCredentials", () => {
  it("drops token keys and keeps the declared app and other config", () => {
    expect(withoutMintedTokenCredentials(undefined)).toBeUndefined();
    const mixed = {
      client_id: "a",
      client_secret: "b",
      access_token: "t",
      refresh_token: "r",
      expires_in: 3600,
      scope: "read",
      scopes: ["read"],
    };
    expect(withoutMintedTokenCredentials(mixed)).toEqual({ client_id: "a", client_secret: "b", scopes: ["read"] });
  });

  it("returns undefined (not {}) when only minted keys are present, so a restore never blanks the fields", () => {
    expect(withoutMintedTokenCredentials({ access_token: "t", refresh_token: "r", expires_in: 3600 })).toBeUndefined();
    // A declared client is always kept, so a stored client_id can never be overwritten with empty.
    expect(withoutMintedTokenCredentials({ client_id: "x", access_token: "t" })).toEqual({ client_id: "x" });
  });
});

describe("credentialAuthClass", () => {
  it("collapses the client-forwarded modes to one class and leaves others distinct", () => {
    expect(credentialAuthClass(AUTH_TYPE.TRUE_PASSTHROUGH)).toBe("client_forwarded");
    expect(credentialAuthClass(AUTH_TYPE.OAUTH_DELEGATE)).toBe("client_forwarded");
    expect(credentialAuthClass(AUTH_TYPE.OAUTH2)).toBe(AUTH_TYPE.OAUTH2);
    expect(credentialAuthClass(null)).toBeNull();
  });
});

describe("id_jag auth type", () => {
  it("classifies oauth2_id_jag as the id_jag oauth mode", async () => {
    const { getMcpOAuthMode, AUTH_TYPE } = await import("./types");
    expect(getMcpOAuthMode({ auth_type: AUTH_TYPE.OAUTH2_ID_JAG })).toBe("id_jag");
  });

  it("auto-connects only oauth2_id_jag servers in the Apps grid", async () => {
    const { isAutoConnectedAuthType, AUTH_TYPE } = await import("./types");
    expect(isAutoConnectedAuthType(AUTH_TYPE.OAUTH2_ID_JAG)).toBe(true);
    expect(isAutoConnectedAuthType(AUTH_TYPE.OAUTH2)).toBe(false);
    expect(isAutoConnectedAuthType(AUTH_TYPE.OAUTH2_TOKEN_EXCHANGE)).toBe(false);
    expect(isAutoConnectedAuthType(null)).toBe(false);
    expect(isAutoConnectedAuthType(undefined)).toBe(false);
  });
});
