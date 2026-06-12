import { describe, it, expect } from "vitest";
import {
  AUTH_TYPE,
  OAUTH_FLOW,
  MCP_OAUTH2_FLOW_M2M,
  TRANSPORT,
  handleTransport,
  handleAuth,
  getMcpOAuthMode,
} from "./types";

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

  it("classifies an interactive server without delegation as obo", () => {
    expect(getMcpOAuthMode({ auth_type: AUTH_TYPE.OAUTH2, oauth2_flow: null, delegate_auth_to_upstream: false })).toBe(
      "obo",
    );
  });

  it("defaults to obo when delegate_auth_to_upstream is undefined", () => {
    expect(getMcpOAuthMode({ auth_type: AUTH_TYPE.OAUTH2 })).toBe("obo");
  });

  it("treats explicit authorization_code as interactive, not m2m", () => {
    expect(
      getMcpOAuthMode({
        auth_type: AUTH_TYPE.OAUTH2,
        oauth2_flow: "authorization_code",
        delegate_auth_to_upstream: false,
      }),
    ).toBe("obo");
  });

  // Regression: the old heuristic labeled any OAuth2 server with a token endpoint
  // as M2M. getMcpOAuthMode ignores token_url, so an interactive server that
  // legitimately carries one is classified by oauth2_flow + delegate, never M2M.
  it("does not treat an interactive server with a token endpoint as m2m", () => {
    expect(getMcpOAuthMode({ auth_type: AUTH_TYPE.OAUTH2, oauth2_flow: null, delegate_auth_to_upstream: false })).toBe(
      "obo",
    );
  });
});
