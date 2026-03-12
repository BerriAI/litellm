import { describe, it, expect } from "vitest";
import { AUTH_TYPE, OAUTH_FLOW, TRANSPORT, handleTransport, handleAuth } from "./types";

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
});
