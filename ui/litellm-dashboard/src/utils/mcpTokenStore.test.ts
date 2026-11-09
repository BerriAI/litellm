// @vitest-environment jsdom

import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { clearAllMcpTokens, getToken, isTokenValid, removeToken, setToken } from "./mcpTokenStore";

const decodeMaybeBase64 = (raw: string): string => {
  try {
    return atob(raw);
  } catch {
    return raw;
  }
};

const allStoredValues = (): string =>
  Array.from({ length: sessionStorage.length }, (_, i) => sessionStorage.key(i) ?? "")
    .map((key) => sessionStorage.getItem(key) ?? "")
    .flatMap((raw) => [raw, decodeMaybeBase64(raw)])
    .join("\n");

describe("mcpTokenStore", () => {
  beforeEach(() => {
    sessionStorage.clear();
  });

  afterEach(() => {
    sessionStorage.clear();
  });

  it("never persists a refresh token, even when a caller supplies one", () => {
    const callerPayload = {
      access_token: "access-value",
      expires_in: 3600,
      refresh_token: "refresh-value-must-not-persist",
      token_type: "bearer",
    };

    setToken("server-a", callerPayload, "user-1");

    expect(getToken("server-a", "user-1")?.access_token).toBe("access-value");
    expect(allStoredValues()).not.toContain("refresh-value-must-not-persist");
  });

  it("does not write the token payload as readable text", () => {
    setToken("server-a", { access_token: "plain-access-value" }, "user-1");

    const raw = sessionStorage.getItem("mcp-session-token:user-1:server-a");
    expect(raw).not.toBeNull();
    expect(raw).not.toContain("plain-access-value");
    expect(getToken("server-a", "user-1")?.access_token).toBe("plain-access-value");
  });

  it("scopes tokens by user id", () => {
    setToken("server-a", { access_token: "user1-token" }, "user-1");
    setToken("server-a", { access_token: "user2-token" }, "user-2");

    expect(getToken("server-a", "user-1")?.access_token).toBe("user1-token");
    expect(getToken("server-a", "user-2")?.access_token).toBe("user2-token");
    expect(getToken("server-a", "user-3")).toBeNull();
  });

  it("validates expiry per user scope", () => {
    setToken("server-a", { access_token: "tok", expires_in: 3600 }, "user-1");
    expect(isTokenValid("server-a", "user-1")).toBe(true);
    removeToken("server-a", "user-1");
    expect(isTokenValid("server-a", "user-1")).toBe(false);
  });

  it("clearAllMcpTokens removes every mcp-session-token entry", () => {
    setToken("s1", { access_token: "a" }, "u1");
    setToken("s2", { access_token: "b" }, "u2");
    sessionStorage.setItem("unrelated", "keep");

    clearAllMcpTokens();

    expect(getToken("s1", "u1")).toBeNull();
    expect(getToken("s2", "u2")).toBeNull();
    expect(sessionStorage.getItem("unrelated")).toBe("keep");
  });
});
