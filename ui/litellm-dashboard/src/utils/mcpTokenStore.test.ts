import { afterEach, beforeEach, describe, expect, it } from "vitest";
import {
  clearAllMcpTokens,
  getToken,
  isTokenValid,
  removeToken,
  setToken,
} from "./mcpTokenStore";

describe("mcpTokenStore", () => {
  beforeEach(() => {
    sessionStorage.clear();
  });

  afterEach(() => {
    sessionStorage.clear();
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
