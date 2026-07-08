import { act, renderHook, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import * as networking from "@/components/networking";
import * as mcpTokenStore from "@/utils/mcpTokenStore";
import { getSecureItem, setSecureItem } from "@/utils/secureStorage";
import { useToolsOAuthFlow } from "./useToolsOAuthFlow";
import { useUserMcpOAuthFlow } from "./useUserMcpOAuthFlow";

vi.mock("@/components/networking", () => ({
  exchangeMcpOAuthToken: vi.fn(),
  registerMcpOAuthClient: vi.fn(),
  buildMcpOAuthAuthorizeUrl: vi.fn(() => "https://idp.example.com/authorize"),
  storeMCPOAuthUserCredential: vi.fn(),
  getProxyBaseUrl: vi.fn(() => ""),
  serverRootPath: "",
}));

vi.mock("@/components/molecules/notifications_manager", () => ({
  default: { success: vi.fn(), error: vi.fn() },
}));

const TOOLS_FLOW_STATE_KEY = "litellm-tools-mcp-oauth-flow-state";
const TOOLS_RESULT_KEY = "litellm-tools-mcp-oauth-result";
const USER_FLOW_STATE_KEY = "litellm-user-mcp-oauth-flow-state";
const USER_RESULT_KEY = "litellm-user-mcp-oauth-result";

/** Seed the storage for a completed IdP redirect, keyed for a specific flow. */
function seedCompletedRedirect(flowStateKey: string, resultKey: string, extra: Record<string, unknown> = {}) {
  setSecureItem(resultKey, JSON.stringify({ state: "state-1", code: "code-1" }));
  setSecureItem(
    flowStateKey,
    JSON.stringify({
      state: "state-1",
      codeVerifier: "verifier-1",
      serverId: "server-1",
      clientId: "client-1",
      redirectUri: "https://app.example.com/ui/mcp/oauth/callback",
      ...extra,
    }),
  );
}

describe("MCP OAuth PKCE flow wrappers", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    window.sessionStorage.clear();
    window.localStorage.clear();
  });

  describe("useToolsOAuthFlow", () => {
    it("persists the exchanged token to sessionStorage (with userId) and never to the backend", async () => {
      const token = { access_token: "tok-tools", expires_in: 3600, refresh_token: "ref-tools", token_type: "bearer" };
      vi.mocked(networking.exchangeMcpOAuthToken).mockResolvedValue(token);
      const setTokenSpy = vi.spyOn(mcpTokenStore, "setToken");
      seedCompletedRedirect(TOOLS_FLOW_STATE_KEY, TOOLS_RESULT_KEY);

      const onSuccess = vi.fn();
      const { result } = renderHook(() =>
        useToolsOAuthFlow({ accessToken: "user-token", serverId: "server-1", userId: "user-42", onSuccess }),
      );

      await waitFor(() => expect(result.current.status).toBe("success"));

      expect(setTokenSpy).toHaveBeenCalledWith(
        "server-1",
        {
          access_token: "tok-tools",
          expires_in: 3600,
          refresh_token: "ref-tools",
          token_type: "bearer",
        },
        "user-42",
      );
      expect(networking.storeMCPOAuthUserCredential).not.toHaveBeenCalled();
      expect(onSuccess).toHaveBeenCalledWith("tok-tools");
    });

    it("does not consume a result written under the user flow's key", async () => {
      vi.mocked(networking.exchangeMcpOAuthToken).mockResolvedValue({ access_token: "x" });
      seedCompletedRedirect(USER_FLOW_STATE_KEY, USER_RESULT_KEY);

      const onSuccess = vi.fn();
      const { result } = renderHook(() =>
        useToolsOAuthFlow({ accessToken: "user-token", serverId: "server-1", onSuccess }),
      );

      // Give the on-mount resume a chance to (not) run.
      await Promise.resolve();
      expect(networking.exchangeMcpOAuthToken).not.toHaveBeenCalled();
      expect(result.current.status).toBe("idle");
    });

    it("stores the raw current URL as the post-redirect return target", async () => {
      vi.mocked(networking.registerMcpOAuthClient).mockResolvedValue({ client_id: "c" });
      const { result } = renderHook(() =>
        useToolsOAuthFlow({ accessToken: "user-token", serverId: "server-1", onSuccess: vi.fn() }),
      );

      await act(async () => {
        await result.current.startOAuthFlow();
      });

      expect(getSecureItem("litellm-mcp-oauth-return-url")).toBe(window.location.href);
    });
  });

  describe("useUserMcpOAuthFlow", () => {
    it("persists the exchanged token to the backend (with flow scopes) and never to sessionStorage", async () => {
      const token = { access_token: "tok-user", refresh_token: "ref-user", expires_in: 1800 };
      vi.mocked(networking.exchangeMcpOAuthToken).mockResolvedValue(token);
      const setTokenSpy = vi.spyOn(mcpTokenStore, "setToken");
      seedCompletedRedirect(USER_FLOW_STATE_KEY, USER_RESULT_KEY, { scopes: ["repo", "read:user"] });

      const onSuccess = vi.fn();
      const { result } = renderHook(() =>
        useUserMcpOAuthFlow({ accessToken: "user-token", serverId: "server-1", onSuccess }),
      );

      await waitFor(() => expect(result.current.status).toBe("success"));

      expect(networking.storeMCPOAuthUserCredential).toHaveBeenCalledWith("user-token", "server-1", {
        access_token: "tok-user",
        refresh_token: "ref-user",
        expires_in: 1800,
        scopes: ["repo", "read:user"],
      });
      expect(setTokenSpy).not.toHaveBeenCalled();
      expect(onSuccess).toHaveBeenCalledTimes(1);
    });

    it("does not consume a result written under the tools flow's key", async () => {
      vi.mocked(networking.exchangeMcpOAuthToken).mockResolvedValue({ access_token: "x" });
      seedCompletedRedirect(TOOLS_FLOW_STATE_KEY, TOOLS_RESULT_KEY);

      const { result } = renderHook(() =>
        useUserMcpOAuthFlow({ accessToken: "user-token", serverId: "server-1", onSuccess: vi.fn() }),
      );

      await Promise.resolve();
      expect(networking.exchangeMcpOAuthToken).not.toHaveBeenCalled();
      expect(networking.storeMCPOAuthUserCredential).not.toHaveBeenCalled();
      expect(result.current.status).toBe("idle");
    });

    it("tags the return URL with mcpOauthReturn=apps so the redirect lands back on the apps view", async () => {
      vi.mocked(networking.registerMcpOAuthClient).mockResolvedValue({ client_id: "c" });
      const { result } = renderHook(() =>
        useUserMcpOAuthFlow({ accessToken: "user-token", serverId: "server-1", onSuccess: vi.fn() }),
      );

      await act(async () => {
        await result.current.startOAuthFlow();
      });

      const returnUrl = getSecureItem("litellm-mcp-oauth-return-url");
      expect(returnUrl).not.toBeNull();
      expect(new URL(returnUrl as string).searchParams.get("mcpOauthReturn")).toBe("apps");
    });
  });
});
