import { renderHook, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import * as networking from "@/components/networking";
import { setSecureItem } from "@/utils/secureStorage";
import { useUserMcpOAuthFlow } from "./useUserMcpOAuthFlow";

vi.mock("@/components/networking", () => ({
  exchangeMcpOAuthToken: vi.fn(),
  storeMCPOAuthUserCredential: vi.fn(),
  registerMcpOAuthClient: vi.fn(),
  buildMcpOAuthAuthorizeUrl: vi.fn(),
  getProxyBaseUrl: () => "",
  serverRootPath: "",
}));
vi.mock("@/lib/toast", () => ({ toast: { success: vi.fn(), error: vi.fn() } }));

describe("user OAuth granted scopes", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    window.sessionStorage.clear();
    setSecureItem(
      "litellm-user-mcp-oauth-flow-state",
      JSON.stringify({
        state: "callback-state",
        codeVerifier: "verifier",
        serverId: "server-1",
        redirectUri: "https://app.example/callback",
        scopes: ["tools.write"],
      }),
    );
    setSecureItem(
      "litellm-user-mcp-oauth-result",
      JSON.stringify({
        state: "callback-state",
        code: "provider-code",
      }),
    );
  });

  it.each([
    ["tools.read tools.write", ["tools.read", "tools.write"]],
    ["tools.read", ["tools.read"]],
    ["", []],
    [undefined, ["tools.write"]],
  ])("persists the actual grant %s after reconnect", async (scope, expected) => {
    vi.mocked(networking.exchangeMcpOAuthToken).mockResolvedValue({
      access_token: "vendor-token",
      refresh_token: "vendor-refresh",
      scope,
    });
    const onSuccess = vi.fn();
    const { result } = renderHook(() =>
      useUserMcpOAuthFlow({
        accessToken: "caller",
        serverId: "server-1",
        onSuccess,
      }),
    );
    await waitFor(() => expect(result.current.status).toBe("success"));
    expect(networking.storeMCPOAuthUserCredential).toHaveBeenCalledExactlyOnceWith("caller", "server-1", {
      access_token: "vendor-token",
      refresh_token: "vendor-refresh",
      expires_in: undefined,
      scopes: expected,
    });
    expect(onSuccess).toHaveBeenCalledOnce();
  });
});
