import { act, renderHook } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import * as networking from "@/components/networking";
import { useToolsOAuthFlow } from "./useToolsOAuthFlow";

vi.mock("@/components/networking", () => ({
  exchangeMcpOAuthToken: vi.fn(),
  registerMcpOAuthClient: vi.fn(),
  buildMcpOAuthAuthorizeUrl: vi.fn(() => "https://gw.example.com/v1/mcp/server/oauth/server-1/authorize"),
  getProxyBaseUrl: vi.fn(() => ""),
  serverRootPath: "",
}));

vi.mock("@/components/molecules/notifications_manager", () => ({
  default: { success: vi.fn(), error: vi.fn() },
}));

vi.mock("@/utils/pkce", () => ({
  generateCodeVerifier: () => "verifier-1",
  generateCodeChallenge: async () => "challenge-1",
}));

function renderFlow(options: { gatewayMintsClient?: boolean }) {
  return renderHook(() =>
    useToolsOAuthFlow({
      accessToken: "user-token",
      serverId: "server-1",
      serverAlias: "server one",
      userId: "user-1",
      gatewayMintsClient: options.gatewayMintsClient,
      onSuccess: vi.fn(),
    }),
  );
}

describe("useToolsOAuthFlow client registration", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    window.sessionStorage.clear();
    window.localStorage.clear();
    Object.defineProperty(window, "location", {
      value: { href: "https://app.example.com/ui?page=mcp-servers" },
      writable: true,
      configurable: true,
    });
  });

  it("skips browser-side client registration when the gateway mints the client", async () => {
    // Client-forwarded token modes: the gateway's /authorize mints and carries the
    // OAuth client itself, so a browser-side registration would only create an
    // extra orphan client at the IdP. The authorize URL must go out clientless.
    const { result } = renderFlow({ gatewayMintsClient: true });

    await act(async () => {
      await result.current.startOAuthFlow();
    });

    expect(networking.registerMcpOAuthClient).not.toHaveBeenCalled();
    expect(networking.buildMcpOAuthAuthorizeUrl).toHaveBeenCalledWith(
      expect.objectContaining({ serverId: "server-1", clientId: undefined }),
    );
    expect(window.location.href).toBe("https://gw.example.com/v1/mcp/server/oauth/server-1/authorize");
  });

  it("still registers browser-side for servers the gateway does not mint for", async () => {
    // The cells where gatewayMintsClientFor is false keep the browser-held client: an
    // oauth_delegate dcr_bridge server (interactive sign-in) and the legacy oauth2 passthrough both
    // register first, so the minted client_id rides the authorize URL through the front-door relay.
    vi.mocked(networking.registerMcpOAuthClient).mockResolvedValue({ client_id: "reg-client-1" });

    const { result } = renderFlow({});

    await act(async () => {
      await result.current.startOAuthFlow();
    });

    expect(networking.registerMcpOAuthClient).toHaveBeenCalledTimes(1);
    // The dcr_bridge relay rejects registrations without the client's own
    // callback, so the browser must bind its minted client to it.
    expect(networking.registerMcpOAuthClient).toHaveBeenCalledWith(
      "user-token",
      "server-1",
      expect.objectContaining({ redirect_uris: [expect.stringContaining("callback")] }),
    );
    expect(networking.buildMcpOAuthAuthorizeUrl).toHaveBeenCalledWith(
      expect.objectContaining({ serverId: "server-1", clientId: "reg-client-1" }),
    );
  });
});
