import { act, renderHook, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import * as networking from "@/components/networking";
import { setSecureItem } from "@/utils/secureStorage";
import { useMcpOAuthFlow } from "./useMcpOAuthFlow";

vi.mock("@/components/networking", () => ({
  exchangeMcpOAuthToken: vi.fn(),
  cacheTemporaryMcpServer: vi.fn(),
  registerMcpOAuthClient: vi.fn(),
  buildMcpOAuthAuthorizeUrl: vi.fn(),
  getProxyBaseUrl: vi.fn(() => ""),
  serverRootPath: "",
}));

vi.mock("@/components/molecules/notifications_manager", () => ({
  default: { success: vi.fn(), error: vi.fn() },
}));

const FLOW_STATE_KEY = "litellm-mcp-oauth-flow-state";
const RESULT_KEY = "litellm-mcp-oauth-result";

/** Seed storage so the hook's on-mount resume flow exchanges a code for a token. */
function seedCompletedRedirect() {
  setSecureItem(RESULT_KEY, JSON.stringify({ state: "state-1", code: "code-1" }));
  setSecureItem(
    FLOW_STATE_KEY,
    JSON.stringify({
      state: "state-1",
      codeVerifier: "verifier-1",
      serverId: "server-1",
      clientId: "client-1",
      redirectUri: "https://app.example.com/ui/mcp/oauth/callback",
      flowSource: "create",
    }),
  );
}

function renderFlow(onTokenReceived = vi.fn()) {
  return renderHook(() =>
    useMcpOAuthFlow({
      accessToken: "admin-token",
      getCredentials: () => ({}),
      getTemporaryPayload: () => ({ url: "https://server-1.example.com/mcp", transport: "http" }),
      onTokenReceived,
      flowSource: "create",
    }),
  );
}

describe("useMcpOAuthFlow reset", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    window.sessionStorage.clear();
    window.localStorage.clear();
  });

  it("clears a successfully fetched token so it cannot leak into the next session", async () => {
    const token = { access_token: "tok-123", expires_in: 3600 };
    vi.mocked(networking.exchangeMcpOAuthToken).mockResolvedValue(token);
    seedCompletedRedirect();

    const onTokenReceived = vi.fn();
    const { result } = renderFlow(onTokenReceived);

    await waitFor(() => expect(result.current.status).toBe("success"));
    expect(result.current.tokenResponse).toEqual(token);
    expect(onTokenReceived).toHaveBeenCalledWith(token);

    act(() => {
      result.current.reset();
    });

    expect(result.current.status).toBe("idle");
    expect(result.current.tokenResponse).toBeNull();
    expect(result.current.error).toBeNull();
  });
});
