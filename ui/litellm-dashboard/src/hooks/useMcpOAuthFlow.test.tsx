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

/** Seed the redirect result (the code returned by the IdP callback). */
function seedResult(code: string) {
  setSecureItem(RESULT_KEY, JSON.stringify({ state: "state-1", code }));
}

/** Seed the flow state stored before the redirect. */
function seedFlowState() {
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

/** Seed storage so the hook's on-mount resume flow exchanges a code for a token. */
function seedCompletedRedirect() {
  seedResult("code-1");
  seedFlowState();
}

function renderFlow(onTokenReceived = vi.fn()) {
  return renderHook(
    ({ onTokenReceived: cb }: { onTokenReceived: (t: any) => void }) =>
      useMcpOAuthFlow({
        accessToken: "admin-token",
        getCredentials: () => ({}),
        getTemporaryPayload: () => ({ url: "https://server-1.example.com/mcp", transport: "http" }),
        onTokenReceived: cb,
        flowSource: "create",
      }),
    { initialProps: { onTokenReceived } },
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

  it("clears the in-flight guard so a callback after a mid-exchange close is not swallowed", async () => {
    // First exchange hangs, mimicking the modal being closed while the token
    // endpoint is still in flight. processingRef is left true at that point.
    vi.mocked(networking.exchangeMcpOAuthToken).mockReturnValueOnce(new Promise<any>(() => {}));
    seedFlowState();
    seedResult("code-1");

    const onTokenReceived1 = vi.fn();
    const { result, rerender } = renderFlow(onTokenReceived1);

    await waitFor(() => expect(result.current.status).toBe("exchanging"));

    act(() => {
      result.current.reset();
    });

    // The reopened modal receives a fresh callback; it must be processed, not
    // dropped by a stale in-flight guard.
    const token = { access_token: "tok-2" };
    vi.mocked(networking.exchangeMcpOAuthToken).mockResolvedValueOnce(token);
    seedResult("code-2");
    const onTokenReceived2 = vi.fn();
    rerender({ onTokenReceived: onTokenReceived2 });

    await waitFor(() => expect(onTokenReceived2).toHaveBeenCalledWith(token));
  });
});
