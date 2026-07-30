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
    expect(onTokenReceived).toHaveBeenCalledWith(token, expect.objectContaining({ clientId: "client-1" }));

    act(() => {
      result.current.reset();
    });

    expect(result.current.status).toBe("idle");
    expect(result.current.tokenResponse).toBeNull();
    expect(result.current.error).toBeNull();
  });

  it("ignores an in-flight exchange result after reset", async () => {
    const token = { access_token: "stale-token" };
    let resolveExchange: (value: typeof token) => void = () => undefined;
    const exchangePromise = new Promise<typeof token>((resolve) => {
      resolveExchange = resolve;
    });
    vi.mocked(networking.exchangeMcpOAuthToken).mockReturnValueOnce(exchangePromise);
    seedCompletedRedirect();

    const onTokenReceived = vi.fn();
    const { result } = renderFlow(onTokenReceived);

    await waitFor(() => expect(result.current.status).toBe("exchanging"));

    act(() => {
      result.current.reset();
    });

    await act(async () => {
      resolveExchange(token);
      await exchangePromise;
    });

    expect(onTokenReceived).not.toHaveBeenCalled();
    expect(result.current.status).toBe("idle");
    expect(result.current.tokenResponse).toBeNull();
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

    await waitFor(() =>
      expect(onTokenReceived2).toHaveBeenCalledWith(token, expect.objectContaining({ clientId: "client-1" })),
    );
  });

  it("passes the DCR-registered client_id and client_secret to onTokenReceived so the created server persists them", async () => {
    const token = { access_token: "tok-xyz", refresh_token: "ref-xyz", expires_in: 3600 };
    vi.mocked(networking.exchangeMcpOAuthToken).mockResolvedValue(token);
    setSecureItem(RESULT_KEY, JSON.stringify({ state: "state-1", code: "code-1" }));
    setSecureItem(
      FLOW_STATE_KEY,
      JSON.stringify({
        state: "state-1",
        codeVerifier: "verifier-1",
        serverId: "server-1",
        clientId: "dcr-client-xyz",
        clientSecret: "dcr-secret-abc",
        redirectUri: "https://app.example.com/ui/mcp/oauth/callback",
        flowSource: "create",
      }),
    );

    const onTokenReceived = vi.fn();
    const { result } = renderFlow(onTokenReceived);

    await waitFor(() => expect(result.current.status).toBe("success"));
    expect(onTokenReceived).toHaveBeenCalledWith(token, {
      clientId: "dcr-client-xyz",
      clientSecret: "dcr-secret-abc",
    });
  });

  it("reuses an existing client_id and does not register a new client (second Authorize & Fetch, same server)", async () => {
    vi.mocked(networking.cacheTemporaryMcpServer).mockResolvedValue({ server_id: "server-1" });
    vi.mocked(networking.buildMcpOAuthAuthorizeUrl).mockReturnValue("https://idp.example.com/authorize");

    const { result } = renderHook(() =>
      useMcpOAuthFlow({
        accessToken: "admin-token",
        getCredentials: () => ({ client_id: "existing-client" }),
        getTemporaryPayload: () => ({
          url: "https://server-1.example.com/mcp",
          transport: "http",
          credentials: { client_id: "existing-client" },
        }),
        onTokenReceived: vi.fn(),
        flowSource: "create",
      }),
    );

    await act(async () => {
      await result.current.startOAuthFlow();
    });

    expect(networking.registerMcpOAuthClient).not.toHaveBeenCalled();
    expect(networking.buildMcpOAuthAuthorizeUrl).toHaveBeenCalledWith(
      expect.objectContaining({ clientId: "existing-client" }),
    );
  });

  it("registers a fresh client when no client_id is present (new URL after the derived client is cleared)", async () => {
    vi.mocked(networking.cacheTemporaryMcpServer).mockResolvedValue({ server_id: "server-2" });
    vi.mocked(networking.registerMcpOAuthClient).mockResolvedValue({ client_id: "fresh-client" });
    vi.mocked(networking.buildMcpOAuthAuthorizeUrl).mockReturnValue("https://idp.example.com/authorize");

    const { result } = renderHook(() =>
      useMcpOAuthFlow({
        accessToken: "admin-token",
        getCredentials: () => ({}),
        getTemporaryPayload: () => ({
          url: "https://server-2.example.com/mcp",
          transport: "http",
          credentials: {},
        }),
        onTokenReceived: vi.fn(),
        flowSource: "create",
      }),
    );

    await act(async () => {
      await result.current.startOAuthFlow();
    });

    expect(networking.registerMcpOAuthClient).toHaveBeenCalledTimes(1);
    expect(networking.buildMcpOAuthAuthorizeUrl).toHaveBeenCalledWith(
      expect.objectContaining({ clientId: "fresh-client" }),
    );
  });
});
