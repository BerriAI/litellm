import { act, renderHook, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import * as networking from "@/components/networking";
import { getSecureItem, setSecureItem } from "@/utils/secureStorage";
import { useMcpOAuthFlow } from "./useMcpOAuthFlow";

vi.mock("@/components/networking", () => ({
  exchangeMcpOAuthToken: vi.fn(),
  cacheTemporaryMcpServer: vi.fn(),
  registerMcpOAuthClient: vi.fn(),
  buildMcpOAuthAuthorizeUrl: vi.fn(),
  getProxyBaseUrl: vi.fn(() => ""),
  serverRootPath: "",
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
    expect(onTokenReceived).toHaveBeenCalledWith(token, expect.objectContaining({ client_id: "client-1" }));

    act(() => {
      result.current.reset();
    });

    expect(result.current.status).toBe("idle");
    expect(result.current.tokenResponse).toBeNull();
    expect(result.current.error).toBeNull();
  });

  it("retains client binding from a redirect started before the state format changed", async () => {
    seedCompletedRedirect();
    const client = {
      client_id: "client-1",
      client_secret: "registered-secret",
      dcr_issuer: "https://issuer.example.com",
      dcr_server_url: "https://server-1.example.com/mcp",
      redirect_uris: ["https://app.example.com/ui/mcp/oauth/callback"],
    };
    const state = JSON.parse(getSecureItem(FLOW_STATE_KEY)!);
    setSecureItem(
      FLOW_STATE_KEY,
      JSON.stringify({ ...state, clientSecret: client.client_secret, dcrCredentials: client }),
    );
    const token = { access_token: "registered-token" };
    vi.mocked(networking.exchangeMcpOAuthToken).mockResolvedValue(token);
    const onTokenReceived = vi.fn();
    const { result } = renderFlow(onTokenReceived);
    await waitFor(() => expect(result.current.status).toBe("success"));
    expect(onTokenReceived).toHaveBeenCalledWith(token, client);
    expect(networking.exchangeMcpOAuthToken).toHaveBeenCalledWith(
      expect.objectContaining({ clientId: client.client_id, clientSecret: client.client_secret }),
    );
  });

  it("resumes a legacy server-managed flow without exposing a registered client", async () => {
    seedCompletedRedirect();
    const state = JSON.parse(getSecureItem(FLOW_STATE_KEY)!);
    delete state.clientId;
    setSecureItem(FLOW_STATE_KEY, JSON.stringify(state));
    const token = { access_token: "server-managed-token" };
    vi.mocked(networking.exchangeMcpOAuthToken).mockResolvedValue(token);
    const onTokenReceived = vi.fn();
    const { result } = renderFlow(onTokenReceived);
    await waitFor(() => expect(result.current.status).toBe("success"));
    expect(onTokenReceived).toHaveBeenCalledWith(token, undefined);
    expect(networking.exchangeMcpOAuthToken).toHaveBeenCalledWith(
      expect.objectContaining({
        serverId: "server-1",
        clientId: undefined,
        clientSecret: undefined,
      }),
    );
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
      expect(onTokenReceived2).toHaveBeenCalledWith(token, expect.objectContaining({ client_id: "client-1" })),
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
      client_id: "dcr-client-xyz",
      client_secret: "dcr-secret-abc",
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

  it.each([
    { method: "none", secret: undefined, issuer: "https://idp.example.com", bound: true },
    { method: "client_secret_basic", secret: "registered-secret", issuer: "https://idp.example.com", bound: true },
    { method: "none", secret: undefined, issuer: undefined, bound: true },
    { method: "none", secret: undefined, issuer: undefined, bound: false },
  ])(
    "retains a fresh $method client with issuer $issuer and binding $bound",
    async ({ method, secret, issuer, bound }) => {
      vi.mocked(networking.cacheTemporaryMcpServer).mockResolvedValue({ server_id: "server-2" });
      const registration = {
        client_id: "fresh-client",
        client_secret: secret,
        token_endpoint_auth_method: method,
        dcr_issuer: issuer,
        dcr_server_url: bound ? "https://server-2.example.com/mcp" : undefined,
        dcr_redirect_uris: ["https://gateway.example.com/callback"],
      };
      vi.mocked(networking.registerMcpOAuthClient).mockResolvedValue(registration);
      vi.mocked(networking.buildMcpOAuthAuthorizeUrl).mockReturnValue("https://idp.example.com/authorize");

      const options = {
        accessToken: "admin-token",
        getCredentials: () => ({}),
        getTemporaryPayload: () => ({
          url: "https://server-2.example.com/mcp",
          transport: "http",
          credentials: {},
        }),
        onTokenReceived: vi.fn(),
        flowSource: "create",
      };
      const { result } = renderHook(() => useMcpOAuthFlow(options));

      await act(async () => {
        await result.current.startOAuthFlow();
      });

      expect(JSON.parse(getSecureItem(FLOW_STATE_KEY)!)).toEqual(
        expect.objectContaining({
          client: {
            client_id: "fresh-client",
            client_secret: secret ?? null,
            ...(bound
              ? {
                  token_endpoint_auth_method: method === "client_secret_basic" ? method : null,
                  dcr_issuer: issuer ?? null,
                  dcr_server_url: "https://server-2.example.com/mcp",
                  redirect_uris: ["https://gateway.example.com/callback"],
                }
              : {}),
          },
        }),
      );
      const stored = JSON.parse(getSecureItem(FLOW_STATE_KEY)!);
      vi.mocked(networking.exchangeMcpOAuthToken).mockResolvedValue({ access_token: "new-token" });
      setSecureItem(RESULT_KEY, JSON.stringify({ state: stored.state, code: "new-code" }));
      const resumed = renderHook(() => useMcpOAuthFlow(options));
      await waitFor(() => expect(resumed.result.current.status).toBe("success"));
      expect(options.onTokenReceived).toHaveBeenCalledWith({ access_token: "new-token" }, stored.client);
      expect(networking.exchangeMcpOAuthToken).toHaveBeenCalledWith(
        expect.objectContaining({
          clientId: "fresh-client",
          clientSecret: secret,
        }),
      );
      expect(networking.registerMcpOAuthClient).toHaveBeenCalledTimes(1);
      expect(networking.buildMcpOAuthAuthorizeUrl).toHaveBeenCalledWith(
        expect.objectContaining({ clientId: "fresh-client" }),
      );
    },
  );
});
