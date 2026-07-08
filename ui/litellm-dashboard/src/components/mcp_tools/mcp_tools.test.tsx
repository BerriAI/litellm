import { render, screen, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { describe, expect, it, vi, beforeEach } from "vitest";
import MCPToolsViewer from "./mcp_tools";
import { listMCPTools, getMCPOAuthUserCredentialStatus } from "../networking";
import { isTokenValid, getToken } from "@/utils/mcpTokenStore";

vi.mock("../networking", () => ({
  listMCPTools: vi.fn(),
  callMCPTool: vi.fn(),
  getMCPOAuthUserCredentialStatus: vi.fn(),
}));

vi.mock("@/utils/mcpTokenStore", () => ({
  isTokenValid: vi.fn(),
  getToken: vi.fn(),
  removeToken: vi.fn(),
}));

vi.mock("@/hooks/useToolsOAuthFlow", () => ({
  useToolsOAuthFlow: () => ({ startOAuthFlow: vi.fn(), status: "idle", error: null }),
}));

vi.mock("@/hooks/useUserMcpOAuthFlow", () => ({
  useUserMcpOAuthFlow: () => ({ startOAuthFlow: vi.fn(), status: "idle", error: null }),
}));

const GATE_TEXT = "Authentication required";
// Realistic interactive servers carry a token endpoint; the old heuristic
// (`oauth2 && !tokenUrl`) mislabeled exactly these as M2M. Setting it here is
// what makes the passthrough cases fail on the pre-fix code.
const TOKEN_URL = "https://slack.com/api/oauth.v2.user.access";

const renderViewer = (props: Record<string, unknown>) =>
  render(
    <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
      <MCPToolsViewer
        serverId="srv-1"
        accessToken="litellm-key"
        userRole="admin"
        userID="tin@berri.ai"
        serverAlias="slack"
        auth_type="oauth2"
        tokenUrl={TOKEN_URL}
        {...props}
      />
    </QueryClientProvider>,
  );

const credStatus = (overrides: Record<string, unknown> = {}) => ({
  server_id: "srv-1",
  has_credential: true,
  is_expired: false,
  ...overrides,
});

describe("MCPToolsViewer auth gate routing", () => {
  beforeEach(() => {
    vi.mocked(listMCPTools).mockReset().mockResolvedValue({ tools: [], error: null });
    vi.mocked(isTokenValid).mockReset().mockReturnValue(false);
    vi.mocked(getToken)
      .mockReset()
      .mockReturnValue(undefined as any);
    // Default: the OBO credential exists and is valid, so OBO servers list tools.
    vi.mocked(getMCPOAuthUserCredentialStatus).mockReset().mockResolvedValue(credStatus());
  });

  it("shows the Authorize gate for a passthrough server with a token endpoint and does not list tools", async () => {
    renderViewer({ oauth2_flow: null, delegate_auth_to_upstream: true });

    expect(await screen.findByText(GATE_TEXT)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Authorize" })).toBeInTheDocument();
    expect(vi.mocked(listMCPTools)).not.toHaveBeenCalled();
    // Passthrough must not consult the per-user DB credential.
    expect(vi.mocked(getMCPOAuthUserCredentialStatus)).not.toHaveBeenCalled();
  });

  it("forwards the session token via the x-mcp header for a passthrough server that has one", async () => {
    vi.mocked(isTokenValid).mockReturnValue(true);
    vi.mocked(getToken).mockReturnValue({ access_token: "slack-tok" } as any);

    renderViewer({ oauth2_flow: null, delegate_auth_to_upstream: true });

    await waitFor(() =>
      expect(vi.mocked(listMCPTools)).toHaveBeenCalledWith(
        "litellm-key",
        "srv-1",
        expect.objectContaining({ "x-mcp-slack-authorization": "Bearer slack-tok" }),
      ),
    );
    expect(screen.queryByText(GATE_TEXT)).not.toBeInTheDocument();
  });

  it.each([["true_passthrough"], ["oauth_delegate"]])(
    "shows the Authorize gate for a %s server without a browser token and does not list tools",
    async (authType) => {
      renderViewer({ auth_type: authType, oauth2_flow: null, delegate_auth_to_upstream: false });

      expect(await screen.findByText(GATE_TEXT)).toBeInTheDocument();
      expect(screen.getByRole("button", { name: "Authorize" })).toBeInTheDocument();
      expect(vi.mocked(listMCPTools)).not.toHaveBeenCalled();
      expect(vi.mocked(getMCPOAuthUserCredentialStatus)).not.toHaveBeenCalled();
    },
  );

  it.each([["true_passthrough"], ["oauth_delegate"]])(
    "forwards the session token via the x-mcp header for a %s server that has one",
    async (authType) => {
      vi.mocked(isTokenValid).mockReturnValue(true);
      vi.mocked(getToken).mockReturnValue({ access_token: "upstream-tok" } as ReturnType<typeof getToken>);

      renderViewer({ auth_type: authType, oauth2_flow: null, delegate_auth_to_upstream: false });

      await waitFor(() =>
        expect(vi.mocked(listMCPTools)).toHaveBeenCalledWith(
          "litellm-key",
          "srv-1",
          expect.objectContaining({ "x-mcp-slack-authorization": "Bearer upstream-tok" }),
        ),
      );
      expect(screen.queryByText(GATE_TEXT)).not.toBeInTheDocument();
    },
  );

  it("lists tools for an OBO server when the user has a DB credential, with no x-mcp header", async () => {
    renderViewer({ oauth2_flow: null, delegate_auth_to_upstream: false });

    await waitFor(() => expect(vi.mocked(listMCPTools)).toHaveBeenCalledWith("litellm-key", "srv-1", undefined));
    expect(screen.queryByText(GATE_TEXT)).not.toBeInTheDocument();
  });

  it("shows the Authorize gate for an OBO server when the user has no DB credential and does not list tools", async () => {
    vi.mocked(getMCPOAuthUserCredentialStatus).mockResolvedValue(credStatus({ has_credential: false }));

    renderViewer({ oauth2_flow: null, delegate_auth_to_upstream: false });

    expect(await screen.findByText(GATE_TEXT)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Authorize" })).toBeInTheDocument();
    expect(vi.mocked(listMCPTools)).not.toHaveBeenCalled();
  });

  it("shows the Authorize gate for an OBO server when the credential-status check fails", async () => {
    vi.mocked(getMCPOAuthUserCredentialStatus).mockRejectedValue(new Error("network down"));

    renderViewer({ oauth2_flow: null, delegate_auth_to_upstream: false });

    expect(await screen.findByText(GATE_TEXT)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Authorize" })).toBeInTheDocument();
    expect(vi.mocked(listMCPTools)).not.toHaveBeenCalled();
  });

  it("does not gate an OBO server whose stored token is expired; the list call refreshes it server-side", async () => {
    // has_credential=true with is_expired=true must NOT gate: resolve_valid_user_oauth_token
    // refreshes from the stored refresh_token on the list call, so the user never reauthorizes.
    vi.mocked(getMCPOAuthUserCredentialStatus).mockResolvedValue(
      credStatus({ has_credential: true, is_expired: true }),
    );

    renderViewer({ oauth2_flow: null, delegate_auth_to_upstream: false });

    await waitFor(() => expect(vi.mocked(listMCPTools)).toHaveBeenCalledWith("litellm-key", "srv-1", undefined));
    expect(screen.queryByText(GATE_TEXT)).not.toBeInTheDocument();
  });

  it("gates an OBO server whose stored token is expired and the list call 401s (refresh could not mint a token)", async () => {
    // has_credential=true but the list call 401s: the server-side refresh could not
    // produce a valid token (e.g. expired with no usable refresh token), so the user
    // must reauthorize instead of seeing a dead empty list.
    vi.mocked(getMCPOAuthUserCredentialStatus).mockResolvedValue(
      credStatus({ has_credential: true, is_expired: true }),
    );
    vi.mocked(listMCPTools).mockResolvedValue({
      tools: [],
      error: "unauthorized",
      status: 401,
    } as unknown as Awaited<ReturnType<typeof listMCPTools>>);

    renderViewer({ oauth2_flow: null, delegate_auth_to_upstream: false });

    expect(await screen.findByText(GATE_TEXT)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Authorize" })).toBeInTheDocument();
  });

  it("does not gate an M2M server; lists with the LiteLLM key", async () => {
    renderViewer({ oauth2_flow: "client_credentials", delegate_auth_to_upstream: false });

    await waitFor(() => expect(vi.mocked(listMCPTools)).toHaveBeenCalledWith("litellm-key", "srv-1", undefined));
    expect(screen.queryByText(GATE_TEXT)).not.toBeInTheDocument();
    // M2M uses the backend service token, not a per-user DB credential.
    expect(vi.mocked(getMCPOAuthUserCredentialStatus)).not.toHaveBeenCalled();
  });
});
