import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { describe, expect, it, vi, beforeEach } from "vitest";
import MCPToolsViewer from "./mcp_tools";
import {
  listMCPTools,
  listMCPPrompts,
  listMCPResources,
  getMCPOAuthUserCredentialStatus,
} from "@/components/networking";
import { isTokenValid, getToken, removeToken } from "@/utils/mcpTokenStore";

vi.mock("@/components/networking", () => ({
  listMCPTools: vi.fn(),
  listMCPPrompts: vi.fn(),
  listMCPResources: vi.fn(),
  callMCPTool: vi.fn(),
  getMCPOAuthUserCredentialStatus: vi.fn(),
}));

vi.mock("@/utils/mcpTokenStore", () => ({
  isTokenValid: vi.fn(),
  getToken: vi.fn(),
  removeToken: vi.fn(),
}));

const { toolsOAuthFlowSpy, userMcpOAuthFlowSpy } = vi.hoisted(() => {
  type FlowState = { startOAuthFlow: () => void; status: string; error: string | null };
  const idle = (): FlowState => ({ startOAuthFlow: () => {}, status: "idle", error: null });
  return {
    toolsOAuthFlowSpy: vi.fn((_options: { onSuccess: (token: string) => void }) => idle()),
    userMcpOAuthFlowSpy: vi.fn((_options: { onSuccess: () => void }) => idle()),
  };
});

vi.mock("@/hooks/useToolsOAuthFlow", () => ({
  useToolsOAuthFlow: toolsOAuthFlowSpy,
}));

vi.mock("@/hooks/useUserMcpOAuthFlow", () => ({
  useUserMcpOAuthFlow: userMcpOAuthFlowSpy,
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

describe("MCPToolsViewer gatewayMintsClient wiring", () => {
  // Pins the call site (not just the helper): the viewer must pass the bridge-AWARE
  // gatewayMintsClientFor value to useToolsOAuthFlow, so the browser skips its own register exactly
  // when the gateway mints. The oauth_delegate + dcr_bridge cell is the regression guard: with the
  // old bridge-blind predicate it would have passed true here and dead-ended.
  beforeEach(() => toolsOAuthFlowSpy.mockClear());

  it.each([
    { auth_type: "true_passthrough", dcr_bridge: true, gatewayMintsClient: true },
    { auth_type: "true_passthrough", dcr_bridge: false, gatewayMintsClient: true },
    { auth_type: "oauth_delegate", dcr_bridge: false, gatewayMintsClient: true },
    { auth_type: "oauth_delegate", dcr_bridge: true, gatewayMintsClient: false },
  ])(
    "passes gatewayMintsClient=$gatewayMintsClient for $auth_type dcr_bridge=$dcr_bridge",
    ({ auth_type, dcr_bridge, gatewayMintsClient }) => {
      renderViewer({ auth_type, dcr_bridge, tokenUrl: null });
      expect(toolsOAuthFlowSpy).toHaveBeenCalledWith(expect.objectContaining({ gatewayMintsClient }));
    },
  );
});

describe("MCPToolsViewer auth gate routing", () => {
  beforeEach(() => {
    vi.mocked(listMCPTools).mockReset().mockResolvedValue({ tools: [], error: null });
    vi.mocked(listMCPPrompts).mockReset().mockResolvedValue({ prompts: [] });
    vi.mocked(listMCPResources).mockReset().mockResolvedValue({ resources: [], resource_templates: [] });
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

describe("MCPToolsViewer prompts and resources catalog", () => {
  beforeEach(() => {
    vi.mocked(listMCPTools).mockReset().mockResolvedValue({ tools: [], error: null });
    vi.mocked(listMCPPrompts)
      .mockReset()
      .mockResolvedValue({
        prompts: [
          {
            name: "summarize",
            description: "Summarize a block of text",
            arguments: [
              { name: "text", required: true },
              { name: "style", required: false },
            ],
          },
        ],
      });
    vi.mocked(listMCPResources)
      .mockReset()
      .mockResolvedValue({
        resources: [{ name: "readme", uri: "demo://readme", description: "Project readme", mimeType: "text/markdown" }],
        resource_templates: [{ name: "profile", uriTemplate: "demo://users/{user_id}/profile" }],
      });
    vi.mocked(isTokenValid).mockReset().mockReturnValue(false);
    vi.mocked(getToken).mockReset().mockReturnValue(null);
    vi.mocked(getMCPOAuthUserCredentialStatus).mockReset().mockResolvedValue(credStatus());
  });

  it("lists the server's prompts and resources next to its tools", async () => {
    renderViewer({ auth_type: "api_key", tokenUrl: null });

    const prompts = await screen.findByRole("region", { name: "Prompts" });
    expect(await within(prompts).findByText("summarize")).toBeInTheDocument();
    expect(within(prompts).getByText("Summarize a block of text")).toBeInTheDocument();
    expect(within(prompts).getByText("text")).toBeInTheDocument();
    expect(within(prompts).getByText("style")).toBeInTheDocument();

    const resources = screen.getByRole("region", { name: "Resources" });
    expect(await within(resources).findByText("demo://readme")).toBeInTheDocument();
    expect(within(resources).getByText("text/markdown")).toBeInTheDocument();
    expect(within(resources).getByText("demo://users/{user_id}/profile")).toBeInTheDocument();
    expect(within(resources).getByText("2")).toBeInTheDocument();

    expect(vi.mocked(listMCPPrompts)).toHaveBeenCalledWith("litellm-key", "srv-1", undefined);
    expect(vi.mocked(listMCPResources)).toHaveBeenCalledWith("litellm-key", "srv-1", undefined);
  });

  it("forwards the browser session token to the prompt and resource listings like tools", async () => {
    vi.mocked(isTokenValid).mockReturnValue(true);
    vi.mocked(getToken).mockReturnValue({
      access_token: "slack-tok",
      expires_at: Date.now() + 60_000,
      token_type: "bearer",
    });

    renderViewer({ oauth2_flow: null, delegate_auth_to_upstream: true });

    await screen.findByText("summarize");
    const passthroughHeader = expect.objectContaining({ "x-mcp-slack-authorization": "Bearer slack-tok" });
    expect(vi.mocked(listMCPPrompts)).toHaveBeenCalledWith("litellm-key", "srv-1", passthroughHeader);
    expect(vi.mocked(listMCPResources)).toHaveBeenCalledWith("litellm-key", "srv-1", passthroughHeader);
  });

  it("does not list prompts or resources while the auth gate is shown", async () => {
    vi.mocked(getMCPOAuthUserCredentialStatus).mockResolvedValue(credStatus({ has_credential: false }));

    renderViewer({ oauth2_flow: null, delegate_auth_to_upstream: false });

    expect(await screen.findByText(GATE_TEXT)).toBeInTheDocument();
    expect(screen.queryByRole("region", { name: "Prompts" })).not.toBeInTheDocument();
    expect(vi.mocked(listMCPPrompts)).not.toHaveBeenCalled();
    expect(vi.mocked(listMCPResources)).not.toHaveBeenCalled();
  });

  it("shows the upstream error for a catalog that failed to load", async () => {
    const failedResources = {
      resources: [],
      resource_templates: [],
      error: "http_502",
      message: "upstream unreachable",
      status: 502,
    };
    vi.mocked(listMCPResources).mockResolvedValue(failedResources);

    renderViewer({ auth_type: "api_key", tokenUrl: null });

    const resources = await screen.findByRole("region", { name: "Resources" });
    expect(await within(resources).findByText("Error: upstream unreachable")).toBeInTheDocument();
    expect(await screen.findByText("summarize")).toBeInTheDocument();
  });

  const unauthorized = { error: "auth_required", message: "upstream credential expired", status: 401 };

  it("gates on a prompts 401 even when tools load, and reloads all three listings after Authorize (stored credential)", async () => {
    vi.mocked(listMCPTools).mockResolvedValue({ tools: [], error: null });
    vi.mocked(listMCPPrompts).mockResolvedValue({ prompts: [], ...unauthorized });
    // The hook completes the redirect flow out of band; here Authorize resolves it immediately.
    userMcpOAuthFlowSpy.mockReset().mockImplementation((options) => ({
      startOAuthFlow: () => options.onSuccess(),
      status: "idle",
      error: null,
    }));

    renderViewer({ oauth2_flow: null, delegate_auth_to_upstream: false });

    expect(await screen.findByText(GATE_TEXT)).toBeInTheDocument();
    expect(screen.queryByRole("region", { name: "Prompts" })).not.toBeInTheDocument();
    expect(vi.mocked(listMCPTools)).toHaveBeenCalledTimes(1);
    expect(vi.mocked(listMCPPrompts)).toHaveBeenCalledTimes(1);
    expect(vi.mocked(listMCPResources)).toHaveBeenCalledTimes(1);

    vi.mocked(listMCPPrompts).mockResolvedValue({ prompts: [{ name: "summarize" }] });
    await userEvent.click(screen.getByRole("button", { name: "Authorize" }));

    const prompts = await screen.findByRole("region", { name: "Prompts" });
    expect(await within(prompts).findByText("summarize")).toBeInTheDocument();
    expect(screen.queryByText(GATE_TEXT)).not.toBeInTheDocument();
    expect(vi.mocked(listMCPTools)).toHaveBeenCalledTimes(2);
    expect(vi.mocked(listMCPPrompts)).toHaveBeenCalledTimes(2);
    expect(vi.mocked(listMCPResources)).toHaveBeenCalledTimes(2);
  });

  it("gates on a resources 401 even when tools load, and relists all three with the new browser token after Authorize", async () => {
    vi.mocked(isTokenValid).mockReturnValue(true);
    vi.mocked(getToken).mockReturnValue({
      access_token: "stale-tok",
      expires_at: Date.now() + 60_000,
      token_type: "bearer",
    });
    vi.mocked(listMCPResources).mockResolvedValue({ resources: [], resource_templates: [], ...unauthorized });
    toolsOAuthFlowSpy.mockReset().mockImplementation((options) => ({
      startOAuthFlow: () => options.onSuccess("fresh-tok"),
      status: "idle",
      error: null,
    }));

    renderViewer({ oauth2_flow: null, delegate_auth_to_upstream: true });

    expect(await screen.findByText(GATE_TEXT)).toBeInTheDocument();
    expect(vi.mocked(removeToken)).toHaveBeenCalledWith("srv-1", "tin@berri.ai");
    expect(screen.queryByRole("region", { name: "Resources" })).not.toBeInTheDocument();

    vi.mocked(listMCPResources).mockResolvedValue({
      resources: [{ name: "readme", uri: "demo://readme" }],
      resource_templates: [],
    });
    await userEvent.click(screen.getByRole("button", { name: "Authorize" }));

    const resources = await screen.findByRole("region", { name: "Resources" });
    expect(await within(resources).findByText("demo://readme")).toBeInTheDocument();
    const freshHeader = expect.objectContaining({ "x-mcp-slack-authorization": "Bearer fresh-tok" });
    expect(vi.mocked(listMCPTools)).toHaveBeenLastCalledWith("litellm-key", "srv-1", freshHeader);
    expect(vi.mocked(listMCPPrompts)).toHaveBeenLastCalledWith("litellm-key", "srv-1", freshHeader);
    expect(vi.mocked(listMCPResources)).toHaveBeenLastCalledWith("litellm-key", "srv-1", freshHeader);
  });
});
