import userEvent from "@testing-library/user-event";
import type { OnUrlUpdateFunction } from "nuqs/adapters/testing";
import { describe, expect, it, vi, beforeEach } from "vitest";
import MCPToolsViewer from "./mcp_tools";
import { callMCPTool, listMCPTools, getMCPOAuthUserCredentialStatus } from "@/components/networking";
import type { MCPTool } from "@/components/mcp_tools/types";
import { isTokenValid, getToken } from "@/utils/mcpTokenStore";
import { fireEvent, renderWithProviders, screen, testQueryClient, waitFor } from "@/../tests/test-utils";

vi.mock("@/components/networking", () => ({
  listMCPTools: vi.fn(),
  callMCPTool: vi.fn(),
  getMCPOAuthUserCredentialStatus: vi.fn(),
}));

vi.mock("@/utils/mcpTokenStore", () => ({
  isTokenValid: vi.fn(),
  getToken: vi.fn(),
  removeToken: vi.fn(),
}));

const { toolsOAuthFlowSpy } = vi.hoisted(() => ({
  toolsOAuthFlowSpy: vi.fn(() => ({ startOAuthFlow: vi.fn(), status: "idle", error: null })),
}));

vi.mock("@/hooks/useToolsOAuthFlow", () => ({
  useToolsOAuthFlow: toolsOAuthFlowSpy,
}));

vi.mock("@/hooks/useUserMcpOAuthFlow", () => ({
  useUserMcpOAuthFlow: () => ({ startOAuthFlow: vi.fn(), status: "idle", error: null }),
}));

const GATE_TEXT = "Authentication required";
// Realistic interactive servers carry a token endpoint; the old heuristic
// (`oauth2 && !tokenUrl`) mislabeled exactly these as M2M. Setting it here is
// what makes the passthrough cases fail on the pre-fix code.
const TOKEN_URL = "https://slack.com/api/oauth.v2.user.access";

const renderViewer = (props: Record<string, unknown>, searchParams = "", onUrlUpdate?: OnUrlUpdateFunction) =>
  renderWithProviders(
    <MCPToolsViewer
      serverId="srv-1"
      accessToken="litellm-key"
      userRole="admin"
      userID="tin@berri.ai"
      serverAlias="slack"
      auth_type="oauth2"
      tokenUrl={TOKEN_URL}
      {...props}
    />,
    { searchParams, onUrlUpdate },
  );

beforeEach(() => {
  testQueryClient.clear();
});

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

const makeTool = (name: string, description: string): MCPTool => ({
  name,
  description,
  inputSchema: { type: "object", properties: {} },
  mcp_info: { server_name: "slack" },
});

const M2M = { oauth2_flow: "client_credentials", delegate_auth_to_upstream: false };

describe("MCPToolsViewer URL state", () => {
  beforeEach(() => {
    vi.mocked(listMCPTools)
      .mockReset()
      .mockResolvedValue({
        tools: [makeTool("search_docs", "Search the docs"), makeTool("fetch_page", "Fetch one page")],
        error: null,
      });
    vi.mocked(callMCPTool)
      .mockReset()
      .mockResolvedValue({ content: [{ type: "text", text: "tool output" }] });
  });

  const toolListItem = (name: string) => screen.getByRole("heading", { level: 4, name });

  it("opens the tool named in the URL and calls that tool", async () => {
    renderViewer(M2M, "?tool=fetch_page");

    expect(await screen.findByText("Input Parameters")).toBeInTheDocument();
    expect(screen.queryByText("Select a Tool to Test")).not.toBeInTheDocument();

    await userEvent.click(screen.getByRole("button", { name: "Call Tool" }));

    await waitFor(() =>
      expect(vi.mocked(callMCPTool)).toHaveBeenCalledWith(
        "litellm-key",
        "srv-1",
        "fetch_page",
        expect.anything(),
        expect.anything(),
      ),
    );
    expect(await screen.findByText("Tool executed successfully")).toBeInTheDocument();
  });

  it("shows the empty playground for a tool the server does not list", async () => {
    renderViewer(M2M, "?tool=gone");

    await screen.findByRole("heading", { level: 4, name: "search_docs" });
    expect(screen.getByText("Select a Tool to Test")).toBeInTheDocument();
  });

  it("pushes the clicked tool, clears it on close, and drops the previous result", async () => {
    const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
    renderViewer(M2M, "?server=srv-1&tool=search_docs", onUrlUpdate);

    await userEvent.click(await screen.findByRole("button", { name: "Call Tool" }));
    expect(await screen.findByText("Tool executed successfully")).toBeInTheDocument();

    await userEvent.click(toolListItem("fetch_page"));

    const picked = onUrlUpdate.mock.calls.at(-1)?.[0];
    expect(picked?.searchParams.get("tool")).toBe("fetch_page");
    expect(picked?.searchParams.get("server")).toBe("srv-1");
    expect(picked?.options.history).toBe("push");
    expect(await screen.findByText("Ready to Call Tool")).toBeInTheDocument();
    expect(screen.queryByText("Tool executed successfully")).not.toBeInTheDocument();

    await userEvent.click(screen.getByRole("button", { name: "Close" }));

    expect(onUrlUpdate.mock.calls.at(-1)?.[0].searchParams.has("tool")).toBe(false);
    expect(onUrlUpdate.mock.calls.at(-1)?.[0].options.history).toBe("push");
    expect(await screen.findByText("Select a Tool to Test")).toBeInTheDocument();
  });

  it("clears the previous result when the selected tool is picked again", async () => {
    renderViewer(M2M, "?tool=search_docs");

    await userEvent.click(await screen.findByRole("button", { name: "Call Tool" }));
    expect(await screen.findByText("Tool executed successfully")).toBeInTheDocument();

    await userEvent.click(toolListItem("search_docs"));

    expect(await screen.findByText("Ready to Call Tool")).toBeInTheDocument();
  });

  it("filters the tool list by the search in the URL and writes what the user types", async () => {
    const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
    renderViewer(M2M, "?tool_search=fetch", onUrlUpdate);

    expect(await screen.findByRole("heading", { level: 4, name: "fetch_page" })).toBeInTheDocument();
    expect(screen.queryByRole("heading", { level: 4, name: "search_docs" })).not.toBeInTheDocument();
    const searchBox = screen.getByPlaceholderText("Search tools...");
    expect(searchBox).toHaveValue("fetch");

    fireEvent.change(searchBox, { target: { value: "docs" } });

    await waitFor(() => expect(onUrlUpdate.mock.calls.at(-1)?.[0].searchParams.get("tool_search")).toBe("docs"));
    expect(toolListItem("search_docs")).toBeInTheDocument();
    expect(screen.queryByRole("heading", { level: 4, name: "fetch_page" })).not.toBeInTheDocument();

    fireEvent.change(searchBox, { target: { value: "" } });

    await waitFor(() => expect(onUrlUpdate.mock.calls.at(-1)?.[0].searchParams.has("tool_search")).toBe(false));
  });
});
