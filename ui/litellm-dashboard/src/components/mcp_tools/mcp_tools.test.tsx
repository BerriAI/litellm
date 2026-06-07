import { render, screen, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { describe, expect, it, vi, beforeEach } from "vitest";
import MCPToolsViewer from "./mcp_tools";
import { listMCPTools } from "../networking";
import { isTokenValid, getToken } from "@/utils/mcpTokenStore";

vi.mock("../networking", () => ({
  listMCPTools: vi.fn(),
  callMCPTool: vi.fn(),
}));

vi.mock("@/utils/mcpTokenStore", () => ({
  isTokenValid: vi.fn(),
  getToken: vi.fn(),
  removeToken: vi.fn(),
}));

vi.mock("@/hooks/useToolsOAuthFlow", () => ({
  useToolsOAuthFlow: () => ({ startOAuthFlow: vi.fn(), status: "idle", error: null }),
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

describe("MCPToolsViewer auth gate routing", () => {
  beforeEach(() => {
    vi.mocked(listMCPTools).mockReset().mockResolvedValue({ tools: [], error: null });
    vi.mocked(isTokenValid).mockReset().mockReturnValue(false);
    vi.mocked(getToken)
      .mockReset()
      .mockReturnValue(undefined as any);
  });

  it("shows the Authorize gate for a passthrough server with a token endpoint and does not list tools", async () => {
    renderViewer({ oauth2_flow: null, delegate_auth_to_upstream: true });

    expect(await screen.findByText(GATE_TEXT)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Authorize" })).toBeInTheDocument();
    expect(vi.mocked(listMCPTools)).not.toHaveBeenCalled();
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

  it("does not gate an OBO server with a token endpoint; lists with the LiteLLM key and no x-mcp header", async () => {
    renderViewer({ oauth2_flow: null, delegate_auth_to_upstream: false });

    await waitFor(() => expect(vi.mocked(listMCPTools)).toHaveBeenCalledWith("litellm-key", "srv-1", undefined));
    expect(screen.queryByText(GATE_TEXT)).not.toBeInTheDocument();
  });

  it("does not gate an M2M server; lists with the LiteLLM key", async () => {
    renderViewer({ oauth2_flow: "client_credentials", delegate_auth_to_upstream: false });

    await waitFor(() => expect(vi.mocked(listMCPTools)).toHaveBeenCalledWith("litellm-key", "srv-1", undefined));
    expect(screen.queryByText(GATE_TEXT)).not.toBeInTheDocument();
  });
});
