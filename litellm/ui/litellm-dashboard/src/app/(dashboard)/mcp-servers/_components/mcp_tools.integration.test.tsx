import userEvent from "@testing-library/user-event";
import { parseAsString, useQueryState } from "nuqs";
import type { OnUrlUpdateFunction } from "nuqs/adapters/testing";
import { describe, expect, it, vi, beforeEach } from "vitest";
import MCPToolsViewer from "./mcp_tools";
import { callMCPTool, listMCPTools } from "@/components/networking";
import type { MCPTool } from "@/components/mcp_tools/types";
import { act, fireEvent, renderWithProviders, screen, testQueryClient, waitFor } from "@/../tests/test-utils";

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

const TOKEN_URL = "https://slack.com/api/oauth.v2.user.access";

const viewer = (props: Record<string, unknown>) => (
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
);

const renderViewer = (props: Record<string, unknown>, searchParams = "", onUrlUpdate?: OnUrlUpdateFunction) =>
  renderWithProviders(viewer(props), { searchParams, onUrlUpdate });

beforeEach(() => {
  testQueryClient.clear();
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

  it("shows the empty playground for a tool the server does not list and keeps the link", async () => {
    const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
    renderViewer(M2M, "?tool=gone", onUrlUpdate);

    await screen.findByRole("heading", { level: 4, name: "search_docs" });
    expect(screen.getByText("Select a Tool to Test")).toBeInTheDocument();
    expect(onUrlUpdate.mock.calls.some(([update]) => !update.searchParams.has("tool"))).toBe(false);
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

  it("hides the previous tool's result when the URL switches to another tool", async () => {
    const ToolLink = ({ name }: { name: string }) => {
      const [, setTool] = useQueryState("tool", parseAsString.withOptions({ history: "push" }));
      return <button onClick={() => void setTool(name)}>go to {name}</button>;
    };
    renderWithProviders(
      <>
        {viewer(M2M)}
        <ToolLink name="fetch_page" />
      </>,
      { searchParams: "?tool=search_docs" },
    );

    await userEvent.click(await screen.findByRole("button", { name: "Call Tool" }));
    expect(await screen.findByText("Tool executed successfully")).toBeInTheDocument();

    await userEvent.click(screen.getByRole("button", { name: "go to fetch_page" }));

    expect(await screen.findByText("Ready to Call Tool")).toBeInTheDocument();
    expect(screen.queryByText("Tool executed successfully")).not.toBeInTheDocument();
  });

  it("clears the previous result without a new history entry when the selected tool is picked again", async () => {
    const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
    renderViewer(M2M, "?tool=search_docs", onUrlUpdate);

    await userEvent.click(await screen.findByRole("button", { name: "Call Tool" }));
    expect(await screen.findByText("Tool executed successfully")).toBeInTheDocument();

    await userEvent.click(toolListItem("search_docs"));

    expect(await screen.findByText("Ready to Call Tool")).toBeInTheDocument();
    await act(() => new Promise((resolve) => setTimeout(resolve, 100)));

    await userEvent.click(toolListItem("fetch_page"));
    await waitFor(() => expect(onUrlUpdate).toHaveBeenCalled());
    expect(onUrlUpdate.mock.calls.map(([update]) => update.searchParams.get("tool"))).toEqual(["fetch_page"]);
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
    expect(onUrlUpdate.mock.calls.at(-1)?.[0].options.history).toBe("replace");
    expect(toolListItem("search_docs")).toBeInTheDocument();
    expect(screen.queryByRole("heading", { level: 4, name: "fetch_page" })).not.toBeInTheDocument();

    fireEvent.change(searchBox, { target: { value: "" } });

    await waitFor(() => expect(onUrlUpdate.mock.calls.at(-1)?.[0].searchParams.has("tool_search")).toBe(false));
  });
});
