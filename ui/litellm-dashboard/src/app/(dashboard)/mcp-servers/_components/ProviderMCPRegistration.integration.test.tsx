import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import type { MCPServer } from "@/components/mcp_tools/types";
import * as networking from "@/components/networking";
import CreateMCPServer from "./CreateMCPServer";
import MCPServerEdit from "./mcp_server_edit";
import { MCPServerView } from "./mcp_server_view";
import MCPServerCard from "./MCPServerCard";
import { selectOption } from "./testUtils";

vi.mock("@/components/networking", () => ({
  createMCPServer: vi.fn(),
  updateMCPServer: vi.fn(),
  fetchOpenAPIRegistry: vi.fn().mockResolvedValue({ apis: [] }),
  listMCPTools: vi.fn().mockResolvedValue({ tools: [] }),
  testMCPToolsListRequest: vi.fn().mockResolvedValue({ tools: [] }),
  getMCPOAuthUserCredentialStatus: vi.fn(),
}));

const server: MCPServer = {
  server_id: "provider-docs",
  execution_mode: "provider",
  server_name: "docs_native",
  url: "https://docs.example.com/mcp",
  allowed_tools: ["search_docs", "fetch_doc"],
  mcp_info: { tool_allowlist_enforced: true },
  created_at: "2026-01-01T00:00:00Z",
  updated_at: "2026-01-01T00:00:00Z",
  created_by: "admin",
  updated_by: "admin",
};

const props = {
  accessToken: "test-token",
  userRole: "Admin",
  userID: "admin",
  availableAccessGroups: ["engineering"],
};

function renderView(isConfig = false) {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={queryClient}>
      <MCPServerView
        {...props}
        mcpServer={{ ...server, is_config: isConfig }}
        isProxyAdmin
        isEditing={false}
        onBack={vi.fn()}
      />
    </QueryClientProvider>,
  );
}

describe("provider registration UI", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(networking.createMCPServer).mockResolvedValue(server);
    vi.mocked(networking.updateMCPServer).mockResolvedValue({ ...server, allowed_tools: [] });
  });

  it("switches from a gateway draft to provider registration without sending connection settings", async () => {
    const onCreateSuccess = vi.fn();
    render(<CreateMCPServer {...props} isModalVisible setModalVisible={vi.fn()} onCreateSuccess={onCreateSuccess} />);
    await selectOption("Transport Type", "Streamable HTTP");
    await selectOption("Authentication", "API Key");
    fireEvent.change(screen.getByLabelText("Authentication Value"), { target: { value: "draft-credential" } });
    await selectOption("Execution mode", "Provider-executed");
    fireEvent.change(screen.getByLabelText("MCP Server Name"), { target: { value: "docs_native" } });
    fireEvent.change(screen.getByLabelText("MCP Server URL"), { target: { value: server.url } });
    fireEvent.change(screen.getByLabelText("Allowed tools"), { target: { value: "search_docs\nfetch_doc" } });
    fireEvent.click(screen.getByRole("button", { name: "Add MCP Server" }));
    await waitFor(() => expect(onCreateSuccess).toHaveBeenCalledWith(server));
    expect(networking.createMCPServer).toHaveBeenCalledExactlyOnceWith("test-token", {
      execution_mode: "provider",
      server_name: "docs_native",
      alias: null,
      description: null,
      url: server.url,
      allowed_tools: ["search_docs", "fetch_doc"],
      mcp_access_groups: [],
      allow_all_keys: false,
      mcp_info: { tool_allowlist_enforced: true },
    });
    expect(networking.testMCPToolsListRequest).not.toHaveBeenCalled();
    expect(screen.queryByLabelText("Authentication Value")).not.toBeInTheDocument();
  });

  it("edits stored tools and can deny all without discovery", async () => {
    const onSuccess = vi.fn();
    render(<MCPServerEdit {...props} mcpServer={server} onCancel={vi.fn()} onSuccess={onSuccess} />);
    expect(screen.getByLabelText("Allowed tools")).toHaveValue("search_docs\nfetch_doc");
    fireEvent.change(screen.getByLabelText("Allowed tools"), { target: { value: "" } });
    fireEvent.click(screen.getByRole("button", { name: "Save Changes" }));
    await waitFor(() => expect(onSuccess).toHaveBeenCalledWith(expect.objectContaining({ allowed_tools: [] })));
    expect(networking.updateMCPServer).toHaveBeenCalledWith(
      "test-token",
      expect.objectContaining({
        server_id: server.server_id,
        allowed_tools: [],
        mcp_info: { tool_allowlist_enforced: true },
      }),
    );
    expect(networking.listMCPTools).not.toHaveBeenCalled();
    expect(networking.testMCPToolsListRequest).not.toHaveBeenCalled();
  });

  it("displays stored tools without fetching live tools or credentials", async () => {
    renderView();
    fireEvent.click(screen.getByRole("tab", { name: "Allowed tools" }));
    expect(await screen.findByRole("cell", { name: "search_docs" })).toBeInTheDocument();
    expect(screen.queryByRole("tab", { name: /User Credentials/i })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /Test Connection/i })).not.toBeInTheDocument();
    expect(networking.listMCPTools).not.toHaveBeenCalled();
    expect(networking.getMCPOAuthUserCredentialStatus).not.toHaveBeenCalled();
  });

  it("keeps config registrations read-only", () => {
    renderView(true);
    fireEvent.click(screen.getByRole("tab", { name: "Settings" }));
    expect(screen.getByRole("button", { name: "Edit Settings" })).toBeDisabled();
    expect(screen.getByText("Defined in config. Edit your YAML configuration to make changes")).toBeInTheDocument();
  });

  it("keeps deletion available but hides gateway connection actions on provider cards", async () => {
    const onDelete = vi.fn();
    render(<MCPServerCard server={server} onClick={vi.fn()} onRecheckHealth={vi.fn()} onDelete={onDelete} />);
    fireEvent.click(screen.getByRole("button", { name: "Server actions" }));
    expect(await screen.findByRole("menuitem", { name: "Delete" })).toBeInTheDocument();
    expect(screen.queryByRole("menuitem", { name: "Test Connection" })).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole("menuitem", { name: "Delete" }));
    expect(onDelete).toHaveBeenCalledOnce();
    expect(screen.getByText("2 allowed tools")).toBeInTheDocument();
  });
});
