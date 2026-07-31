import { describe, it, expect, vi, beforeEach } from "vitest";
import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { renderWithProviders, testQueryClient } from "../../../tests/test-utils";
import MCPToolPermissions from "./MCPToolPermissions";
import * as networking from "../networking";
import { NO_MCP_SERVERS_SENTINEL } from "../mcp_tools/constants";

vi.mock("../networking");

describe("MCPToolPermissions", () => {
  const mockAccessToken = "test-token";
  const mockServerId = "server-123";
  const mockServerName = "Test MCP Server";

  beforeEach(() => {
    vi.clearAllMocks();
    testQueryClient.clear();
    vi.mocked(networking.fetchMCPToolsets).mockResolvedValue([]);
  });

  it("should update tool permissions when user selects a tool", async () => {
    /**
     * Tests that clicking a tool checkbox calls onChange with updated permissions.
     * Pre-populates toolPermissions so the auto-populate logic on fetch is skipped,
     * and switches to flat view for predictable checkbox ordering.
     */
    const mockOnChange = vi.fn();
    const mockTools = [
      { name: "read_wiki_structure", description: "Get documentation topics" },
      { name: "read_wiki_contents", description: "View documentation" },
      { name: "ask_question", description: "Ask questions" },
    ];

    // Mock fetchMCPServers to return server details
    vi.mocked(networking.fetchMCPServers).mockResolvedValue([
      {
        server_id: mockServerId,
        server_name: mockServerName,
        alias: mockServerName,
      },
    ]);

    // Mock listMCPTools to return tools for the server
    vi.mocked(networking.listMCPTools).mockResolvedValue({
      tools: mockTools,
      error: false,
    });

    // Pre-populate with all tools selected so auto-populate doesn't fire
    renderWithProviders(
      <MCPToolPermissions
        accessToken={mockAccessToken}
        selectedServers={[mockServerId]}
        toolPermissions={{ [mockServerId]: ["read_wiki_structure", "read_wiki_contents", "ask_question"] }}
        onChange={mockOnChange}
      />,
    );

    // Wait for server and tools to load
    await waitFor(() => {
      expect(screen.getByText(mockServerName)).toBeInTheDocument();
    });

    await waitFor(() => {
      expect(screen.getByText("read_wiki_structure")).toBeInTheDocument();
    });

    // Switch to Flat List view for predictable checkbox ordering
    const flatListOption = screen.getByText("Flat List");
    await userEvent.click(flatListOption);

    // Click the first checkbox to deselect read_wiki_structure
    const checkboxes = screen.getAllByRole("checkbox");
    await userEvent.click(checkboxes[0]);

    // Verify onChange was called with read_wiki_structure removed
    expect(mockOnChange).toHaveBeenCalledWith({
      [mockServerId]: ["read_wiki_contents", "ask_question"],
    });

    // Verify API calls
    // Note: useMCPServers uses useAuthorized() internally, which returns "123" from global mock
    expect(networking.fetchMCPServers).toHaveBeenCalledWith("123", undefined);
    // listMCPTools uses the accessToken prop directly
    expect(networking.listMCPTools).toHaveBeenCalledWith(mockAccessToken, mockServerId);
  });

  it("should select all tools when Select All button is clicked", async () => {
    const mockOnChange = vi.fn();
    const mockTools = [
      { name: "read_wiki_structure", description: "Get documentation topics" },
      { name: "read_wiki_contents", description: "View documentation" },
      { name: "ask_question", description: "Ask questions" },
    ];

    // Mock fetchMCPServers to return server details
    vi.mocked(networking.fetchMCPServers).mockResolvedValue([
      {
        server_id: mockServerId,
        server_name: mockServerName,
        alias: mockServerName,
      },
    ]);

    // Mock listMCPTools to return tools for the server
    vi.mocked(networking.listMCPTools).mockResolvedValue({
      tools: mockTools,
      error: false,
    });

    renderWithProviders(
      <MCPToolPermissions
        accessToken={mockAccessToken}
        selectedServers={[mockServerId]}
        toolPermissions={{}}
        onChange={mockOnChange}
      />,
    );

    // Wait for server and tools to load
    await waitFor(() => {
      expect(screen.getByText(mockServerName)).toBeInTheDocument();
    });

    await waitFor(() => {
      expect(screen.getByText("read_wiki_structure")).toBeInTheDocument();
    });

    // Click the Select All button
    const selectAllButton = screen.getByRole("button", { name: "Select All" });
    await userEvent.click(selectAllButton);

    // Verify onChange was called with all tools selected
    expect(mockOnChange).toHaveBeenCalledWith({
      [mockServerId]: ["read_wiki_structure", "read_wiki_contents", "ask_question"],
    });
  });

  it("should deselect all tools when Deselect All button is clicked", async () => {
    const mockOnChange = vi.fn();
    const mockTools = [
      { name: "read_wiki_structure", description: "Get documentation topics" },
      { name: "read_wiki_contents", description: "View documentation" },
      { name: "ask_question", description: "Ask questions" },
    ];

    // Mock fetchMCPServers to return server details
    vi.mocked(networking.fetchMCPServers).mockResolvedValue([
      {
        server_id: mockServerId,
        server_name: mockServerName,
        alias: mockServerName,
      },
    ]);

    // Mock listMCPTools to return tools for the server
    vi.mocked(networking.listMCPTools).mockResolvedValue({
      tools: mockTools,
      error: false,
    });

    renderWithProviders(
      <MCPToolPermissions
        accessToken={mockAccessToken}
        selectedServers={[mockServerId]}
        toolPermissions={{ [mockServerId]: ["read_wiki_structure", "read_wiki_contents"] }}
        onChange={mockOnChange}
      />,
    );

    // Wait for server and tools to load
    await waitFor(() => {
      expect(screen.getByText(mockServerName)).toBeInTheDocument();
    });

    await waitFor(() => {
      expect(screen.getByText("read_wiki_structure")).toBeInTheDocument();
    });

    // Click the Deselect All button
    const deselectAllButton = screen.getByRole("button", { name: "Deselect All" });
    await userEvent.click(deselectAllButton);

    // Verify onChange was called with no tools selected
    expect(mockOnChange).toHaveBeenCalledWith({
      [mockServerId]: [],
    });
  });

  describe("servers reached indirectly", () => {
    const groupServer = {
      server_id: "srv-group-1",
      server_name: "Group Server",
      alias: "Group Server",
      mcp_access_groups: ["production-group"],
    };
    const groupTools = [
      { name: "list_issues", description: "List issues" },
      { name: "delete_issue", description: "Delete an issue" },
    ];

    it("renders the tool matrix for a server granted only through an access group", async () => {
      vi.mocked(networking.fetchMCPServers).mockResolvedValue([groupServer]);
      vi.mocked(networking.fetchMCPToolsets).mockResolvedValue([]);
      vi.mocked(networking.listMCPTools).mockResolvedValue({ tools: groupTools, error: false });

      const mockOnChange = vi.fn();
      renderWithProviders(
        <MCPToolPermissions
          accessToken={mockAccessToken}
          selectedServers={[]}
          selectedAccessGroups={["production-group"]}
          toolPermissions={{}}
          onChange={mockOnChange}
        />,
      );

      expect(await screen.findByText("Group Server")).toBeInTheDocument();
      expect(await screen.findByText("list_issues")).toBeInTheDocument();
      expect(screen.getByText("delete_issue")).toBeInTheDocument();
      expect(networking.listMCPTools).toHaveBeenCalledWith(mockAccessToken, groupServer.server_id);
    });

    it("marks an access-group server as inherited and leaves a directly selected one unmarked", async () => {
      const directServer = { server_id: "srv-direct-1", server_name: "Direct Server", alias: "Direct Server" };
      vi.mocked(networking.fetchMCPServers).mockResolvedValue([directServer, groupServer]);
      vi.mocked(networking.fetchMCPToolsets).mockResolvedValue([]);
      vi.mocked(networking.listMCPTools).mockResolvedValue({ tools: groupTools, error: false });

      renderWithProviders(
        <MCPToolPermissions
          accessToken={mockAccessToken}
          selectedServers={[directServer.server_id]}
          selectedAccessGroups={["production-group"]}
          toolPermissions={{ [directServer.server_id]: ["list_issues"], [groupServer.server_id]: ["list_issues"] }}
          onChange={vi.fn()}
        />,
      );

      expect(await screen.findByText("Direct Server")).toBeInTheDocument();
      expect(await screen.findByText("Group Server")).toBeInTheDocument();
      expect(screen.getByText("Via access group: production-group")).toBeInTheDocument();
      expect(screen.queryAllByText(/^Via /)).toHaveLength(1);
    });

    it("renders a toolset server as inherited from that toolset", async () => {
      const toolsetServer = { server_id: "srv-toolset-1", server_name: "Toolset Server", alias: "Toolset Server" };
      vi.mocked(networking.fetchMCPServers).mockResolvedValue([toolsetServer]);
      vi.mocked(networking.fetchMCPToolsets).mockResolvedValue([
        {
          toolset_id: "ts-1",
          toolset_name: "Support Toolset",
          tools: [{ server_id: toolsetServer.server_id, tool_name: "list_issues" }],
        },
      ]);
      vi.mocked(networking.listMCPTools).mockResolvedValue({ tools: groupTools, error: false });

      renderWithProviders(
        <MCPToolPermissions
          accessToken={mockAccessToken}
          selectedServers={[]}
          selectedToolsets={["ts-1"]}
          toolPermissions={{}}
          onChange={vi.fn()}
        />,
      );

      expect(await screen.findByText("Toolset Server")).toBeInTheDocument();
      expect(screen.getByText("Via toolset: Support Toolset")).toBeInTheDocument();
    });

    it("does not write a default allowlist for an inherited server", async () => {
      vi.mocked(networking.fetchMCPServers).mockResolvedValue([groupServer]);
      vi.mocked(networking.fetchMCPToolsets).mockResolvedValue([]);
      vi.mocked(networking.listMCPTools).mockResolvedValue({ tools: groupTools, error: false });

      const mockOnChange = vi.fn();
      renderWithProviders(
        <MCPToolPermissions
          accessToken={mockAccessToken}
          selectedServers={[]}
          selectedAccessGroups={["production-group"]}
          toolPermissions={{}}
          onChange={mockOnChange}
        />,
      );

      expect(await screen.findByText("list_issues")).toBeInTheDocument();
      expect(mockOnChange).not.toHaveBeenCalled();
    });

    it("keeps blocking delete tools by default for a directly selected server", async () => {
      const directServer = { server_id: "srv-direct-1", server_name: "Direct Server", alias: "Direct Server" };
      vi.mocked(networking.fetchMCPServers).mockResolvedValue([directServer]);
      vi.mocked(networking.fetchMCPToolsets).mockResolvedValue([]);
      vi.mocked(networking.listMCPTools).mockResolvedValue({ tools: groupTools, error: false });

      const mockOnChange = vi.fn();
      renderWithProviders(
        <MCPToolPermissions
          accessToken={mockAccessToken}
          selectedServers={[directServer.server_id]}
          toolPermissions={{}}
          onChange={mockOnChange}
        />,
      );

      await waitFor(() => {
        expect(mockOnChange).toHaveBeenCalledWith({ [directServer.server_id]: ["list_issues"] });
      });
    });

    it("shows a server that only a stale tool-permission entry still entitles", async () => {
      vi.mocked(networking.fetchMCPServers).mockResolvedValue([groupServer]);
      vi.mocked(networking.fetchMCPToolsets).mockResolvedValue([]);
      vi.mocked(networking.listMCPTools).mockResolvedValue({ tools: groupTools, error: false });

      renderWithProviders(
        <MCPToolPermissions
          accessToken={mockAccessToken}
          selectedServers={[]}
          selectedAccessGroups={[]}
          toolPermissions={{ [groupServer.server_id]: ["list_issues"] }}
          onChange={vi.fn()}
        />,
      );

      expect(await screen.findByText("Group Server")).toBeInTheDocument();
      expect(screen.getByText("Via tool permissions")).toBeInTheDocument();
    });

    it("shows nothing for a principal blocked from every MCP server", async () => {
      vi.mocked(networking.fetchMCPServers).mockResolvedValue([groupServer]);
      vi.mocked(networking.fetchMCPToolsets).mockResolvedValue([]);
      vi.mocked(networking.listMCPTools).mockResolvedValue({ tools: groupTools, error: false });

      const { container } = renderWithProviders(
        <MCPToolPermissions
          accessToken={mockAccessToken}
          selectedServers={[NO_MCP_SERVERS_SENTINEL]}
          toolPermissions={{ [groupServer.server_id]: ["list_issues"] }}
          onChange={vi.fn()}
        />,
      );

      expect(container).toBeEmptyDOMElement();
      expect(networking.listMCPTools).not.toHaveBeenCalled();
    });

    it("warns instead of showing no inherited servers when the server list cannot be loaded", async () => {
      vi.mocked(networking.fetchMCPServers).mockRejectedValue(new Error("boom"));
      vi.mocked(networking.fetchMCPToolsets).mockResolvedValue([]);

      renderWithProviders(
        <MCPToolPermissions
          accessToken={mockAccessToken}
          selectedServers={[]}
          selectedAccessGroups={["production-group"]}
          toolPermissions={{}}
          onChange={vi.fn()}
        />,
      );

      expect(await screen.findByText("Unable to load MCP servers")).toBeInTheDocument();
    });

    it("warns when the selected toolsets cannot be resolved to servers", async () => {
      vi.mocked(networking.fetchMCPServers).mockResolvedValue([]);
      vi.mocked(networking.fetchMCPToolsets).mockRejectedValue(new Error("boom"));

      renderWithProviders(
        <MCPToolPermissions
          accessToken={mockAccessToken}
          selectedServers={[]}
          selectedToolsets={["ts-1"]}
          toolPermissions={{}}
          onChange={vi.fn()}
        />,
      );

      expect(await screen.findByText("Unable to load toolsets")).toBeInTheDocument();
    });
  });

  describe("grants keyed by server name", () => {
    const namedServer = {
      server_id: "1f4bd6c1-0000-4000-8000-000000000001",
      server_name: "github_mcp",
      alias: "GitHub",
    };
    const namedTools = [
      { name: "list_issues", description: "List issues" },
      { name: "delete_issue", description: "Delete an issue" },
    ];

    it("renders the tool matrix for a grant that names the server instead of its id", async () => {
      vi.mocked(networking.fetchMCPServers).mockResolvedValue([namedServer]);
      vi.mocked(networking.fetchMCPToolsets).mockResolvedValue([]);
      vi.mocked(networking.listMCPTools).mockResolvedValue({ tools: namedTools, error: false });

      renderWithProviders(
        <MCPToolPermissions
          accessToken={mockAccessToken}
          selectedServers={["github_mcp"]}
          toolPermissions={{ github_mcp: ["list_issues"] }}
          onChange={vi.fn()}
        />,
      );

      expect(await screen.findByText("github_mcp")).toBeInTheDocument();
      expect(await screen.findByText("list_issues")).toBeInTheDocument();
      expect(screen.getByText("delete_issue")).toBeInTheDocument();
      expect(networking.listMCPTools).toHaveBeenCalledWith(mockAccessToken, namedServer.server_id);
    });

    it("writes an edit back to the name key instead of adding a second id-keyed entry", async () => {
      vi.mocked(networking.fetchMCPServers).mockResolvedValue([namedServer]);
      vi.mocked(networking.fetchMCPToolsets).mockResolvedValue([]);
      vi.mocked(networking.listMCPTools).mockResolvedValue({ tools: namedTools, error: false });

      const mockOnChange = vi.fn();
      renderWithProviders(
        <MCPToolPermissions
          accessToken={mockAccessToken}
          selectedServers={["github_mcp"]}
          toolPermissions={{ github_mcp: ["list_issues"] }}
          onChange={mockOnChange}
        />,
      );

      expect(await screen.findByText("list_issues")).toBeInTheDocument();
      await userEvent.click(screen.getByRole("button", { name: "Deselect All" }));

      expect(mockOnChange).toHaveBeenCalledWith({ github_mcp: [] });
    });
  });

  describe("a server named by several equivalent keys", () => {
    const namedServer = {
      server_id: "1f4bd6c1-0000-4000-8000-000000000001",
      server_name: "github_mcp",
      alias: "GitHub",
      mcp_access_groups: ["production-group"],
    };
    const namedTools = [
      { name: "list_issues", description: "List issues" },
      { name: "create_issue", description: "Open an issue" },
      { name: "delete_issue", description: "Delete an issue" },
    ];

    const renderWithBothKeys = (onChange: () => void) =>
      renderWithProviders(
        <MCPToolPermissions
          accessToken={mockAccessToken}
          selectedServers={[namedServer.server_id]}
          toolPermissions={{ [namedServer.server_id]: ["list_issues"], github_mcp: ["create_issue"] }}
          onChange={onChange}
        />,
      );

    beforeEach(() => {
      vi.mocked(networking.fetchMCPServers).mockResolvedValue([namedServer]);
      vi.mocked(networking.fetchMCPToolsets).mockResolvedValue([]);
      vi.mocked(networking.listMCPTools).mockResolvedValue({ tools: namedTools, error: false });
    });

    it("renders one card showing the union both keys grant", async () => {
      renderWithBothKeys(vi.fn());

      expect(await screen.findByText("github_mcp")).toBeInTheDocument();
      expect(screen.getAllByText("github_mcp")).toHaveLength(1);
      expect(await screen.findByText("list_issues")).toBeInTheDocument();

      // Flat view keeps checkbox order identical to the fetched tool order.
      await userEvent.click(screen.getByText("Flat List"));
      const [listIssues, createIssue, deleteIssue] = screen.getAllByRole("checkbox");
      expect(listIssues).toBeChecked();
      expect(createIssue).toBeChecked();
      expect(deleteIssue).not.toBeChecked();
    });

    it("removes a deselected tool from every equivalent key, leaving one entry for the server", async () => {
      const mockOnChange = vi.fn();
      renderWithBothKeys(mockOnChange);

      expect(await screen.findByText("list_issues")).toBeInTheDocument();
      await userEvent.click(screen.getByText("Flat List"));
      await userEvent.click(screen.getAllByRole("checkbox")[0]);

      const written = mockOnChange.mock.calls.at(-1)?.[0] as Record<string, string[]>;
      expect(Object.keys(written)).toEqual([namedServer.server_id]);
      expect(written[namedServer.server_id]).not.toContain("list_issues");
      expect(written[namedServer.server_id]).toContain("create_issue");
    });

    // Both catalog orders, because a name resolves to two servers here and a first-match
    // implementation is only wrong in one of them.
    it.each([
      { label: "edited server first", editedFirst: true },
      { label: "twin first", editedFirst: false },
    ])(
      "says on the card when a key names another server too, since its tools cannot be revoked here ($label)",
      async ({ editedFirst }) => {
        const twin = { server_id: "1f4bd6c1-0000-4000-8000-000000000002", server_name: "github_mcp", alias: "Twin" };
        vi.mocked(networking.fetchMCPServers).mockResolvedValue(
          editedFirst ? [namedServer, twin] : [twin, namedServer],
        );

        renderWithProviders(
          <MCPToolPermissions
            accessToken={mockAccessToken}
            selectedServers={[namedServer.server_id]}
            toolPermissions={{ [namedServer.server_id]: ["list_issues"], github_mcp: ["create_issue"] }}
            onChange={vi.fn()}
          />,
        );

        expect(
          await screen.findByText(
            'Also granted by "github_mcp", which names another server too. Those tools stay allowed here until the servers no longer share that name',
          ),
        ).toBeInTheDocument();
      },
    );

    it("says nothing about shared names when every key names one server", async () => {
      renderWithBothKeys(vi.fn());

      expect(await screen.findByText("github_mcp")).toBeInTheDocument();
      expect(screen.queryByText(/names another server too/)).not.toBeInTheDocument();
    });

    it("badges the server once, by its strongest grant, when a key and a group both name it", async () => {
      renderWithProviders(
        <MCPToolPermissions
          accessToken={mockAccessToken}
          selectedServers={[]}
          selectedAccessGroups={["production-group"]}
          toolPermissions={{ github_mcp: ["list_issues"] }}
          onChange={vi.fn()}
        />,
      );

      expect(await screen.findByText("github_mcp")).toBeInTheDocument();
      expect(screen.getByText("Via access group: production-group")).toBeInTheDocument();
      expect(screen.queryByText("Via tool permissions")).not.toBeInTheDocument();
      expect(screen.queryAllByText(/^Via /)).toHaveLength(1);
    });
  });
});
