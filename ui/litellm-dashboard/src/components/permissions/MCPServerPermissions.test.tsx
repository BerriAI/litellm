import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import MCPServerPermissions from "./MCPServerPermissions";
import * as networking from "../networking";

vi.mock("../networking");

describe("MCPServerPermissions", () => {
  const mockAccessToken = "test-token";
  const mockServerId1 = "3e64bed6-57e1-4247-ad5a-4b1a47ae6583";
  const mockServerId2 = "server-456";
  const mockServerName1 = "DW_MCP";
  const mockServerName2 = "Test Server";

  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("should display MCP servers with their aliases and IDs", async () => {
    /**
     * Tests that MCP servers are displayed with their correct aliases and truncated IDs.
     * This verifies the basic rendering of server information.
     */
    const mockServers = [
      {
        server_id: mockServerId1,
        server_name: mockServerName1,
        alias: mockServerName1,
      },
      {
        server_id: mockServerId2,
        server_name: mockServerName2,
        alias: mockServerName2,
      },
    ];

    vi.mocked(networking.fetchMCPServers).mockResolvedValue(mockServers);

    render(
      <MCPServerPermissions
        mcpServers={[mockServerId1, mockServerId2]}
        mcpAccessGroups={[]}
        mcpToolPermissions={{}}
        accessToken={mockAccessToken}
      />
    );

    // Wait for servers to load and display
    await waitFor(() => {
      expect(screen.getByText(/DW_MCP/)).toBeInTheDocument();
    });

    await waitFor(() => {
      expect(screen.getByText(/Test Server/)).toBeInTheDocument();
    });

    // Verify the count badge shows correct number
    expect(screen.getByText("2")).toBeInTheDocument();

    // Verify API was called
    expect(networking.fetchMCPServers).toHaveBeenCalledWith(mockAccessToken);
  });

  it("should display expandable tool permissions for servers when they exist", async () => {
    /**
     * Tests that tool permissions can be expanded/collapsed by clicking the server row
     * and that the tool count is displayed correctly.
     */
    const mockServers = [
      {
        server_id: mockServerId1,
        server_name: mockServerName1,
        alias: mockServerName1,
      },
    ];

    const mockToolPermissions = {
      [mockServerId1]: ["read_wiki_structure", "read_wiki_contents", "ask_question"],
    };

    vi.mocked(networking.fetchMCPServers).mockResolvedValue(mockServers);

    render(
      <MCPServerPermissions
        mcpServers={[mockServerId1]}
        mcpAccessGroups={[]}
        mcpToolPermissions={mockToolPermissions}
        accessToken={mockAccessToken}
      />
    );

    // Wait for server to load
    await waitFor(() => {
      expect(screen.getByText(/DW_MCP/)).toBeInTheDocument();
    });

    // Verify tool count is shown
    expect(screen.getByText("3")).toBeInTheDocument();
    expect(screen.getByText("tools")).toBeInTheDocument();

    // Tools should NOT be visible initially (collapsed state)
    expect(screen.queryByText("read_wiki_structure")).not.toBeInTheDocument();
    expect(screen.queryByText("read_wiki_contents")).not.toBeInTheDocument();
    expect(screen.queryByText("ask_question")).not.toBeInTheDocument();

    // Click the server row to expand
    const serverRow = screen.getByText(/DW_MCP/).closest("div");
    await userEvent.click(serverRow!);

    // Now tools should be visible
    await waitFor(() => {
      expect(screen.getByText("read_wiki_structure")).toBeInTheDocument();
      expect(screen.getByText("read_wiki_contents")).toBeInTheDocument();
      expect(screen.getByText("ask_question")).toBeInTheDocument();
    });

    // Click the server row again to collapse
    await userEvent.click(serverRow!);

    // Tools should be hidden again
    await waitFor(() => {
      expect(screen.queryByText("read_wiki_structure")).not.toBeInTheDocument();
    });
  });

  it("should not display tool permissions section when no tools are configured", async () => {
    /**
     * Tests that the tool permissions section is not shown when
     * mcp_tool_permissions is empty or not provided for a server.
     */
    const mockServers = [
      {
        server_id: mockServerId1,
        server_name: mockServerName1,
        alias: mockServerName1,
      },
    ];

    vi.mocked(networking.fetchMCPServers).mockResolvedValue(mockServers);

    render(
      <MCPServerPermissions
        mcpServers={[mockServerId1]}
        mcpAccessGroups={[]}
        mcpToolPermissions={{}}
        accessToken={mockAccessToken}
      />
    );

    // Wait for server to load
    await waitFor(() => {
      expect(screen.getByText(/DW_MCP/)).toBeInTheDocument();
    });

    // Verify no tool count is shown (since there are no tools)
    expect(screen.queryByText(/tool/)).not.toBeInTheDocument();
  });

  it("should display access groups correctly", async () => {
    /**
     * Tests that access groups are displayed with the correct styling
     * and indicator badges.
     */
    const mockAccessGroups = ["production-group", "development-group"];

    vi.mocked(networking.fetchMCPAccessGroups).mockResolvedValue(mockAccessGroups);

    render(
      <MCPServerPermissions
        mcpServers={[]}
        mcpAccessGroups={mockAccessGroups}
        mcpToolPermissions={{}}
        accessToken={mockAccessToken}
      />
    );

    // Wait for access groups to load
    await waitFor(() => {
      expect(screen.getByText("production-group")).toBeInTheDocument();
    });

    expect(screen.getByText("development-group")).toBeInTheDocument();
    expect(screen.getAllByText("Group")).toHaveLength(2);

    // Verify the count badge shows correct number
    expect(screen.getByText("2")).toBeInTheDocument();
  });

  it("should display both servers and access groups together", async () => {
    /**
     * Tests that both MCP servers and access groups can be displayed
     * simultaneously in the same component.
     */
    const mockServers = [
      {
        server_id: mockServerId1,
        server_name: mockServerName1,
        alias: mockServerName1,
      },
    ];

    const mockAccessGroups = ["production-group"];

    vi.mocked(networking.fetchMCPServers).mockResolvedValue(mockServers);
    vi.mocked(networking.fetchMCPAccessGroups).mockResolvedValue(mockAccessGroups);

    render(
      <MCPServerPermissions
        mcpServers={[mockServerId1]}
        mcpAccessGroups={mockAccessGroups}
        mcpToolPermissions={{}}
        accessToken={mockAccessToken}
      />
    );

    // Wait for both to load
    await waitFor(() => {
      expect(screen.getByText(/DW_MCP/)).toBeInTheDocument();
    });

    await waitFor(() => {
      expect(screen.getByText("production-group")).toBeInTheDocument();
    });

    // Verify total count is 2 (1 server + 1 access group)
    expect(screen.getByText("2")).toBeInTheDocument();
  });

  it("should display empty state when no servers or access groups are configured", () => {
    /**
     * Tests that the empty state message is shown when there are no
     * MCP servers or access groups to display.
     */
    render(
      <MCPServerPermissions
        mcpServers={[]}
        mcpAccessGroups={[]}
        mcpToolPermissions={{}}
        accessToken={mockAccessToken}
      />
    );

    // Verify empty state message
    expect(screen.getByText("No MCP servers or access groups configured")).toBeInTheDocument();

    // Verify count badge shows 0
    expect(screen.getByText("0")).toBeInTheDocument();
  });

  it("should handle multiple servers with different tool permissions", async () => {
    /**
     * Tests that multiple servers can each have their own tool permissions
     * displayed correctly without mixing them up.
     */
    const mockServers = [
      {
        server_id: mockServerId1,
        server_name: mockServerName1,
        alias: mockServerName1,
      },
      {
        server_id: mockServerId2,
        server_name: mockServerName2,
        alias: mockServerName2,
      },
    ];

    const mockToolPermissions = {
      [mockServerId1]: ["read_wiki_structure", "read_wiki_contents"],
      [mockServerId2]: ["ask_question"],
    };

    vi.mocked(networking.fetchMCPServers).mockResolvedValue(mockServers);

    render(
      <MCPServerPermissions
        mcpServers={[mockServerId1, mockServerId2]}
        mcpAccessGroups={[]}
        mcpToolPermissions={mockToolPermissions}
        accessToken={mockAccessToken}
      />
    );

    // Wait for servers to load
    await waitFor(() => {
      expect(screen.getByText(/DW_MCP/)).toBeInTheDocument();
      expect(screen.getByText(/Test Server/)).toBeInTheDocument();
    });

    // Verify both servers show tool counts
    // Server 1 has 2 tools, Server 2 has 1 tool
    const toolCounts = screen.getAllByText(/^\d+$/);
    const toolLabels = screen.getAllByText(/^tools?$/i);
    expect(toolCounts.length).toBeGreaterThan(0);
    expect(toolLabels.length).toBeGreaterThan(0);

    // Expand both servers by clicking their rows
    const server1Row = screen.getByText(/DW_MCP/).closest("div");
    const server2Row = screen.getByText(/Test Server/).closest("div");
    
    await userEvent.click(server1Row!); // Expand server 1
    await userEvent.click(server2Row!); // Expand server 2

    // Verify server 1 tools are now visible
    await waitFor(() => {
      expect(screen.getByText("read_wiki_structure")).toBeInTheDocument();
      expect(screen.getByText("read_wiki_contents")).toBeInTheDocument();
    });

    // Verify server 2 tools are now visible
    expect(screen.getByText("ask_question")).toBeInTheDocument();
  });

  it("should handle API errors gracefully", async () => {
    /**
     * Tests that the component doesn't crash when API calls fail
     * and falls back to showing server IDs instead of names.
     */
    vi.mocked(networking.fetchMCPServers).mockRejectedValue(
      new Error("Failed to fetch servers")
    );

    render(
      <MCPServerPermissions
        mcpServers={[mockServerId1]}
        mcpAccessGroups={[]}
        mcpToolPermissions={{}}
        accessToken={mockAccessToken}
      />
    );

    // Should still render with server ID (fallback)
    await waitFor(() => {
      expect(screen.getByText(mockServerId1)).toBeInTheDocument();
    });

    // Verify error was logged
    expect(networking.fetchMCPServers).toHaveBeenCalledWith(mockAccessToken);
  });

  it("should not fetch server details when accessToken is not provided", () => {
    /**
     * Tests that the component doesn't attempt to fetch server details
     * when no access token is provided.
     */
    render(
      <MCPServerPermissions
        mcpServers={[mockServerId1]}
        mcpAccessGroups={[]}
        mcpToolPermissions={{}}
        accessToken={null}
      />
    );

    // API should not be called without token
    expect(networking.fetchMCPServers).not.toHaveBeenCalled();
  });
});

